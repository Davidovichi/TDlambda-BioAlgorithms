"""
maze_env.py — Shared maze environment for bio-inspired TD(λ) experiments.

CSVs are read from the same folder as this file (experimentos_bio/).
Add more mazes by appending rows to the CSV files — the loader auto-detects count.
"""

import os
import numpy as np
from collections import deque

# ─── Grid constants ───────────────────────────────────────────────────────────
GRID_ROWS = 20
GRID_COLS = 20
START = (0, 0)
GOAL  = (GRID_ROWS - 1, GRID_COLS - 1)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

# ─── CSV paths — inside experimentos_bio/ ─────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(_HERE, 'laberintostrain.csv')
TEST_CSV  = os.path.join(_HERE, 'laberintostest.csv')


# ─── Maze loading ─────────────────────────────────────────────────────────────
def load_mazes(csv_path, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, validate=True):
    """
    Load mazes stacked vertically in a CSV file.
    Auto-detects number of mazes (total_rows // grid_rows).
    """
    arr = np.loadtxt(csv_path, delimiter=',', dtype=int)
    if arr.ndim == 1:          # single row edge-case
        arr = arr.reshape(1, -1)
    if arr.shape[1] != grid_cols:
        raise ValueError(
            f"Expected {grid_cols} columns in {csv_path}, got {arr.shape[1]}"
        )
    n = arr.shape[0] // grid_rows
    if n == 0:
        raise ValueError(f"CSV {csv_path} has fewer than {grid_rows} rows.")
    mazes = [arr[i * grid_rows:(i + 1) * grid_rows] for i in range(n)]
    if validate:
        for idx, m in enumerate(mazes, 1):
            if not has_path(m):
                raise ValueError(
                    f"Maze {idx} in {csv_path} has no valid path START→GOAL."
                )
    return mazes


# ─── Environment primitives ───────────────────────────────────────────────────
def has_path(maze, start=START, goal=GOAL):
    """BFS reachability check."""
    rows, cols = maze.shape
    visited = set()
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            return True
        if s in visited:
            continue
        visited.add(s)
        r, c = s
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and maze[nr, nc] == 1 and (nr, nc) not in visited):
                q.append((nr, nc))
    return False


# ─── Free-cell lookup (memoized per maze) ─────────────────────────────────────
# step()/simulate_sequence() run billions of times. A Python set membership test
# is ~20× faster than a numpy scalar index maze[i, j] and returns the SAME
# transition (a cell is reachable iff it is in-bounds AND free), so this is a
# pure speedup with bit-identical results. Keyed by id(maze); the maze object is
# stored alongside so its id cannot be reused while cached (mazes are never
# mutated and persist for the whole run).
_FREE_CACHE = {}


def _free_cells(maze):
    """Set of (r, c) free cells for `maze`, built once and cached."""
    key = id(maze)
    entry = _FREE_CACHE.get(key)
    if entry is not None and entry[0] is maze:
        return entry[1]
    rows, cols = maze.shape
    free = {(r, c) for r in range(rows) for c in range(cols) if maze[r, c] == 1}
    _FREE_CACHE[key] = (maze, free)
    return free


def step(s, a, maze, goal=GOAL):
    """One transition. Returns (next_state, reward)."""
    if s == goal:
        return s, 0.0
    di, dj = DIRECTIONS[a]
    nxt = (s[0] + di, s[1] + dj)
    s_next = nxt if nxt in _free_cells(maze) else s
    return s_next, (1.0 if s_next == goal else 0.0) #aqui puedo cambiar la recompensa global


def shortest_path(maze, start=START, goal=GOAL):
    """BFS shortest path length; -1 if unreachable."""
    rows, cols = maze.shape
    visited = {start}
    q = deque([(start, 0)])
    while q:
        s, d = q.popleft()
        if s == goal:
            return d
        r, c = s
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and maze[nr, nc] == 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return -1


# ─── Sequence simulation (shared by all planners) ────────────────────────────
def simulate_sequence(start, action_seq, V, maze, gamma, horizon, goal=GOAL,
                      step_cost=0.0, goal_reward=1.0):
    """
    Rollout action_seq; bootstrap with V if goal not reached.

    step_cost / goal_reward hacen que el lookahead simule la MISMA recompensa
    del MDP que se está aprendiendo (ablaciones use_step_cost / use_shaped_reward).
    Con los defaults (0.0, 1.0) reproduce la recompensa cruda del entorno.
    core.run_experiment publica ambos valores en cfg['step_cost'] y
    cfg['goal_reward'] para que los planificadores los propaguen aquí.
    """
    free = _free_cells(maze)          # fetch once; inline transition below
    # Python list of ints indexes ~5× faster than a numpy array in this tight
    # loop; the action values are unchanged.
    seq = action_seq.tolist() if hasattr(action_seq, 'tolist') else action_seq
    s = start
    G = 0.0
    disc = 1.0
    for i in range(min(horizon, len(seq))):
        if s == goal:
            break
        di, dj = DIRECTIONS[seq[i]]
        nxt = (s[0] + di, s[1] + dj)
        s_next = nxt if nxt in free else s
        if s_next == goal:
            r = goal_reward
        else:
            r = -step_cost
        G += disc * r
        disc *= gamma
        s = s_next
    if s != goal:
        G += disc * V.get(s, 0.0)
    return G


# ─── ASCII maze visualizer ────────────────────────────────────────────────────
def print_maze_ascii(maze, V=None, pos=None, start=START, goal=GOAL,
                     v_threshold=0.05):
    """
    Print a compact ASCII view of the maze.
    If V is provided, free cells show their V value with a shading char.
    Symbols:
      S = start,  G = goal,  @ = current position
      █ = wall,   . = free (V<threshold),  numbers 1-9 = V bucket
    """
    rows, cols = maze.shape
    shading = ' .░▒▓'   # 5 levels: very low → high value
    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            cell = (r, c)
            if cell == start:
                ch = 'S'
            elif cell == goal:
                ch = 'G'
            elif pos is not None and cell == pos:
                ch = '@'
            elif maze[r, c] == 0:
                ch = '█'
            elif V is not None:
                v = V.get(cell, 0.0)
                idx = min(int(v / 0.25 * (len(shading) - 1)), len(shading) - 1)
                ch = shading[idx]
            else:
                ch = '.'
            row_chars.append(ch)
        lines.append(' '.join(row_chars))
    print('\n'.join(lines))
    print()
