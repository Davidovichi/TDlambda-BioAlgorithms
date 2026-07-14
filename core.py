"""
core.py — Shared TD(λ) training and test loops for bio-inspired experiments.

Key design decisions
─────────────────────
• V is a single dict created once and mutated across all training mazes,
  so each maze benefits from what was learned in previous ones.
• Sparse eligibility traces: only states with e > 1e-7 are tracked/updated,
  giving O(visited) updates instead of O(all_states) — major speedup on 20×20.
• Two-phase epsilon schedule per maze:
    Phase 1 (first explore_frac episodes): epsilon_start → epsilon_mid  (exploration)
    Phase 2 (last exploit_frac episodes) : epsilon_mid  → 0.0           (exploitation)
• Planner interface: planner_fn(state, V, maze, cfg) → float ∈ [0, 1]
• Shaped reward (ablation): terminal reward decays from 1.0 to 0.1 based on
    steps taken — encourages path efficiency in addition to goal-reaching.
"""

import random
from array import array
import numpy as np
from maze_env import step, GOAL, START, GRID_ROWS, GRID_COLS, print_maze_ascii

_ALL_STATES = [(i, j) for i in range(GRID_ROWS) for j in range(GRID_COLS)]

# Live monitor — set to a file path by run_all.py when LIVE_MONITOR=True.
# Leaving it None costs nothing (the write function returns immediately).
MONITOR_FILE = None

# Checkpoints de V(s) para monitor.py — cada corrida (algoritmo × semilla)
# guarda su tabla V en monitor_VdE/ al completar cada VDE_EVERY laberintos
# (y al terminar el último). Con VDE_EVERY=1 se sobreescribe tras CADA
# laberinto completado (~3 ms por escritura, <0.01% del tiempo de un
# laberinto); sube el valor solo si quieres espaciar los checkpoints.
# Cada corrida escribe SU PROPIO archivo de forma atómica → funciona igual
# en modo secuencial y en paralelo, sin contención.
# Solo LEE V: no toca RNG ni datos — cero efecto en los resultados.
VDE_EVERY = 1


def _write_vde_dump(name, seed, maze_idx, n_mazes, V, maze):
    """Checkpoint de V(s) en monitor_VdE/V_<algo>_seed<seed>.json. Never raises."""
    import json, os, time
    try:
        here    = os.path.dirname(os.path.abspath(__file__))
        vde_dir = os.path.join(here, 'monitor_VdE')
        os.makedirs(vde_dir, exist_ok=True)
        rows, cols = maze.shape
        data = {
            'algo'     : name,
            'seed'     : seed,
            'maze_idx' : maze_idx,
            'n_mazes'  : n_mazes,
            'v_grid'   : [[round(V.get((r, c), 0.0), 6) for c in range(cols)]
                          for r in range(rows)],
            'free_grid': [[int(maze[r, c]) for c in range(cols)]
                          for r in range(rows)],
            'goal'     : list(GOAL),
            'start'    : list(START),
            'ts'       : time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(vde_dir, f'V_{name}_seed{seed}.json')
        tmp  = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, path)   # atómico: monitor nunca lee un archivo parcial
    except Exception:
        pass   # el checkpoint jamás interrumpe el entrenamiento
_LAST_SNAPSHOT_T = 0.0
_SNAPSHOT_MIN_INTERVAL = 1.0   # s — cap monitor writes (no effect on results)


def _write_snapshot(name, maze_idx, n_mazes, ep, ep_total,
                    plan_means, train_successes, train_steps, V, maze):
    """Write a compact JSON snapshot for monitor.py. Never raises."""
    if not MONITOR_FILE:
        return
    import json, os, time
    # Throttle: at most one write per _SNAPSHOT_MIN_INTERVAL seconds (the final
    # episode always writes). The monitor is a live human-facing view; sub-second
    # writes only add disk churn — costly when MONITOR_FILE lives on a cloud-synced
    # folder (OneDrive), where os.replace can block on sync locks.
    global _LAST_SNAPSHOT_T
    now = time.time()
    if ep < ep_total and (now - _LAST_SNAPSHOT_T) < _SNAPSHOT_MIN_INTERVAL:
        return
    _LAST_SNAPSHOT_T = now
    try:
        rows, cols = maze.shape
        v_grid    = [[round(V.get((r, c), 0.0), 4) for c in range(cols)]
                     for r in range(rows)]
        free_grid = [[int(maze[r, c]) for c in range(cols)]
                     for r in range(rows)]
        data = {
            'algo'           : name,
            'maze_idx'       : maze_idx,
            'n_mazes'        : n_mazes,
            'episode'        : ep,
            'episodes_total' : ep_total,
            'plan_means'     : plan_means[-400:],        # V(s') medio por episodio
            'train_successes': train_successes[-400:],
            'train_steps'    : train_steps[-400:],       # pasos por episodio
            'v_grid'         : v_grid,
            'free_grid'      : free_grid,
            'goal'           : list(GOAL),
            'start'          : list(START),
            'ts'             : time.strftime('%H:%M:%S'),
        }
        tmp = MONITOR_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, MONITOR_FILE)   # atomic: monitor never reads a partial file
    except Exception:
        pass   # never interrupt training


def make_V(grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
           init=0.0, random_init=False, random_max=0.3, seed=42):
    """
    Initialize value table V(s).
      init=0.0, random_init=False  → todos los estados en 0 (comportamiento original).
      init=v, random_init=False    → todos los estados en v (optimista fijo).
      random_init=True             → V(s) ~ Uniform[0, random_max] por estado (optimismo aleatorio).
    Valores clampeados a [0, 1].
    """
    if random_init:
        rng_v = np.random.default_rng(seed)
        hi = max(0.0, min(1.0, float(random_max)))
        return {(i, j): float(rng_v.uniform(0.0, hi))
                for i in range(grid_rows) for j in range(grid_cols)}
    v0 = max(0.0, min(1.0, float(init)))
    return {(i, j): v0 for i in range(grid_rows) for j in range(grid_cols)}


# ─── Shaped terminal reward (ablation) ────────────────────────────────────────
def shaped_terminal_reward(steps, max_steps, good_steps=50, magnitude=1.0):
    """
    Reward for reaching the goal in `steps` steps.
      steps ≤ good_steps  → magnitude        (fast, full reward)
      steps ≥ max_steps   → magnitude * 0.1  (slow, minimum reward)
      in between          → linear decay from magnitude to magnitude * 0.1
    magnitude is clamped to [0.0, 1.0].
    """
    magnitude = max(0.0, min(1.0, float(magnitude)))
    if steps <= good_steps:
        return magnitude
    t = min(1.0, (steps - good_steps) / max(max_steps - good_steps, 1))
    return magnitude * (1.0 - 0.9 * t)


# ─── Solución exacta de Bellman: V* por programación dinámica ────────────────
def compute_v_star(maze, gamma, goal_reward=1.0, step_cost=0.0, goal=GOAL):
    """
    V* EXACTO del MDP determinista del laberinto (análisis, no entrenamiento).

    La política óptima es el camino más corto hacia la meta (tanto γ^d como el
    costo por paso favorecen d mínimo), así que con d = dist_BFS(s, goal):

        V*(s) = γ^(d-1) · goal_reward  −  step_cost · Σ_{k=0}^{d-2} γ^k

    Devuelve dict solo con celdas libres desde las que la meta es alcanzable.
    Se recorta a [0,1] porque la tabla V aprendida vive en ese rango (el gap
    se mide contra el mejor V* representable por el aprendiz).

    Nota shaped-reward: la recompensa moldeada depende de los pasos TOTALES del
    episodio (no-Markov); aquí se usa goal_reward = magnitud (mejor caso), una
    aproximación válida mientras d ≤ shaped_reward_good_steps.
    """
    from collections import deque
    rows, cols = maze.shape
    dist = {goal: 0}
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 1
                    and (nr, nc) not in dist):
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    v_star = {}
    for s, d in dist.items():
        if d == 0:
            v_star[s] = 0.0
            continue
        if gamma < 1.0:
            cost = step_cost * (1.0 - gamma ** (d - 1)) / (1.0 - gamma)
        else:
            cost = step_cost * (d - 1)
        v_star[s] = min(1.0, max(0.0, (gamma ** (d - 1)) * goal_reward - cost))
    return v_star


def bellman_gap(V, v_star, goal=GOAL):
    """(||V − V*||_∞ , RMSE) sobre celdas libres alcanzables, excluida la meta."""
    diffs = [abs(V.get(s, 0.0) - v) for s, v in v_star.items() if s != goal]
    if not diffs:
        return 0.0, 0.0
    arr = np.array(diffs)
    return float(arr.max()), float(np.sqrt(np.mean(arr ** 2)))


def bellman_residual(V, maze, gamma, v_star, goal_reward=1.0, step_cost=0.0,
                     goal=GOAL):
    """
    Residuo VERDADERO del operador de optimalidad de Bellman sobre el modelo:

        T*V(s) = max_a [ r(s,a) + γ · V(s') · 1[s'≠goal] ]
        residuo(s) = |T*V(s) − V(s)|

    Devuelve (max_s residuo, media_s residuo) sobre celdas libres alcanzables.
    A diferencia de |δ| (muestral, depende del target de cada método), este
    residuo usa el modelo real y es comparable entre algoritmos.
    """
    max_r, sum_r, n = 0.0, 0.0, 0
    for s in v_star:
        if s == goal:
            continue
        best = -1e18
        for a in range(4):
            s_next, _ = step(s, a, maze, goal)
            if s_next == goal:
                q = goal_reward
            else:
                q = -step_cost + gamma * V.get(s_next, 0.0)
            if q > best:
                best = q
        res = abs(best - V.get(s, 0.0))
        sum_r += res
        n += 1
        if res > max_r:
            max_r = res
    return max_r, (sum_r / n if n else 0.0)


# ─── Epsilon schedule ─────────────────────────────────────────────────────────
def _get_epsilon(ep, episodes_per_maze, epsilon_start, epsilon_mid, exploit_frac):
    """
    Two-phase epsilon for episode `ep` (1-indexed):
      Phase 1 — episodes 1 … explore_n : linear decay epsilon_start → epsilon_mid
      Phase 2 — episodes explore_n+1 … end : linear decay epsilon_mid → 0.0
    """
    explore_n = max(1, int(round(episodes_per_maze * (1.0 - exploit_frac))))
    exploit_n = episodes_per_maze - explore_n

    if ep <= explore_n:
        t = (ep - 1) / max(explore_n - 1, 1)
        return epsilon_start + (epsilon_mid - epsilon_start) * t
    else:
        t = (ep - explore_n) / max(exploit_n, 1)
        return epsilon_mid * max(0.0, 1.0 - t)


# ─── Single training episode ───────────────────────────────────────────────────
def _train_episode(s, V, maze, planner_fn, cfg,
                   alpha, gamma, lmbda, epsilon, max_steps, rng,
                   use_shaped_reward=False, shaped_reward_good_steps=50,
                   shaped_reward_magnitude=1.0, failure_penalty=0.0,
                   use_step_cost=False, step_cost=0.001,
                   action_selection='planner', td_target='planner',
                   verbose=False, print_interval=100, ep=1, maze_idx=1):
    """
    One TD(λ) episode. V is updated in-place.
    Returns (traj, planner_calls, plan_values, deltas, n_traces_mean, disc_return).
    disc_return = Σ γ^t r_t (discounted sum of rewards for the episode).
    failure_penalty: magnitude [0,1] — terminal TD step on timeout via traces.
    use_step_cost / step_cost: subtract step_cost from r on every non-terminal step.
    action_selection / td_target: ablación de mecanismo —
        'planner' = usa planner_fn (comportamiento original),
        'value'   = usa la tabla V directamente (aísla dónde aporta el bio-algo).
    """
    goal   = cfg.get('goal', GOAL)
    active = {}          # sparse eligibility traces
    traj   = [s]
    steps  = planner_calls = 0
    plan_values, deltas, traces_counts = [], [], []
    disc_return = 0.0
    disc        = 1.0   # current discount factor γ^t

    while s != goal and steps < max_steps:
        steps += 1

        # Con recompensa moldeada el premio terminal depende de los pasos ya
        # dados: publica el valor vigente para que el lookahead simule la
        # misma recompensa (variación ≪ 1e-3 dentro de un horizonte de ~20).
        if use_shaped_reward:
            cfg['goal_reward'] = shaped_terminal_reward(
                steps, max_steps, shaped_reward_good_steps,
                shaped_reward_magnitude)

        # ε-greedy action selection
        if rng.random() < epsilon:
            a = rng.randint(0, 3)
        else:
            best_a, best_q = 0, -1e9
            for aa in range(4):
                sn, r_aa = step(s, aa, maze, goal)
                if sn == goal:
                    pval = 0.0
                elif action_selection == 'value':
                    pval = V.get(sn, 0.0)          # lookup, no cuenta como llamada
                else:
                    pval = planner_fn(sn, V, maze, cfg)
                    planner_calls += 1
                q = r_aa + gamma * pval
                if q > best_q:
                    best_q, best_a = q, aa
            a = best_a

        s_next, r = step(s, a, maze, goal)

        # Per-step cost: subtract from every non-terminal transition.
        # Creates an implicit distance gradient even before any goal is reached.
        if use_step_cost and s_next != goal:
            r -= step_cost

        # Apply shaped terminal reward (ablation)
        if use_shaped_reward and s_next == goal:
            r = shaped_terminal_reward(steps, max_steps, shaped_reward_good_steps,
                                       shaped_reward_magnitude)

        traj.append(s_next)

        # Accumulate discounted return
        disc_return += disc * r
        disc        *= gamma

        # TD target: G_plan (planner) o V(s') (TD(λ) clásico, ablación)
        if s_next == goal:
            r_opt = 0.0
        elif td_target == 'value':
            r_opt = V.get(s_next, 0.0)
        else:
            r_opt = planner_fn(s_next, V, maze, cfg)
            planner_calls += 1
        plan_values.append(r_opt)

        delta = r + gamma * r_opt - V.get(s, 0.0)
        deltas.append(delta)

        # Sparse TD(λ) update
        active[s] = active.get(s, 0.0) + 1.0
        for x in list(active):
            V[x] = min(1.0, max(0.0, V.get(x, 0.0) + alpha * delta * active[x]))
            active[x] *= gamma * lmbda
            if active[x] < 1e-7:
                del active[x]
        traces_counts.append(len(active))

        # Within-episode verbose print
        if verbose and steps % print_interval == 0:
            dir_names = ['↑', '↓', '←', '→']
            v_here = V.get(s, 0.0)
            print(
                f"    │ paso {steps:4d}  pos={s}→{s_next}  acción={dir_names[a]}"
                f"  r={r:.3f}  δ={delta:+.4f}"
                f"  V(s)={v_here:.4f}  plan={r_opt:.4f}"
                f"  trazas={len(active)}"
            )

        s = s_next

    # Terminal failure penalty — one final TD step through remaining traces.
    # δ = -p - V(s_last): states that led to timeout get their V reduced
    # proportionally to how recently they were visited (trace weight).
    if s != goal and failure_penalty > 0.0:
        p = max(0.0, min(1.0, float(failure_penalty)))
        delta_fail = -p - V.get(s, 0.0)
        disc_return += disc * (-p)
        deltas.append(delta_fail)
        for x in list(active):
            V[x] = min(1.0, max(0.0, V.get(x, 0.0) + alpha * delta_fail * active[x]))

    n_traces_mean = float(np.mean(traces_counts)) if traces_counts else 0.0
    return traj, planner_calls, plan_values, deltas, n_traces_mean, disc_return


# ─── Single test episode ───────────────────────────────────────────────────────
def _test_episode(s, V, maze, planner_fn, cfg, gamma, epsilon, max_steps, rng,
                  action_selection='planner'):
    """Greedy test episode — V is read-only."""
    goal = cfg.get('goal', GOAL)
    traj = [s]
    steps = planner_calls = 0
    plan_values = []

    while s != goal and steps < max_steps:
        steps += 1
        if rng.random() < epsilon:
            a = rng.randint(0, 3)
        else:
            best_a, best_q, best_pval = 0, -1e9, 0.0
            for aa in range(4):
                sn, r_aa = step(s, aa, maze, goal)
                if sn == goal:
                    pval = 0.0
                elif action_selection == 'value':
                    pval = V.get(sn, 0.0)
                else:
                    pval = planner_fn(sn, V, maze, cfg)
                    planner_calls += 1
                q = r_aa + gamma * pval
                if q > best_q:
                    best_q, best_a, best_pval = q, aa, pval
            plan_values.append(best_pval)
            a = best_a
        s_next, r = step(s, a, maze, goal)
        traj.append(s_next)
        s = s_next

    return traj, planner_calls, plan_values


# ─── Full experiment runner ────────────────────────────────────────────────────
def run_experiment(name, train_mazes, test_mazes, planner_fn, cfg,
                   alpha=0.08, gamma=0.99, lmbda=0.7,
                   episodes_per_maze=20, max_steps=1000,
                   epsilon_start=1.0, epsilon_mid=0.1, exploit_frac=0.20,
                   test_epsilon=0.01, test_max_steps=None,
                   test_action_selection=None,
                   test_adapt_enabled=False, test_adapt_episodes=5,
                   test_adapt_explore_episodes=3, test_adapt_explore_epsilon=1.0,
                   test_adapt_max_rounds=3, test_adapt_epsilon=0.30,
                   test_adapt_max_steps=None,
                   use_shaped_reward=False, shaped_reward_good_steps=50,
                   shaped_reward_magnitude=1.0, failure_penalty=0.0,
                   use_step_cost=False, step_cost=0.001,
                   action_selection='planner', td_target='planner',
                   v_init=0.0, v_init_random=False, v_init_max=0.3,
                   seed=42, verbose=True, print_interval=100,
                   method_name='', method_idx=1, n_methods=1,
                   progress_cb=None,
                   **_ignored):
    """
    Train on train_mazes (shared V across all mazes), then test on test_mazes.

    Parameters
    ──────────
    exploit_frac            : fraction of episodes at the END of each maze where epsilon→0.
    test_max_steps          : step limit for test episodes (default = max_steps).
    test_action_selection   : 'planner'|'value'|None — mecanismo de acción SOLO
                              durante el test (None = hereda action_selection).
                              Con 'planner' el bio-algoritmo/rollout navega el
                              laberinto de test con lookahead real (control de
                              horizonte retrocedente); para TDL es equivalente
                              a 'value' (su planner ES la tabla V). NO afecta
                              al entrenamiento.
    test_adapt_enabled      : adaptación EN LÍNEA durante el test — si el
                              intento greedy falla, se ejecutan episodios de
                              reajuste TD(λ) (misma regla de aprendizaje del
                              método) sobre una COPIA de V en ese laberinto y
                              se reintenta. La V entrenada nunca se modifica y
                              cada laberinto de test parte de la misma V.
                              Cada ronda tiene DOS fases: exploración (ε alto,
                              descubre el laberinto nuevo) y explotación (ε bajo,
                              afina la ruta), y luego el reintento greedy.
    test_adapt_explore_episodes : episodios de EXPLORACIÓN por ronda (fase 1).
    test_adapt_explore_epsilon  : ε de la fase de exploración (1.0 = pura).
    test_adapt_episodes     : episodios de EXPLOTACIÓN por ronda (fase 2).
    test_adapt_epsilon      : ε (bajo) de la fase de explotación.
    test_adapt_max_rounds   : rondas máximas (explora+explota)→reintento.
    test_adapt_max_steps    : límite de pasos en reajuste (None = max_steps).
    use_shaped_reward       : ablation flag — terminal reward decays with steps taken.
    shaped_reward_good_steps: steps below this threshold get full reward = 1.0.
    action_selection        : 'planner'|'value' — qué guía la acción ε-greedy.
    td_target               : 'planner'|'value' — qué bootstrap usa δ.
    print_interval          : within-episode step interval for detailed logging.
    progress_cb             : callable(phase, done, total, maze_idx, n_mazes,
                              n_success=None) o None. Notificación de avance para
                              el modo paralelo de run_all.py (n_success = éxitos
                              de test acumulados); solo informa, NUNCA afecta
                              resultados.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    cfg = dict(cfg)
    cfg['goal']  = GOAL
    cfg['gamma'] = gamma
    # Recompensa efectiva del MDP aprendido → el lookahead simula lo mismo.
    # (_train_episode actualiza cfg['goal_reward'] por paso si hay shaping;
    #  _goal_reward_ref queda fijo como referencia para las métricas Bellman.)
    cfg['step_cost']   = step_cost if use_step_cost else 0.0
    _goal_reward_ref   = shaped_reward_magnitude if use_shaped_reward else 1.0
    cfg['goal_reward'] = _goal_reward_ref

    _test_max = test_max_steps if test_max_steps is not None else max_steps

    V = make_V(init=v_init, random_init=v_init_random, random_max=v_init_max, seed=seed)

    explore_n = max(1, int(round(episodes_per_maze * (1.0 - exploit_frac))))

    train_successes, train_steps   = [], []
    train_returns,   train_plan_means = [], []
    # Typed float arrays: one entry per training STEP (can reach tens of millions
    # over a full run). array('d') stores 8 B/entry vs ~32 B for a list of Python
    # floats, cutting peak RAM ~4× and the swap-driven slowdown late in training.
    # Values are identical; downstream code wraps these in np.array()/list().
    all_plan_values, all_deltas    = array('d'), array('d')
    total_planner_calls = 0
    state_visits = {s: 0 for s in _ALL_STATES}
    v_norms      = []                              # ||ΔV||₂ per training episode
    bellman_gap_inf, bellman_gap_rmse = [], []     # ||V−V*|| vs V* del laberinto actual
    bellman_resid_max, bellman_resid_mean = [], [] # residuo verdadero |T*V − V|
    best_train_steps = float('inf')
    best_train_traj  = []
    best_train_maze  = None

    sep = '─' * 64

    # ══ Training phase ══════════════════════════════════════════════
    shaped_tag   = '  [RECOMPENSA MOLDEADA]'         if use_shaped_reward  else ''
    penalty_tag  = f'  [PENALIZACIÓN -{failure_penalty:.2f}]' if failure_penalty > 0 else ''
    step_tag     = f'  [COSTO PASO -{step_cost}]'    if use_step_cost      else ''
    vinit_tag    = (f'  [V0~U(0,{v_init_max})]' if v_init_random
                    else (f'  [V0={v_init}]' if v_init > 0 else ''))
    if verbose:
        print(f"\n{'═'*64}")
        print(f"  [{name}]  FASE DE ENTRENAMIENTO{shaped_tag}{penalty_tag}{step_tag}{vinit_tag}")
        print(f"  Laberintos: {len(train_mazes)}  |  Episodios/laberinto: {episodes_per_maze}")
        print(f"  Fase exploración: ep 1–{explore_n}  |  Fase explotación: ep {explore_n+1}–{episodes_per_maze}")
        print(f"{'═'*64}\n")

    _total_eps_per_method = len(train_mazes) * episodes_per_maze
    _total_eps_all        = max(n_methods, 1) * _total_eps_per_method

    for maze_idx, maze in enumerate(train_mazes, 1):
        # V* exacto del laberinto actual (análisis Bellman formal, no afecta
        # el entrenamiento). Con V compartida entre laberintos no existe un
        # punto fijo único: el gap se mide contra el V* del laberinto en curso.
        v_star_maze = compute_v_star(
            maze, gamma,
            goal_reward=_goal_reward_ref, step_cost=cfg['step_cost'],
        )
        if verbose:
            _done = (method_idx - 1) * _total_eps_per_method + (maze_idx - 1) * episodes_per_maze
            _pct  = _done / _total_eps_all * 100 if _total_eps_all > 0 else 0.0
            _mname = method_name or name
            _method_tag = f' con {_mname}({method_idx})' if n_methods > 1 else ''
            _pct_tag    = f'    {_pct:.0f}%'            if n_methods > 1 else ''
            print(f"\n{sep}")
            print(f"  Laberinto de entrenamiento {maze_idx}/{len(train_mazes)}{_method_tag}.{_pct_tag}")
            print(sep)
            print_maze_ascii(maze, V=V)

        maze_successes, maze_steps_list = [], []

        for ep in range(1, episodes_per_maze + 1):
            epsilon = _get_epsilon(ep, episodes_per_maze,
                                   epsilon_start, epsilon_mid, exploit_frac)

            # Snapshot V before episode for ||ΔV||₂ tracking
            v_before = np.array([V[s] for s in _ALL_STATES])

            traj, calls, pvals, dvals, n_traces, ep_disc = _train_episode(
                START, V, maze, planner_fn, cfg,
                alpha, gamma, lmbda, epsilon, max_steps, rng,
                use_shaped_reward=use_shaped_reward,
                shaped_reward_good_steps=shaped_reward_good_steps,
                shaped_reward_magnitude=shaped_reward_magnitude,
                failure_penalty=failure_penalty,
                use_step_cost=use_step_cost, step_cost=step_cost,
                action_selection=action_selection, td_target=td_target,
                verbose=verbose, print_interval=print_interval,
                ep=ep, maze_idx=maze_idx,
            )
            total_planner_calls += calls
            all_plan_values.extend(pvals)
            all_deltas.extend(dvals)

            # ||ΔV||₂ and state visitation
            v_after = np.array([V[s] for s in _ALL_STATES])
            v_norms.append(float(np.linalg.norm(v_after - v_before)))
            for sv in traj:
                state_visits[sv] += 1

            # Convergencia formal a Bellman (por episodio, vs laberinto actual)
            g_inf, g_rmse = bellman_gap(V, v_star_maze)
            r_max, r_mean = bellman_residual(
                V, maze, gamma, v_star_maze,
                goal_reward=_goal_reward_ref, step_cost=cfg['step_cost'],
            )
            bellman_gap_inf.append(g_inf)
            bellman_gap_rmse.append(g_rmse)
            bellman_resid_max.append(r_max)
            bellman_resid_mean.append(r_mean)

            # Live monitor snapshot (no-op when MONITOR_FILE is None)
            _write_snapshot(
                name, maze_idx, len(train_mazes),
                (maze_idx - 1) * episodes_per_maze + ep,
                len(train_mazes) * episodes_per_maze,
                train_plan_means, train_successes, train_steps, V, maze,
            )

            success     = traj[-1] == GOAL
            ep_steps    = len(traj) - 1

            # Track best (shortest successful) training trajectory
            if success and ep_steps < best_train_steps:
                best_train_steps = ep_steps
                best_train_traj  = list(traj)
                best_train_maze  = maze
            mean_delta  = float(np.mean(np.abs(dvals))) if dvals else 0.0
            mean_pval   = float(np.mean(pvals))          if pvals else 0.0

            train_successes.append(int(success))
            train_steps.append(ep_steps)
            train_returns.append(ep_disc)
            train_plan_means.append(mean_pval)
            maze_successes.append(int(success))
            maze_steps_list.append(ep_steps)

            # Running success rate: last 5 episodes
            recent_n    = min(5, len(maze_successes))
            recent_rate = sum(maze_successes[-recent_n:]) / recent_n

            # Phase label
            phase = 'EXPLOR' if ep <= explore_n else 'EXPLOIT'

            if verbose:
                status = '✓ OK  ' if success else '✗ FAIL'
                print(
                    f"  Ep {ep:3d}/{episodes_per_maze}"
                    f" [{phase}]  ε={epsilon:.4f}"
                    f"  {status}"
                    f"  pasos={ep_steps:4d}"
                    f"  G0={ep_disc:.4f}"
                    f"  |δ|={mean_delta:.4f}"
                    f"  plan={mean_pval:.4f}"
                    f"  trazas≈{n_traces:.0f}"
                    f"  tasa(ult{recent_n})={recent_rate:.2f}"
                    f"  calls={calls}"
                )

            if progress_cb is not None:
                try:
                    progress_cb('train',
                                (maze_idx - 1) * episodes_per_maze + ep,
                                len(train_mazes) * episodes_per_maze,
                                maze_idx, len(train_mazes))
                except Exception:
                    pass   # el reporte de progreso nunca interrumpe el entrenamiento

        if verbose:
            maze_rate = np.mean(maze_successes) * 100
            print(f"\n  → Resumen laberinto {maze_idx}: "
                  f"éxito={maze_rate:.1f}%  "
                  f"pasos_med={np.mean(maze_steps_list):.1f}  "
                  f"pasos_min={min(maze_steps_list)}")
            print(f"  → V(start)={V[START]:.4f}  "
                  f"V({GRID_ROWS-2},{GRID_COLS-1})={V.get((GRID_ROWS-2,GRID_COLS-1),0):.4f}  "
                  f"V({GRID_ROWS-1},{GRID_COLS-2})={V.get((GRID_ROWS-1,GRID_COLS-2),0):.4f}")
            print(f"  → Mapa V(s) actualizado:")
            print_maze_ascii(maze, V=V)

        # Checkpoint V(s) cada VDE_EVERY laberintos (+ el último) → monitor_VdE/
        if maze_idx % VDE_EVERY == 0 or maze_idx == len(train_mazes):
            _write_vde_dump(name, seed, maze_idx, len(train_mazes), V, maze)

    # ══ Test phase ══════════════════════════════════════════════════
    test_successes, test_steps_list, test_trajs = [], [], []
    test_returns = []
    plan_values_test = []
    # Métricas del protocolo de test extendido (opciones 1 y 2)
    test_first_successes, test_first_steps = [], []   # intento 1 = zero-shot
    test_adapt_rounds,   test_adapt_eps_used = [], [] # esfuerzo de adaptación

    # Opción 2 — mecanismo de acción EXCLUSIVO del test (no toca el train):
    # None hereda action_selection (comportamiento clásico); 'planner' pone al
    # bio-algoritmo a navegar el laberinto de test con lookahead real.
    _test_action_sel = test_action_selection or action_selection
    _adapt_max = (test_adapt_max_steps if test_adapt_max_steps is not None
                  else max_steps)

    if verbose:
        print(f"\n{'═'*64}")
        print(f"  [{name}]  FASE DE TEST")
        print(f"  Laberintos: {len(test_mazes)}  |  ε_test={test_epsilon}"
              f"  |  acción test: {_test_action_sel}")
        if test_adapt_enabled:
            print(f"  Adaptación en línea: ON — por ronda: "
                  f"{test_adapt_explore_episodes} explora(ε={test_adapt_explore_epsilon}) "
                  f"+ {test_adapt_episodes} explota(ε={test_adapt_epsilon}); "
                  f"{test_adapt_max_rounds} rondas máx, pasos máx={_adapt_max}  "
                  f"(V se copia por laberinto)")
        print(f"{'═'*64}\n")

    for maze_idx, maze in enumerate(test_mazes, 1):
        if verbose:
            from maze_env import shortest_path
            opt = shortest_path(maze)
            print(f"  Test laberinto {maze_idx}/{len(test_mazes)}  (óptimo BFS={opt})")
            print_maze_ascii(maze)

        # Opción 1 — aislamiento: cada laberinto de test parte de una COPIA de
        # la V entrenada; los reajustes viven en la copia y se descartan al
        # pasar al siguiente laberinto. La V entrenada (la que se reporta y
        # grafica) NUNCA se modifica en la fase de test.
        V_maze = dict(V) if test_adapt_enabled else V

        # ── Intento 1 (zero-shot) ────────────────────────────────────────
        traj, calls, pvals = _test_episode(
            START, V_maze, maze, planner_fn, cfg, gamma, test_epsilon,
            _test_max, rng, action_selection=_test_action_sel,
        )
        total_planner_calls += calls
        success  = traj[-1] == GOAL
        ep_steps = len(traj) - 1
        test_first_successes.append(int(success))
        test_first_steps.append(ep_steps)
        if verbose and test_adapt_enabled:
            print(f"    intento 1 (zero-shot): "
                  f"{'✓ ÉXITO' if success else '✗ FALLO'}  pasos={ep_steps}")

        # ── Rondas de adaptación: fallo → (explora + explota) → reintento ─
        #   Dos fases por ronda sobre la COPIA V_maze. MISMA regla de
        #   aprendizaje del método (action_selection/td_target/α/γ/λ/flags de
        #   train); solo cambian ε y el límite de pasos. Sus métricas NO se
        #   mezclan con las de entrenamiento (se descartan).
        rounds_used, eps_used = 0, 0
        if test_adapt_enabled and not success:
            def _adapt_episode(eps):
                _, a_calls, _, _, _, _ = _train_episode(
                    START, V_maze, maze, planner_fn, cfg,
                    alpha, gamma, lmbda, eps, _adapt_max, rng,
                    use_shaped_reward=use_shaped_reward,
                    shaped_reward_good_steps=shaped_reward_good_steps,
                    shaped_reward_magnitude=shaped_reward_magnitude,
                    failure_penalty=failure_penalty,
                    use_step_cost=use_step_cost, step_cost=step_cost,
                    action_selection=action_selection, td_target=td_target,
                    verbose=False,
                )
                return a_calls
            for _round in range(1, test_adapt_max_rounds + 1):
                rounds_used = _round
                # Fase 1 — EXPLORACIÓN (ε alto: descubre el laberinto nuevo)
                for _k in range(test_adapt_explore_episodes):
                    total_planner_calls += _adapt_episode(test_adapt_explore_epsilon)
                    eps_used += 1
                # Fase 2 — EXPLOTACIÓN (ε bajo: afina la ruta encontrada)
                for _k in range(test_adapt_episodes):
                    total_planner_calls += _adapt_episode(test_adapt_epsilon)
                    eps_used += 1
                # Reintento de test (greedy, ε_test)
                traj, calls, pvals = _test_episode(
                    START, V_maze, maze, planner_fn, cfg, gamma, test_epsilon,
                    _test_max, rng, action_selection=_test_action_sel,
                )
                total_planner_calls += calls
                success  = traj[-1] == GOAL
                ep_steps = len(traj) - 1
                if verbose:
                    print(f"    ronda {_round}/{test_adapt_max_rounds} "
                          f"(+{test_adapt_explore_episodes} expl "
                          f"+{test_adapt_episodes} expt): "
                          f"{'✓ ÉXITO' if success else '✗ FALLO'}  pasos={ep_steps}")
                if success:
                    break

        plan_values_test.extend(pvals)   # V(s') del intento FINAL reportado

        # Episode return for test (del intento final)
        if success:
            term_r = (shaped_terminal_reward(ep_steps, _test_max, shaped_reward_good_steps,
                                             shaped_reward_magnitude)
                      if use_shaped_reward else 1.0)
            t_return = term_r * (gamma ** ep_steps)
        else:
            t_return = 0.0

        test_successes.append(int(success))
        test_steps_list.append(ep_steps)
        test_trajs.append(traj)
        test_returns.append(t_return)
        test_adapt_rounds.append(rounds_used)
        test_adapt_eps_used.append(eps_used)

        if progress_cb is not None:
            try:
                progress_cb('test', maze_idx, len(test_mazes),
                            maze_idx, len(test_mazes),
                            n_success=sum(test_successes))
            except Exception:
                pass

        if verbose:
            from maze_env import shortest_path
            opt = shortest_path(maze)
            ratio = ep_steps / opt if opt > 0 else float('inf')
            status = '✓ ÉXITO' if success else '✗ FALLO'
            adapt_tag = (f'  reajuste={rounds_used} ronda(s), {eps_used} ep'
                         if eps_used else '')
            print(f"  {status}  pasos={ep_steps}  G0={t_return:.4f}  "
                  f"óptimo={opt}  ratio={ratio:.2f}  calls={calls}{adapt_tag}")

    results = {
        'name'              : name,
        'V'                 : V,
        'train_successes'   : train_successes,
        'train_steps'       : train_steps,
        'train_returns'     : train_returns,      # G0 per training episode
        'train_plan_means'  : train_plan_means,   # mean plan value per training episode
        'test_successes'    : test_successes,      # intento FINAL (tras adaptación)
        'test_steps'        : test_steps_list,     # pasos del intento final
        'test_returns'      : test_returns,        # G0 per test maze (intento final)
        'test_trajs'        : test_trajs,          # trayectoria del intento final
        'test_first_successes': test_first_successes,  # intento 1 = zero-shot
        'test_first_steps'  : test_first_steps,        # pasos del intento 1
        'test_adapt_rounds' : test_adapt_rounds,       # rondas de reajuste usadas
        'test_adapt_episodes': test_adapt_eps_used,    # episodios de reajuste usados
        'best_train_traj'   : best_train_traj,    # shortest successful train episode traj
        'best_train_maze'   : best_train_maze,    # maze where best traj was recorded
        'plan_values_train' : all_plan_values,
        'plan_values_test'  : plan_values_test,
        'deltas'            : all_deltas,
        'planner_calls'     : total_planner_calls,
        'state_visits'      : state_visits,        # visit count per state (train)
        'v_norms'           : v_norms,             # ||ΔV||₂ per training episode
        'bellman_gap_inf'   : bellman_gap_inf,     # ||V−V*||∞ por episodio (V* exacto)
        'bellman_gap_rmse'  : bellman_gap_rmse,    # RMSE(V, V*) por episodio
        'bellman_resid_max' : bellman_resid_max,   # max |T*V−V| por episodio
        'bellman_resid_mean': bellman_resid_mean,  # media |T*V−V| por episodio
        '_hp': {
            'alpha': alpha, 'gamma': gamma, 'lmbda': lmbda,
            'episodes_per_maze': episodes_per_maze, 'max_steps': max_steps,
            'epsilon_start': epsilon_start, 'epsilon_mid': epsilon_mid,
            'exploit_frac': exploit_frac, 'test_epsilon': test_epsilon,
            'test_action_selection': test_action_selection,
            'test_adapt_enabled': test_adapt_enabled,
            'test_adapt_explore_episodes': test_adapt_explore_episodes,
            'test_adapt_explore_epsilon': test_adapt_explore_epsilon,
            'test_adapt_episodes': test_adapt_episodes,
            'test_adapt_max_rounds': test_adapt_max_rounds,
            'test_adapt_epsilon': test_adapt_epsilon,
            'test_adapt_max_steps': test_adapt_max_steps,
            'use_shaped_reward': use_shaped_reward,
            'shaped_reward_good_steps': shaped_reward_good_steps,
            'shaped_reward_magnitude': shaped_reward_magnitude,
            'failure_penalty': failure_penalty,
            'use_step_cost': use_step_cost, 'step_cost': step_cost,
            'action_selection': action_selection, 'td_target': td_target,
            'v_init': v_init, 'v_init_random': v_init_random, 'v_init_max': v_init_max,
            'seed': seed,
        },
    }

    if verbose:
        tr = np.mean(train_successes) * 100
        te = np.mean(test_successes)  * 100
        print(f"\n  [{name}] ── RESUMEN FINAL ────────────────────────────────")
        print(f"    Train éxito   : {sum(train_successes)}/{len(train_successes)} ({tr:.1f}%)")
        print(f"    Test  éxito   : {sum(test_successes)}/{len(test_successes)} ({te:.1f}%)")
        if test_adapt_enabled and test_first_successes:
            print(f"    Test zero-shot: {sum(test_first_successes)}"
                  f"/{len(test_first_successes)} "
                  f"({np.mean(test_first_successes)*100:.1f}%)  |  "
                  f"rondas reajuste medias: {np.mean(test_adapt_rounds):.2f}")
        print(f"    Pasos train ≈ : {np.mean(train_steps):.1f}")
        print(f"    Pasos test  ≈ : {np.mean(test_steps_list):.1f}")
        print(f"    G0 train ≈    : {np.mean(train_returns):.4f}")
        print(f"    G0 test  ≈    : {np.mean(test_returns):.4f}")
        print(f"    Llamadas plan.: {total_planner_calls}")

    return results
