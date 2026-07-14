"""
TDLambda_clasico.py — TD(λ) clásico como línea base para navegación en laberintos.

El "planificador" es V(s) directamente — sin lookahead, sin optimización bio.
Es TD(λ) puro: el target de bootstrap es r + γ·V(s'), leyendo la tabla actual.
Sirve como línea base mínima para comparación justa contra bio-planificadores y
rollout aleatorio.

Run standalone:  python TDLambda_clasico.py
Run via run_all: (importado automáticamente como 'TDL')
"""

import numpy as np
import matplotlib.pyplot as plt
from maze_env import (
    GOAL, GRID_ROWS, GRID_COLS,
    TRAIN_CSV, TEST_CSV, load_mazes, START
)
from core import run_experiment

# No extra hyperparameters — classic TD(λ) reads V(s) directly.
TDL_CFG = {}


# ─── Planner ──────────────────────────────────────────────────────────────────
def planner(state, V, maze, cfg):
    """Classic TD(λ) planner: returns V(state) directly, no lookahead."""
    goal = cfg.get('goal', GOAL)
    if state == goal:
        return 0.0
    return V.get(state, 0.0)


# ─── Post-hoc instrumented analysis (no effect on training) ───────────────────
def convergence_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    TDL has no search loop — a single table lookup V(s').
    Returns a flat line at V(start_state) over 56 evaluations for visual
    reference (56 = presupuesto común de los planners en fig11).
    This represents the zero-overhead baseline with no search budget.
    """
    val = max(0.0, min(1.0, V.get(start_state, 0.0)))
    return np.full(56, val)


# ─── Standalone runner ────────────────────────────────────────────────────────
def run(name='TDL', train_mazes=None, test_mazes=None, verbose=True, **hp):
    """
    Train and test classic TD(λ). Returns results dict from core.run_experiment.
    Any key in **hp overrides the default hyperparameter (pass COMMON_HP from run_all.py).
    """
    if train_mazes is None:
        train_mazes = load_mazes(TRAIN_CSV)
    if test_mazes is None:
        test_mazes = load_mazes(TEST_CSV)

    defaults = dict(
        alpha=0.08, gamma=0.99, lmbda=0.7,
        episodes_per_maze=80, max_steps=1000,
        epsilon_start=1.0, epsilon_mid=0.1, exploit_frac=0.20,
        test_epsilon=0.01, seed=42, verbose=verbose,
    )
    defaults.update(hp)
    defaults['verbose'] = verbose

    results = run_experiment(
        name, train_mazes, test_mazes, planner, TDL_CFG.copy(), **defaults
    )

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_mazes = load_mazes(TRAIN_CSV)
    test_mazes  = load_mazes(TEST_CSV)
    print(f"Loaded {len(train_mazes)} train mazes, {len(test_mazes)} test mazes.\n")

    results = run(verbose=True)

    V = results['V']

    # V-value heatmap
    v_matrix = np.array([[V[(i, j)] for j in range(GRID_COLS)] for i in range(GRID_ROWS)])
    plt.figure(figsize=(7, 6))
    plt.imshow(v_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='V(s)')
    plt.title('TD(λ) clásico: Función de valor aprendida V(s)')
    plt.savefig('tdl_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # TD errors
    plt.figure(figsize=(10, 3))
    plt.plot(results['deltas'], alpha=0.6, linewidth=0.5, color='gray')
    plt.title('TD(λ) clásico: Errores TD (δ) durante entrenamiento')
    plt.xlabel('Paso de actualización')
    plt.ylabel('δ')
    plt.savefig('tdl_deltas.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Rolling training success
    window = 50
    succ    = np.array(results['train_successes'], dtype=float)
    rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rolling, color='gray')
    plt.axhline(np.mean(results['test_successes']), color='red',
                linestyle='--', label=f"Test rate = {np.mean(results['test_successes']):.2f}")
    plt.title(f'TD(λ) clásico: Tasa de éxito en train (ventana={window})')
    plt.xlabel('Episodio')
    plt.ylabel('Tasa de éxito')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('tdl_train_success.png', dpi=150, bbox_inches='tight')
    plt.show()

    # V(s') distribution during training (plan_values = V(s') used as bootstrap)
    pv = results['plan_values_train']
    plt.figure(figsize=(7, 4))
    plt.hist(pv, bins=40, edgecolor='black', color='gray', alpha=0.7)
    plt.axvline(np.mean(pv), color='red', linestyle='--',
                label=f'Media = {np.mean(pv):.3f}')
    plt.title("TD(λ) clásico: Distribución de V(s') usado como bootstrap (train)")
    plt.xlabel("V(s')"); plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('tdl_plan_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Training steps boxplot
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(results['train_steps'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][0].set_alpha(0.7)
    plt.title('TD(λ) clásico: Variabilidad de pasos por episodio (train)')
    plt.ylabel('Pasos por episodio')
    plt.xticks([1], ['TD(λ)'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('tdl_steps_box.png', dpi=150, bbox_inches='tight')
    plt.show()
