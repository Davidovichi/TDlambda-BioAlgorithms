"""
TDLambda_rrollout.py — Random Rollout + TD(λ) para navegación en laberintos.

El planificador evalúa N secuencias de acciones aleatorias desde el estado actual
y devuelve el MEJOR retorno descontado encontrado, con bootstrap de V al final del
horizonte (usando simulate_sequence de maze_env).

Teoría del rollout en TD(λ)
────────────────────────────
TD(λ) clásico usa V(s') como estimación del valor futuro:
    δ = r + γ·V(s') − V(s)

TD(λ) + Random Rollout reemplaza V(s') por una estimación de rollout:
    G_roll(s') = Σ_t γ^t r_t  sobre H pasos aleatorios, + γ^H·V(s_H)

Parámetros del rollout
──────────────────────
• horizon (H): profundidad del rollout. Controla el trade-off entre coste y
  calidad de la estimación. Al final del horizonte se hace bootstrap con V.
  Adecuado: la distancia Manhattan START→GOAL ≈ 38 celdas, H=12 da lookahead
  parcial útil; rollouts más profundos son posibles pero costosos.

• n_rollouts: número de secuencias aleatorias evaluadas por llamada al
  planificador. Se devuelve el MÁXIMO (max-sampling), lo que equivale a una
  búsqueda aleatoria optimista — comparable con los algoritmos bio-inspirados
  que también seleccionan el mejor individuo de N evaluaciones.
  Si se prefiere una estimación no-sesgada del valor bajo política aleatoria,
  cambiar `best` por `mean` convierte esto en Monte Carlo puro.

Posición en la jerarquía de comparación
─────────────────────────────────────────
  TDL (clasico)  <  ROLLOUT (aleatorio)  <  PSO/ABC/DE/GA (optimizado)
  Sin lookahead     Lookahead aleatorio     Lookahead optimizado bio

Presupuesto: n_rollouts = 56 evaluaciones/llamada — MISMO presupuesto que los
bio-algoritmos (PSO 8×7, ABC 8+3×16, DE/GA 8×(1+6), todos = 56).

Run standalone:  python TDLambda_rrollout.py
Run via run_all: (importado automáticamente como 'ROLLOUT')
"""

import numpy as np
import matplotlib.pyplot as plt
from maze_env import (
    simulate_sequence, GOAL, GRID_ROWS, GRID_COLS,
    TRAIN_CSV, TEST_CSV, load_mazes, START
)
from core import run_experiment

# ─── Rollout hyperparameters ──────────────────────────────────────────────────
ROLLOUT_CFG = {
    'horizon'   : 20,   # lookahead depth (same as bio algos)
    'n_rollouts': 56,   # independent random rollouts (budget = 56, igual que bios)
}


# ─── Planner ──────────────────────────────────────────────────────────────────
def planner(start_state, V, maze, cfg):
    """
    Random rollout planner. Evaluates n_rollouts random action sequences and
    returns the best found discounted return (bootstrapped with V at horizon).
    Same budget as bio algorithms, but without optimization.
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return 0.0

    horizon     = cfg['horizon']
    n_rollouts  = cfg['n_rollouts']
    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    best = -1e9   # con step_cost los retornos pueden ser negativos; el clamp final acota
    for _ in range(n_rollouts):
        actions = np.random.randint(0, 4, horizon)
        val = simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                step_cost=step_cost, goal_reward=goal_reward)
        if val > best:
            best = val

    return max(0.0, min(1.0, best))


# ─── Post-hoc instrumented analysis (no effect on training) ───────────────────
def convergence_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Best-so-far V(s') after each of the n_rollouts random evaluations,
    averaged over n_samples independent runs.
    Serves as the random-search baseline for comparison with bio convergence curves.
    Returns array of shape (n_rollouts,).
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    n_rollouts = cfg['n_rollouts']
    if start_state == goal:
        return np.zeros(n_rollouts)

    horizon     = cfg['horizon']
    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_samples):
        best = 0.0
        curve = []
        for _ in range(n_rollouts):
            actions = rng.integers(0, 4, horizon)
            val = simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                    step_cost=step_cost, goal_reward=goal_reward)
            if val > best:
                best = val
            curve.append(best)
        curves.append(curve)
    return np.mean(curves, axis=0)


# ─── Standalone runner ────────────────────────────────────────────────────────
def run(name='ROLLOUT', train_mazes=None, test_mazes=None,
        verbose=True, **hp):
    """
    Train and test Rollout-TD(λ). Returns results dict from core.run_experiment.
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
        name, train_mazes, test_mazes, planner, ROLLOUT_CFG.copy(), **defaults
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
    plt.imshow(v_matrix, cmap='cividis', vmin=0, vmax=1)
    plt.colorbar(label='V(s)')
    plt.title('Rollout-TD(λ): Función de valor aprendida V(s)')
    plt.savefig('rollout_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # TD errors
    plt.figure(figsize=(10, 3))
    plt.plot(results['deltas'], alpha=0.6, linewidth=0.5, color='#795548')
    plt.title('Rollout-TD(λ): Errores TD (δ) durante entrenamiento')
    plt.xlabel('Paso de actualización')
    plt.ylabel('δ')
    plt.savefig('rollout_deltas.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Rolling training success
    window = 50
    succ    = np.array(results['train_successes'], dtype=float)
    rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rolling, color='#795548')
    plt.axhline(np.mean(results['test_successes']), color='red',
                linestyle='--', label=f"Test rate = {np.mean(results['test_successes']):.2f}")
    plt.title(f'Rollout-TD(λ): Tasa de éxito en train (ventana={window})')
    plt.xlabel('Episodio')
    plt.ylabel('Tasa de éxito')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('rollout_train_success.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plan values distribution
    pv = results['plan_values_train']
    plt.figure(figsize=(7, 4))
    plt.hist(pv, bins=40, edgecolor='black', color='#795548', alpha=0.7)
    plt.axvline(np.mean(pv), color='red', linestyle='--',
                label=f'Media = {np.mean(pv):.3f}')
    plt.title("Rollout-TD(λ): Distribución de V(s') estimado (rollout aleatorio)")
    plt.xlabel("V(s') estimado"); plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('rollout_plan_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Training steps boxplot
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(results['train_steps'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#795548')
    bp['boxes'][0].set_alpha(0.7)
    plt.title('Rollout-TD(λ): Variabilidad de pasos por episodio (train)')
    plt.ylabel('Pasos por episodio')
    plt.xticks([1], ['ROLLOUT'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('rollout_steps_box.png', dpi=150, bbox_inches='tight')
    plt.show()
