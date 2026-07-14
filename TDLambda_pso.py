"""
TDLambda_pso.py — Particle Swarm Optimization + TD(λ) for maze navigation.

PSO is used as a lookahead planner: given the current state and value function V,
a swarm of action-sequence particles searches for the highest-return trajectory
over a short horizon. The best found value is used as the TD bootstrap target.

Algorithm (PSO/gbest topology):
  1. Initialize swarm as random integer action sequences [0,3]^horizon
  2. Each particle has a continuous velocity; positions are rounded to ints
  3. Per iteration: evaluate fitness, update pbest/gbest, update velocity/position
  4. Return gbest fitness (clipped to [0,1])

Budget: swarm_size × iterations = 8 × 7 = 56 fitness evaluations per plan call
        (el enjambre completo se evalúa en cada iteración; no hay evaluación
        inicial separada). MISMO presupuesto que ABC, DE, GA y ROLLOUT (56).

Run standalone:  python TDLambda_pso.py
Run via run_all: (imported automatically as 'PSO')
"""

import numpy as np
import matplotlib.pyplot as plt
from maze_env import (
    simulate_sequence, GOAL, GRID_ROWS, GRID_COLS,
    TRAIN_CSV, TEST_CSV, load_mazes, START
)
from core import run_experiment

# ─── PSO hyperparameters ──────────────────────────────────────────────────────
PSO_CFG = {
    'horizon'    : 20,   # lookahead depth
    'swarm_size' : 8,    # number of particles
    'iterations' : 7,    # PSO iterations per plan call  (budget = 8×7 = 56)
    'w'          : 0.6,  # inertia weight
    'c1'         : 1.5,  # cognitive coefficient
    'c2'         : 2.0,  # social coefficient
}


# ─── Planner ──────────────────────────────────────────────────────────────────
def planner(start_state, V, maze, cfg):
    """PSO lookahead planner. Returns estimated value in [0, 1]."""
    goal    = cfg.get('goal', GOAL)
    gamma   = cfg.get('gamma', 0.99)
    if start_state == goal:
        return 0.0

    horizon    = cfg['horizon']
    swarm_size = cfg['swarm_size']
    iterations = cfg['iterations']
    w, c1, c2  = cfg['w'], cfg['c1'], cfg['c2']
    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    # Initialize particles as integer action sequences, velocities as continuous
    swarm      = [np.random.randint(0, 4, horizon) for _ in range(swarm_size)]
    velocities = [np.random.uniform(-1.0, 1.0, horizon) for _ in range(swarm_size)]
    pbest      = [p.copy() for p in swarm]
    pbest_fits = [-1e9] * swarm_size
    gbest_fit  = -1e9
    gbest      = swarm[0].copy()

    for _ in range(iterations):
        # Evaluate and update personal / global bests
        for i in range(swarm_size):
            fit = simulate_sequence(start_state, swarm[i], V, maze, gamma, horizon, goal,
                                    step_cost=step_cost, goal_reward=goal_reward)
            if fit > pbest_fits[i]:
                pbest_fits[i] = fit
                pbest[i] = swarm[i].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest = swarm[i].copy()

        # Update velocities and clamp positions to discrete action space [0,3]
        for i in range(swarm_size):
            r1 = np.random.random(horizon)
            r2 = np.random.random(horizon)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - swarm[i])
                + c2 * r2 * (gbest  - swarm[i])
            )
            swarm[i] = np.clip(np.round(swarm[i] + velocities[i]), 0, 3).astype(int)

    return max(0.0, min(1.0, float(gbest_fit)))


# ─── Post-hoc instrumented analysis (no effect on training) ───────────────────
def convergence_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Best-so-far V(s') after each fitness evaluation, averaged over n_samples PSO runs.
    Used to visualise optimiser convergence (Kennedy & Eberhart 1995 style).
    Returns array of shape (swarm_size × iterations,).
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    budget     = cfg['swarm_size'] * cfg['iterations']
    if start_state == goal:
        return np.zeros(budget)

    horizon    = cfg['horizon']
    swarm_size = cfg['swarm_size']
    iterations = cfg['iterations']
    w, c1, c2  = cfg['w'], cfg['c1'], cfg['c2']
    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_samples):
        swarm      = [rng.integers(0, 4, horizon) for _ in range(swarm_size)]
        velocities = [rng.uniform(-1.0, 1.0, horizon) for _ in range(swarm_size)]
        pbest      = [p.copy() for p in swarm]
        pbest_fits = [-1e9] * swarm_size
        gbest_fit  = 0.0
        gbest      = swarm[0].copy()
        evals = []
        for _ in range(iterations):
            for i in range(swarm_size):
                fit = simulate_sequence(start_state, swarm[i], V, maze, gamma, horizon, goal,
                                        step_cost=step_cost, goal_reward=goal_reward)
                if fit > pbest_fits[i]:
                    pbest_fits[i] = fit; pbest[i] = swarm[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit; gbest = swarm[i].copy()
                evals.append(gbest_fit)
            for i in range(swarm_size):
                r1 = rng.random(horizon); r2 = rng.random(horizon)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest[i] - swarm[i])
                                 + c2 * r2 * (gbest    - swarm[i]))
                swarm[i] = np.clip(np.round(swarm[i] + velocities[i]), 0, 3).astype(int)
        curves.append(evals)
    return np.mean(curves, axis=0)


def diversity_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Mean population diversity (std of particle positions across dimensions) per iteration.
    Črepinšek et al. (2013) style exploration–exploitation balance metric.
    Returns array of shape (iterations,).
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return np.zeros(cfg['iterations'])

    horizon    = cfg['horizon']
    swarm_size = cfg['swarm_size']
    iterations = cfg['iterations']
    w, c1, c2  = cfg['w'], cfg['c1'], cfg['c2']
    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    rng = np.random.default_rng(seed)
    div_curves = []
    for _ in range(n_samples):
        swarm      = [rng.integers(0, 4, horizon) for _ in range(swarm_size)]
        velocities = [rng.uniform(-1.0, 1.0, horizon) for _ in range(swarm_size)]
        pbest      = [p.copy() for p in swarm]
        pbest_fits = [-1e9] * swarm_size
        gbest_fit  = 0.0
        gbest      = swarm[0].copy()
        divs = []
        for _ in range(iterations):
            for i in range(swarm_size):
                fit = simulate_sequence(start_state, swarm[i], V, maze, gamma, horizon, goal,
                                        step_cost=step_cost, goal_reward=goal_reward)
                if fit > pbest_fits[i]:
                    pbest_fits[i] = fit; pbest[i] = swarm[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit; gbest = swarm[i].copy()
            divs.append(float(np.array(swarm, dtype=float).std(axis=0).mean()))
            for i in range(swarm_size):
                r1 = rng.random(horizon); r2 = rng.random(horizon)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest[i] - swarm[i])
                                 + c2 * r2 * (gbest    - swarm[i]))
                swarm[i] = np.clip(np.round(swarm[i] + velocities[i]), 0, 3).astype(int)
        div_curves.append(divs)
    return np.mean(div_curves, axis=0)


# ─── Standalone runner ────────────────────────────────────────────────────────
def run(name='PSO', train_mazes=None, test_mazes=None, verbose=True, **hp):
    """
    Train and test PSO-TD(λ). Returns results dict from core.run_experiment.
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
    defaults['verbose'] = verbose   # CLI verbose always wins

    results = run_experiment(
        name, train_mazes, test_mazes, planner, PSO_CFG.copy(), **defaults
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
    plt.imshow(v_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='V(s)')
    plt.title('PSO-TD(λ): Learned value function V(s)')
    plt.savefig('pso_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # TD errors over training
    plt.figure(figsize=(10, 3))
    plt.plot(results['deltas'], alpha=0.6, linewidth=0.5)
    plt.title('PSO-TD(λ): TD errors (δ) during training')
    plt.xlabel('Update step')
    plt.ylabel('δ')
    plt.savefig('pso_deltas.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Rolling training success rate
    window = 50
    succ = np.array(results['train_successes'], dtype=float)
    rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rolling)
    plt.axhline(np.mean(results['test_successes']), color='red',
                linestyle='--', label=f"Test rate = {np.mean(results['test_successes']):.2f}")
    plt.title(f'PSO-TD(λ): Training success rate (rolling window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('pso_train_success.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Planner return distribution
    pv = results['plan_values_train']
    plt.figure(figsize=(7, 4))
    plt.hist(pv, bins=40, edgecolor='black', color='#2196F3', alpha=0.7)
    plt.axvline(np.mean(pv), color='red', linestyle='--',
                label=f'Media = {np.mean(pv):.3f}')
    plt.title("PSO-TD(λ): Distribución de V(s') estimado por PSO (train)")
    plt.xlabel("V(s') estimado"); plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('pso_planner_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Training steps boxplot
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(results['train_steps'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][0].set_alpha(0.7)
    plt.title('PSO-TD(λ): Variabilidad de pasos por episodio (train)')
    plt.ylabel('Pasos por episodio')
    plt.xticks([1], ['PSO'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('pso_steps_box.png', dpi=150, bbox_inches='tight')
    plt.show()
