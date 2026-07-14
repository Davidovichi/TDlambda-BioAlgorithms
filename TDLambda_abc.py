"""
TDLambda_abc.py — Artificial Bee Colony + TD(λ) for maze navigation.

ABC is used as a lookahead planner. A colony of bees searches the space of
action sequences (food sources) over a short horizon.

Algorithm (standard ABC — Karaboga 2005):
  Employee phase : each bee generates a neighbour by perturbing one dimension
                   of its source; keeps the better of old vs new.
  Onlooker phase : bees choose sources probabilistically by fitness, then
                   generate neighbours the same way.
  Scout phase    : sources that have not improved after `limit` trials are
                   abandoned and replaced by a new random source.

Representation  : food sources are real vectors in [0, 4); actions are obtained
                  by floor+clip to {0,1,2,3}.  Continuous representation lets
                  the perturbation operator work smoothly.

Budget: n_employed (init)  +  iterations × (n_employed + n_onlooker)
      = 8 + 3 × (8 + 8) = 56 fitness evaluations per plan call, más scouts
        ocasionales (una fuente agotada re-muestreada cuesta +1 eval; con
        limit=6 y 3 ciclos casi nunca ocurre). Colonia simétrica estándar
        (Karaboga 2005: mitad empleadas, mitad observadoras).
        MISMO presupuesto que PSO, DE, GA y ROLLOUT (56).

Run standalone:  python TDLambda_abc.py
Run via run_all: (importado automáticamente como 'ABC')
"""

import numpy as np
import matplotlib.pyplot as plt
from maze_env import (
    simulate_sequence, GOAL, GRID_ROWS, GRID_COLS,
    TRAIN_CSV, TEST_CSV, load_mazes, START
)
from core import run_experiment

# ─── ABC hyperparameters ──────────────────────────────────────────────────────
ABC_CFG = {
    'horizon'    : 20,   # lookahead depth
    'n_employed' : 8,    # employed bees (= number of food sources)
    'n_onlooker' : 8,    # onlooker bees (colonia 50/50 estándar)
    'iterations' : 3,    # ABC cycles  (budget = 8 + 3×(8+8) = 56)
    'limit'      : 6,    # max trials before scout replaces a source (~2×ciclos)
}


# ─── Planner ──────────────────────────────────────────────────────────────────
def planner(start_state, V, maze, cfg):
    """ABC lookahead planner. Returns estimated value in [0, 1]."""
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return 0.0

    horizon    = cfg['horizon']
    n_emp      = cfg['n_employed']
    n_onl      = cfg['n_onlooker']
    iterations = cfg['iterations']
    limit      = cfg['limit']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(src):
        actions = np.clip(np.floor(src).astype(int), 0, 3)
        return simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    def _neighbour(src, other_src):
        """Generate neighbour by perturbing one random dimension toward other_src."""
        nb = src.copy()
        j = np.random.randint(0, horizon)
        phi = np.random.uniform(-1.0, 1.0)
        nb[j] = np.clip(nb[j] + phi * (nb[j] - other_src[j]), 0.0, 3.999)
        return nb

    # Initialise food sources (continuous in [0, 4))
    sources  = np.random.uniform(0.0, 4.0, (n_emp, horizon))
    fits     = np.array([_fitness(sources[i]) for i in range(n_emp)])
    trials   = np.zeros(n_emp, dtype=int)

    best_fit = float(np.max(fits))

    for _ in range(iterations):
        # ── Employee phase ────────────────────────────────────────────────────
        for i in range(n_emp):
            # pick a different source as reference
            k = np.random.choice([j for j in range(n_emp) if j != i])
            nb = _neighbour(sources[i], sources[k])
            f  = _fitness(nb)
            if f > fits[i]:
                sources[i] = nb
                fits[i]    = f
                trials[i]  = 0
            else:
                trials[i] += 1

        # ── Onlooker phase ────────────────────────────────────────────────────
        # Selection probability proportional to shifted fitness (avoids negatives)
        shifted = fits - fits.min() + 1e-9
        probs   = shifted / shifted.sum()
        for _ in range(n_onl):
            i = np.random.choice(n_emp, p=probs)
            k = np.random.choice([j for j in range(n_emp) if j != i])
            nb = _neighbour(sources[i], sources[k])
            f  = _fitness(nb)
            if f > fits[i]:
                sources[i] = nb
                fits[i]    = f
                trials[i]  = 0
            else:
                trials[i] += 1

        # ── Scout phase ───────────────────────────────────────────────────────
        for i in range(n_emp):
            if trials[i] > limit:
                sources[i] = np.random.uniform(0.0, 4.0, horizon)
                fits[i]    = _fitness(sources[i])
                trials[i]  = 0

        # Track global best
        cur_best = float(np.max(fits))
        if cur_best > best_fit:
            best_fit = cur_best

    return max(0.0, min(1.0, best_fit))


# ─── Post-hoc instrumented analysis (no effect on training) ───────────────────
def convergence_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Best-so-far V(s') after each fitness evaluation for ABC (Karaboga 2005).
    Tracks evals through init + employee + onlooker + scout phases.
    Returns 1D array with one entry per actual fitness evaluation.
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return np.zeros(cfg['n_employed'])

    horizon    = cfg['horizon']
    n_emp      = cfg['n_employed']
    n_onl      = cfg['n_onlooker']
    iterations = cfg['iterations']
    limit      = cfg['limit']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(src):
        actions = np.clip(np.floor(src).astype(int), 0, 3)
        return simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    def _neighbour(src, other, rng):
        nb = src.copy(); j = rng.integers(0, horizon)
        phi = rng.uniform(-1.0, 1.0)
        nb[j] = np.clip(nb[j] + phi * (nb[j] - other[j]), 0.0, 3.999)
        return nb

    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_samples):
        sources = rng.uniform(0.0, 4.0, (n_emp, horizon))
        fits    = np.array([_fitness(sources[i]) for i in range(n_emp)])
        trials  = np.zeros(n_emp, dtype=int)
        best_fit = float(np.max(fits))
        evals = [best_fit] * n_emp         # init evaluations
        for _ in range(iterations):
            # Employee phase
            for i in range(n_emp):
                others = [j for j in range(n_emp) if j != i]
                k = int(rng.choice(others))
                nb = _neighbour(sources[i], sources[k], rng)
                f  = _fitness(nb)
                if f > fits[i]:
                    sources[i] = nb; fits[i] = f; trials[i] = 0
                else:
                    trials[i] += 1
                best_fit = max(best_fit, f); evals.append(best_fit)
            # Onlooker phase
            shifted = fits - fits.min() + 1e-9; probs = shifted / shifted.sum()
            for _ in range(n_onl):
                i = int(rng.choice(n_emp, p=probs))
                others = [j for j in range(n_emp) if j != i]
                k = int(rng.choice(others))
                nb = _neighbour(sources[i], sources[k], rng)
                f  = _fitness(nb)
                if f > fits[i]:
                    sources[i] = nb; fits[i] = f; trials[i] = 0
                else:
                    trials[i] += 1
                best_fit = max(best_fit, f); evals.append(best_fit)
            # Scout phase
            for i in range(n_emp):
                if trials[i] > limit:
                    sources[i] = rng.uniform(0.0, 4.0, horizon)
                    fits[i]    = _fitness(sources[i])
                    trials[i]  = 0
                    best_fit   = max(best_fit, fits[i])
                    evals.append(best_fit)
        curves.append(evals)
    max_len = max(len(c) for c in curves)
    padded  = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    return np.mean(padded, axis=0)


def diversity_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Mean spread of food sources (std across sources per dimension) per cycle.
    Returns array of shape (iterations,).
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return np.zeros(cfg['iterations'])

    horizon    = cfg['horizon']
    n_emp      = cfg['n_employed']
    n_onl      = cfg['n_onlooker']
    iterations = cfg['iterations']
    limit      = cfg['limit']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(src):
        actions = np.clip(np.floor(src).astype(int), 0, 3)
        return simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    def _neighbour(src, other, rng):
        nb = src.copy(); j = rng.integers(0, horizon)
        phi = rng.uniform(-1.0, 1.0)
        nb[j] = np.clip(nb[j] + phi * (nb[j] - other[j]), 0.0, 3.999)
        return nb

    rng = np.random.default_rng(seed)
    div_curves = []
    for _ in range(n_samples):
        sources = rng.uniform(0.0, 4.0, (n_emp, horizon))
        fits    = np.array([_fitness(sources[i]) for i in range(n_emp)])
        trials  = np.zeros(n_emp, dtype=int)
        divs = []
        for _ in range(iterations):
            for i in range(n_emp):
                others = [j for j in range(n_emp) if j != i]
                k = int(rng.choice(others))
                nb = _neighbour(sources[i], sources[k], rng)
                f  = _fitness(nb)
                if f > fits[i]:
                    sources[i] = nb; fits[i] = f; trials[i] = 0
                else:
                    trials[i] += 1
            shifted = fits - fits.min() + 1e-9; probs = shifted / shifted.sum()
            for _ in range(n_onl):
                i = int(rng.choice(n_emp, p=probs))
                others = [j for j in range(n_emp) if j != i]
                k = int(rng.choice(others))
                nb = _neighbour(sources[i], sources[k], rng)
                f  = _fitness(nb)
                if f > fits[i]:
                    sources[i] = nb; fits[i] = f; trials[i] = 0
                else:
                    trials[i] += 1
            for i in range(n_emp):
                if trials[i] > limit:
                    sources[i] = rng.uniform(0.0, 4.0, horizon); fits[i] = _fitness(sources[i]); trials[i] = 0
            divs.append(float(sources.std(axis=0).mean()))
        div_curves.append(divs)
    return np.mean(div_curves, axis=0)


# ─── Standalone runner ────────────────────────────────────────────────────────
def run(name='ABC', train_mazes=None, test_mazes=None, verbose=True, **hp):
    """
    Train and test ABC-TD(λ). Returns results dict from core.run_experiment.
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
        name, train_mazes, test_mazes, planner, ABC_CFG.copy(), **defaults
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
    plt.imshow(v_matrix, cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(label='V(s)')
    plt.title('ABC-TD(λ): Learned value function V(s)')
    plt.savefig('abc_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # TD errors
    plt.figure(figsize=(10, 3))
    plt.plot(results['deltas'], alpha=0.6, linewidth=0.5, color='orange')
    plt.title('ABC-TD(λ): TD errors (δ) during training')
    plt.xlabel('Update step')
    plt.ylabel('δ')
    plt.savefig('abc_deltas.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Rolling training success
    window = 50
    succ    = np.array(results['train_successes'], dtype=float)
    rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rolling, color='orange')
    plt.axhline(np.mean(results['test_successes']), color='red',
                linestyle='--', label=f"Test rate = {np.mean(results['test_successes']):.2f}")
    plt.title(f'ABC-TD(λ): Training success rate (rolling window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('abc_train_success.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plan values distribution
    pv = results['plan_values_train']
    plt.figure(figsize=(7, 4))
    plt.hist(pv, bins=40, edgecolor='black', color='#FF9800', alpha=0.7)
    plt.axvline(np.mean(pv), color='red', linestyle='--',
                label=f'Media = {np.mean(pv):.3f}')
    plt.title("ABC-TD(λ): Distribución de V(s') estimado por ABC (train)")
    plt.xlabel("V(s') estimado"); plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('abc_plan_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Training steps boxplot
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(results['train_steps'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#FF9800')
    bp['boxes'][0].set_alpha(0.7)
    plt.title('ABC-TD(λ): Variabilidad de pasos por episodio (train)')
    plt.ylabel('Pasos por episodio')
    plt.xticks([1], ['ABC'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('abc_steps_box.png', dpi=150, bbox_inches='tight')
    plt.show()
