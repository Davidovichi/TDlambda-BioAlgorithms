"""
TDLambda_ga.py — Genetic Algorithm + TD(λ) for maze navigation.

GA is used as a lookahead planner. A population of action sequences evolves
via tournament selection, one-point crossover, and bit-level mutation.

Algorithm:
  Selection : 3-way tournament (pick best of 3 random individuals)
  Crossover : one-point crossover (random cut between positions 1..horizon-1)
  Mutation  : each gene independently flipped to a random action with prob mutation_rate
  Elitism   : best individual from previous generation always survives

Budget: pop_size (init) + pop_size × generations = 8 × (1 + 6) = 56 fitness
        evaluations per plan call (población inicial + población completa
        re-evaluada en cada generación, élite incluida).
        MISMO presupuesto que PSO, ABC, DE y ROLLOUT (56).

Run standalone:  python TDLambda_ga.py
Run via run_all: (importado automáticamente como 'GA')
"""

import numpy as np
import matplotlib.pyplot as plt
from maze_env import (
    simulate_sequence, GOAL, GRID_ROWS, GRID_COLS,
    TRAIN_CSV, TEST_CSV, load_mazes, START
)
from core import run_experiment

# ─── GA hyperparameters ───────────────────────────────────────────────────────
GA_CFG = {
    'horizon'      : 20,   # lookahead depth
    'pop_size'     : 8,    # population size
    'generations'  : 6,    # GA generations  (budget = 8 + 8×6 = 56)
    'mutation_rate': 0.17, # per-gene mutation probability
}


# ─── Planner ──────────────────────────────────────────────────────────────────
def planner(start_state, V, maze, cfg):
    """GA lookahead planner with elitism. Returns estimated value in [0, 1]."""
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return 0.0

    horizon    = cfg['horizon']
    pop_size   = cfg['pop_size']
    generations = cfg['generations']
    mut_rate   = cfg['mutation_rate']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(ind):
        return simulate_sequence(start_state, ind, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    def _tournament(pop, fits, k=3):
        """k-way tournament: returns index of winner."""
        idxs = np.random.choice(len(pop), min(k, len(pop)), replace=False)
        return idxs[int(np.argmax([fits[i] for i in idxs]))]

    # Initialise population as integer action sequences
    pop  = [np.random.randint(0, 4, horizon) for _ in range(pop_size)]
    fits = [_fitness(ind) for ind in pop]

    best_idx = int(np.argmax(fits))
    best_fit = fits[best_idx]
    best_ind = pop[best_idx].copy()

    for _ in range(generations):
        new_pop = [best_ind.copy()]  # elitism: carry forward best

        while len(new_pop) < pop_size:
            # Parent selection via tournament
            i1 = _tournament(pop, fits)
            i2 = _tournament(pop, fits)
            p1, p2 = pop[i1].copy(), pop[i2].copy()

            # One-point crossover
            cut = np.random.randint(1, horizon)
            c1 = np.concatenate([p1[:cut], p2[cut:]])
            c2 = np.concatenate([p2[:cut], p1[cut:]])

            # Mutation
            for child in (c1, c2):
                mask = np.random.random(horizon) < mut_rate
                child[mask] = np.random.randint(0, 4, mask.sum())
                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break

        pop  = new_pop[:pop_size]
        fits = [_fitness(ind) for ind in pop]

        cur_best_idx = int(np.argmax(fits))
        if fits[cur_best_idx] > best_fit:
            best_fit = fits[cur_best_idx]
            best_ind = pop[cur_best_idx].copy()

    return max(0.0, min(1.0, float(best_fit)))


# ─── Post-hoc instrumented analysis (no effect on training) ───────────────────
def convergence_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Best-so-far V(s') after each fitness evaluation for GA with elitism.
    Holland (1975) / Goldberg (1989) style convergence. Returns array of
    shape (pop_size + pop_size × generations,) including init evaluations.
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        budget = cfg['pop_size'] * (1 + cfg['generations'])
        return np.zeros(budget)

    horizon     = cfg['horizon']
    pop_size    = cfg['pop_size']
    generations = cfg['generations']
    mut_rate    = cfg['mutation_rate']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(ind):
        return simulate_sequence(start_state, ind, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    def _tournament(pop, fits, k=3):
        idxs = np.random.choice(len(pop), min(k, len(pop)), replace=False)
        return idxs[int(np.argmax([fits[i] for i in idxs]))]

    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_samples):
        np.random.seed(int(rng.integers(0, 2**31)))
        pop  = [np.random.randint(0, 4, horizon) for _ in range(pop_size)]
        fits = [_fitness(ind) for ind in pop]
        best_fit = max(fits); best_ind = pop[int(np.argmax(fits))].copy()
        evals = [best_fit] * pop_size      # one entry per init eval
        for _ in range(generations):
            new_pop = [best_ind.copy()]
            while len(new_pop) < pop_size:
                i1 = _tournament(pop, fits); i2 = _tournament(pop, fits)
                p1, p2 = pop[i1].copy(), pop[i2].copy()
                cut = np.random.randint(1, horizon)
                c1 = np.concatenate([p1[:cut], p2[cut:]])
                c2 = np.concatenate([p2[:cut], p1[cut:]])
                for child in (c1, c2):
                    mask = np.random.random(horizon) < mut_rate
                    child[mask] = np.random.randint(0, 4, mask.sum())
                    new_pop.append(child)
                    if len(new_pop) >= pop_size:
                        break
            pop  = new_pop[:pop_size]
            fits = [_fitness(ind) for ind in pop]
            gen_best = max(fits)
            if gen_best > best_fit:
                best_fit = gen_best; best_ind = pop[int(np.argmax(fits))].copy()
            evals.extend([best_fit] * pop_size)
        curves.append(evals)
    max_len = max(len(c) for c in curves)
    padded  = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    return np.mean(padded, axis=0)


def diversity_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Unique-individual ratio (unique chromosomes / pop_size) per generation.
    Classic GA diversity metric from Goldberg (1989).
    Returns array of shape (generations,).
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return np.zeros(cfg['generations'])

    horizon     = cfg['horizon']
    pop_size    = cfg['pop_size']
    generations = cfg['generations']
    mut_rate    = cfg['mutation_rate']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(ind):
        return simulate_sequence(start_state, ind, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    def _tournament(pop, fits, k=3):
        idxs = np.random.choice(len(pop), min(k, len(pop)), replace=False)
        return idxs[int(np.argmax([fits[i] for i in idxs]))]

    rng = np.random.default_rng(seed)
    div_curves = []
    for _ in range(n_samples):
        np.random.seed(int(rng.integers(0, 2**31)))
        pop  = [np.random.randint(0, 4, horizon) for _ in range(pop_size)]
        fits = [_fitness(ind) for ind in pop]
        best_ind = pop[int(np.argmax(fits))].copy()
        divs = []
        for _ in range(generations):
            new_pop = [best_ind.copy()]
            while len(new_pop) < pop_size:
                i1 = _tournament(pop, fits); i2 = _tournament(pop, fits)
                p1, p2 = pop[i1].copy(), pop[i2].copy()
                cut = np.random.randint(1, horizon)
                c1 = np.concatenate([p1[:cut], p2[cut:]])
                c2 = np.concatenate([p2[:cut], p1[cut:]])
                for child in (c1, c2):
                    mask = np.random.random(horizon) < mut_rate
                    child[mask] = np.random.randint(0, 4, mask.sum())
                    new_pop.append(child)
                    if len(new_pop) >= pop_size:
                        break
            pop  = new_pop[:pop_size]
            fits = [_fitness(ind) for ind in pop]
            best_ind = pop[int(np.argmax(fits))].copy()
            unique = len({tuple(ind) for ind in pop})
            divs.append(unique / pop_size)
        div_curves.append(divs)
    return np.mean(div_curves, axis=0)


# ─── Standalone runner ────────────────────────────────────────────────────────
def run(name='GA', train_mazes=None, test_mazes=None, verbose=True, **hp):
    """
    Train and test GA-TD(λ). Returns results dict from core.run_experiment.
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
        name, train_mazes, test_mazes, planner, GA_CFG.copy(), **defaults
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
    plt.imshow(v_matrix, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label='V(s)')
    plt.title('GA-TD(λ): Learned value function V(s)')
    plt.savefig('ga_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # TD errors
    plt.figure(figsize=(10, 3))
    plt.plot(results['deltas'], alpha=0.6, linewidth=0.5, color='green')
    plt.title('GA-TD(λ): TD errors (δ) during training')
    plt.xlabel('Update step')
    plt.ylabel('δ')
    plt.savefig('ga_deltas.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Rolling training success
    window = 50
    succ    = np.array(results['train_successes'], dtype=float)
    rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rolling, color='green')
    plt.axhline(np.mean(results['test_successes']), color='red',
                linestyle='--', label=f"Test rate = {np.mean(results['test_successes']):.2f}")
    plt.title(f'GA-TD(λ): Training success rate (rolling window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('ga_train_success.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plan values distribution
    pv = results['plan_values_train']
    plt.figure(figsize=(7, 4))
    plt.hist(pv, bins=40, edgecolor='black', color='#4CAF50', alpha=0.7)
    plt.axvline(np.mean(pv), color='red', linestyle='--',
                label=f'Media = {np.mean(pv):.3f}')
    plt.title("GA-TD(λ): Distribución de V(s') estimado por GA (train)")
    plt.xlabel("V(s') estimado"); plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('ga_plan_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Training steps boxplot
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(results['train_steps'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    plt.title('GA-TD(λ): Variabilidad de pasos por episodio (train)')
    plt.ylabel('Pasos por episodio')
    plt.xticks([1], ['GA'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('ga_steps_box.png', dpi=150, bbox_inches='tight')
    plt.show()
