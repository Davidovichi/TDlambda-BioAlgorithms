"""
TDLambda_de.py — Differential Evolution + TD(λ) for maze navigation.

DE is used as a lookahead planner. A population of action sequences evolves
over a short horizon using DE/rand/1/bin — the classic and most robust DE variant.

Algorithm (DE/rand/1/bin):
  Mutation  : for each target vector x_i, select three distinct random
              individuals a, b, c (a ≠ b ≠ c ≠ i) and compute
              mutant v = a + F·(b − c)
  Crossover : trial[j] = v[j] if U[0,1] < CR, else x_i[j]
              (at least one component always taken from mutant)
  Selection : greedy one-to-one: keep whichever of x_i or trial is fitter

Representation: population lives in [0, 4); actions are floor+clipped to {0,1,2,3}.

Budget: pop_size (init) + pop_size × iterations = 8 × (1 + 6) = 56 fitness
        evaluations per plan call (la población inicial se evalúa una vez y
        luego 1 trial por individuo por generación).
        MISMO presupuesto que PSO, ABC, GA y ROLLOUT (56).

Run standalone:  python TDLambda_de.py
Run via run_all: (importado automáticamente como 'DE')
"""

import numpy as np
import matplotlib.pyplot as plt
from maze_env import (
    simulate_sequence, GOAL, GRID_ROWS, GRID_COLS,
    TRAIN_CSV, TEST_CSV, load_mazes, START
)
from core import run_experiment

# ─── DE hyperparameters ───────────────────────────────────────────────────────
DE_CFG = {
    'horizon'   : 20,   # lookahead depth
    'pop_size'  : 8,    # population size
    'iterations': 6,    # DE generations  (budget = 8 + 8×6 = 56)
    'F'         : 0.8,  # mutation scaling factor
    'CR'        : 0.9,  # crossover rate (binomial)
}


# ─── Planner ──────────────────────────────────────────────────────────────────
def planner(start_state, V, maze, cfg):
    """DE/rand/1/bin lookahead planner. Returns estimated value in [0, 1]."""
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return 0.0

    horizon    = cfg['horizon']
    pop_size   = cfg['pop_size']
    iterations = cfg['iterations']
    F          = cfg['F']
    CR         = cfg['CR']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fitness(ind):
        actions = np.clip(np.floor(ind).astype(int), 0, 3)
        return simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    # Initialise population in [0, 4)
    pop  = np.random.uniform(0.0, 4.0, (pop_size, horizon))
    fits = np.array([_fitness(pop[i]) for i in range(pop_size)])

    best_fit = float(np.max(fits))

    for _ in range(iterations):
        for i in range(pop_size):
            # Select three distinct random indices ≠ i
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)

            # Mutation: DE/rand/1
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, 0.0, 3.999)

            # Crossover: binomial — guarantee at least one dimension from mutant
            cross_mask = np.random.random(horizon) < CR
            cross_mask[np.random.randint(0, horizon)] = True
            trial = np.where(cross_mask, mutant, pop[i])

            # Greedy selection
            f_trial = _fitness(trial)
            if f_trial >= fits[i]:
                pop[i]  = trial
                fits[i] = f_trial

        cur_best = float(np.max(fits))
        if cur_best > best_fit:
            best_fit = cur_best

    return max(0.0, min(1.0, best_fit))


# ─── Post-hoc instrumented analysis (no effect on training) ───────────────────
def convergence_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Best-so-far V(s') after each fitness evaluation for DE/rand/1/bin.
    Storn & Price (1997) convergence curve. Returns array of shape
    (pop_size + pop_size × iterations,) = all evaluations including init.
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        budget = cfg['pop_size'] * (1 + cfg['iterations'])
        return np.zeros(budget)

    horizon    = cfg['horizon']
    pop_size   = cfg['pop_size']
    iterations = cfg['iterations']
    F_val      = cfg['F']
    CR         = cfg['CR']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fit(ind):
        actions = np.clip(np.floor(ind).astype(int), 0, 3)
        return simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_samples):
        pop  = rng.uniform(0.0, 4.0, (pop_size, horizon))
        fits = np.array([_fit(pop[i]) for i in range(pop_size)])
        best_fit = float(np.max(fits))
        evals = [best_fit] * pop_size       # one entry per init eval
        for _ in range(iterations):
            for i in range(pop_size):
                cands = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(cands, 3, replace=False)
                mutant = np.clip(pop[a] + F_val * (pop[b] - pop[c]), 0.0, 3.999)
                mask = rng.random(horizon) < CR
                mask[rng.integers(0, horizon)] = True
                trial = np.where(mask, mutant, pop[i])
                f_tr = _fit(trial)
                if f_tr >= fits[i]:
                    pop[i] = trial; fits[i] = f_tr
                best_fit = max(best_fit, f_tr)
                evals.append(best_fit)
        curves.append(evals)
    max_len = max(len(c) for c in curves)
    padded  = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    return np.mean(padded, axis=0)


def diversity_curve(start_state, V, maze, cfg, n_samples=8, seed=99):
    """
    Mean population diversity (std across individuals per gene) per DE iteration.
    Returns array of shape (iterations,).
    """
    goal       = cfg.get('goal', GOAL)
    gamma      = cfg.get('gamma', 0.99)
    if start_state == goal:
        return np.zeros(cfg['iterations'])

    horizon    = cfg['horizon']
    pop_size   = cfg['pop_size']
    iterations = cfg['iterations']
    F_val      = cfg['F']
    CR         = cfg['CR']

    step_cost   = cfg.get('step_cost', 0.0)
    goal_reward = cfg.get('goal_reward', 1.0)

    def _fit(ind):
        actions = np.clip(np.floor(ind).astype(int), 0, 3)
        return simulate_sequence(start_state, actions, V, maze, gamma, horizon, goal,
                                 step_cost=step_cost, goal_reward=goal_reward)

    rng = np.random.default_rng(seed)
    div_curves = []
    for _ in range(n_samples):
        pop  = rng.uniform(0.0, 4.0, (pop_size, horizon))
        fits = np.array([_fit(pop[i]) for i in range(pop_size)])
        divs = []
        for _ in range(iterations):
            for i in range(pop_size):
                cands = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(cands, 3, replace=False)
                mutant = np.clip(pop[a] + F_val * (pop[b] - pop[c]), 0.0, 3.999)
                mask = rng.random(horizon) < CR
                mask[rng.integers(0, horizon)] = True
                trial = np.where(mask, mutant, pop[i])
                f_tr = _fit(trial)
                if f_tr >= fits[i]:
                    pop[i] = trial; fits[i] = f_tr
            divs.append(float(pop.std(axis=0).mean()))
        div_curves.append(divs)
    return np.mean(div_curves, axis=0)


# ─── Standalone runner ────────────────────────────────────────────────────────
def run(name='DE', train_mazes=None, test_mazes=None, verbose=True, **hp):
    """
    Train and test DE-TD(λ). Returns results dict from core.run_experiment.
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
        name, train_mazes, test_mazes, planner, DE_CFG.copy(), **defaults
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
    plt.imshow(v_matrix, cmap='inferno', vmin=0, vmax=1)
    plt.colorbar(label='V(s)')
    plt.title('DE-TD(λ): Learned value function V(s)')
    plt.savefig('de_v_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # TD errors
    plt.figure(figsize=(10, 3))
    plt.plot(results['deltas'], alpha=0.6, linewidth=0.5, color='purple')
    plt.title('DE-TD(λ): TD errors (δ) during training')
    plt.xlabel('Update step')
    plt.ylabel('δ')
    plt.savefig('de_deltas.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Rolling training success
    window = 50
    succ    = np.array(results['train_successes'], dtype=float)
    rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rolling, color='purple')
    plt.axhline(np.mean(results['test_successes']), color='red',
                linestyle='--', label=f"Test rate = {np.mean(results['test_successes']):.2f}")
    plt.title(f'DE-TD(λ): Training success rate (rolling window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('de_train_success.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Fitness landscape: distribution of mutation magnitudes over a sample population
    sample = np.random.uniform(0.0, 4.0, (1000, DE_CFG['horizon']))
    F = DE_CFG['F']
    diffs = []
    for _ in range(200):
        a, b, c = np.random.choice(1000, 3, replace=False)
        diffs.append(np.linalg.norm(F * (sample[b] - sample[c])))
    plt.figure(figsize=(7, 4))
    plt.hist(diffs, bins=30, edgecolor='black', color='purple', alpha=0.7)
    plt.title(f'DE: Distribution of mutation step magnitudes (F={F})')
    plt.xlabel('||F·(b−c)||')
    plt.ylabel('Count')
    plt.savefig('de_mutation_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plan values distribution
    pv = results['plan_values_train']
    plt.figure(figsize=(7, 4))
    plt.hist(pv, bins=40, edgecolor='black', color='#9C27B0', alpha=0.7)
    plt.axvline(np.mean(pv), color='red', linestyle='--',
                label=f'Media = {np.mean(pv):.3f}')
    plt.title("DE-TD(λ): Distribución de V(s') estimado por DE (train)")
    plt.xlabel("V(s') estimado"); plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('de_plan_dist.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Training steps boxplot
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(results['train_steps'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#9C27B0')
    bp['boxes'][0].set_alpha(0.7)
    plt.title('DE-TD(λ): Variabilidad de pasos por episodio (train)')
    plt.ylabel('Pasos por episodio')
    plt.xticks([1], ['DE'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('de_steps_box.png', dpi=150, bbox_inches='tight')
    plt.show()
