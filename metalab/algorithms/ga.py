"""Real-coded Genetic Algorithm (Holland, 1975; Goldberg, 1989).

One generation = selection -> crossover -> mutation -> elitism.
Same step-able interface as :class:`PSO` so both can be raced side by side.
"""

import numpy as np


class GeneticAlgorithm:
    """Minimise ``func`` with a real-valued genetic algorithm.

    Parameters
    ----------
    selection : {"tournament", "roulette"}
        Tournament (size ``tournament_k``) or fitness-proportional roulette.
    crossover_rate : float
        Probability that a parent pair is recombined (arithmetic / BLX-alpha).
    mutation_rate : float
        Per-gene probability of a Gaussian mutation.
    mutation_scale : float
        Mutation std as a fraction of the search range.
    elitism : int
        Number of best individuals copied unchanged to the next generation.
    """

    name = "GA"

    def __init__(self, func, dim, bounds, pop_size=30, max_iter=100,
                 crossover_rate=0.9, mutation_rate=0.1, mutation_scale=0.1,
                 selection="tournament", tournament_k=3, elitism=2,
                 blx_alpha=0.3, seed=None):
        self.func = func
        self.dim = int(dim)
        self.low, self.high = float(bounds[0]), float(bounds[1])
        self.n = int(pop_size)
        self.max_iter = int(max_iter)
        self.pc = float(crossover_rate)
        self.pm = float(mutation_rate)
        self.sigma = float(mutation_scale) * (self.high - self.low)
        self.selection = selection
        self.k = int(tournament_k)
        self.elitism = int(min(elitism, self.n - 1))
        self.alpha = float(blx_alpha)
        self.rng = np.random.default_rng(seed)

        self.pop = self.rng.uniform(self.low, self.high, (self.n, self.dim))
        self.fitness = self.func(self.pop)
        best = int(np.argmin(self.fitness))
        self.gbest_x = self.pop[best].copy()
        self.gbest_f = float(self.fitness[best])

        self.iteration = 0
        self.evaluations = self.n
        self.history = [self.gbest_f]
        self.mean_history = [float(np.mean(self.fitness))]

    @property
    def positions(self):
        return self.pop

    @property
    def done(self):
        return self.iteration >= self.max_iter

    # ---------------------------- operators --------------------------- #
    def _select(self, count):
        if self.selection == "roulette":
            # Minimisation -> convert to "bigger is better" weights
            f = self.fitness
            weights = (f.max() - f) + 1e-12
            probs = weights / weights.sum()
            return self.rng.choice(self.n, size=count, p=probs)
        # Tournament: pick k random individuals, keep the best one
        contenders = self.rng.integers(0, self.n, (count, self.k))
        winners = np.argmin(self.fitness[contenders], axis=1)
        return contenders[np.arange(count), winners]

    def _crossover(self, p1, p2):
        """BLX-alpha: child genes drawn from the parents' range, slightly extended."""
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        span = hi - lo
        c1 = self.rng.uniform(lo - self.alpha * span, hi + self.alpha * span)
        c2 = self.rng.uniform(lo - self.alpha * span, hi + self.alpha * span)
        return c1, c2

    def _mutate(self, children):
        mask = self.rng.random(children.shape) < self.pm
        children[mask] += self.rng.normal(0, self.sigma, mask.sum())
        return children

    # ------------------------------------------------------------------ #
    def step(self):
        order = np.argsort(self.fitness)
        elites = self.pop[order[: self.elitism]].copy()

        n_children = self.n - self.elitism
        parents = self.pop[self._select(n_children + (n_children % 2))]
        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i + 1]
            if self.rng.random() < self.pc:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.extend([c1, c2])
        children = self._mutate(np.array(children[:n_children]))
        children = np.clip(children, self.low, self.high)

        child_f = self.func(children)
        self.evaluations += len(children)
        self.pop = np.vstack([elites, children])
        self.fitness = np.concatenate([self.fitness[order[: self.elitism]], child_f])

        best = int(np.argmin(self.fitness))
        if self.fitness[best] < self.gbest_f:
            self.gbest_f = float(self.fitness[best])
            self.gbest_x = self.pop[best].copy()

        self.iteration += 1
        self.history.append(self.gbest_f)
        self.mean_history.append(float(np.mean(self.fitness)))
        return {"iteration": self.iteration}

    def run(self, callback=None):
        while not self.done:
            self.step()
            if callback:
                callback(self)
        return self.gbest_x, self.gbest_f
