"""Particle Swarm Optimization (Kennedy & Eberhart, 1995).

Velocity and position update for every particle i and dimension d:

    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x = x + v

The optimizer is *step-able*: call :meth:`step` once per iteration so a GUI
can redraw the swarm between iterations (animation, step-by-step teaching).
"""

import numpy as np


class PSO:
    """Minimise ``func`` with a particle swarm.

    Parameters
    ----------
    func : callable
        Vectorised objective, ``func(P)`` with ``P.shape == (n, dim)``
        returning ``(n,)`` fitness values (smaller is better).
    dim : int
        Number of decision variables.
    bounds : (float, float)
        Search interval used for every dimension.
    n_particles, max_iter : int
    w_max, w_min : float
        Inertia weight decreases linearly from ``w_max`` to ``w_min``
        (Shi & Eberhart, 1998). Set them equal for a constant ``w``.
    c1, c2 : float
        Cognitive (own memory) and social (swarm memory) coefficients.
    v_max_ratio : float or None
        Velocity is clamped to ``v_max_ratio * (high - low)``. ``None`` = no clamp.
    fixed_r : float or None
        If given, r1 = r2 = fixed_r (as in the lecture's hand calculation).
        ``None`` draws new random numbers in [0, 1) every step (standard PSO).
    init_positions : array-like or None
        Optional starting positions, shape (n_particles, dim).
    seed : int or None
        Random seed for reproducible runs.
    """

    name = "PSO"

    def __init__(self, func, dim, bounds, n_particles=30, max_iter=100,
                 w_max=0.9, w_min=0.4, c1=1.494, c2=1.494, v_max_ratio=0.2,
                 fixed_r=None, init_positions=None, clip_to_bounds=True, seed=None):
        self.func = func
        self.dim = int(dim)
        self.low, self.high = float(bounds[0]), float(bounds[1])
        self.n = int(n_particles)
        self.max_iter = int(max_iter)
        self.w_max, self.w_min = float(w_max), float(w_min)
        self.c1, self.c2 = float(c1), float(c2)
        self.fixed_r = fixed_r
        self.clip = clip_to_bounds
        self.rng = np.random.default_rng(seed)

        span = self.high - self.low
        self.v_max = None if v_max_ratio is None else v_max_ratio * span

        if init_positions is not None:
            self.x = np.array(init_positions, dtype=float).reshape(-1, self.dim)
            self.n = self.x.shape[0]
        else:
            self.x = self.rng.uniform(self.low, self.high, (self.n, self.dim))
        self.v = np.zeros((self.n, self.dim))

        self.fitness = self.func(self.x)
        self.pbest_x = self.x.copy()
        self.pbest_f = self.fitness.copy()
        best = int(np.argmin(self.pbest_f))
        self.gbest_x = self.pbest_x[best].copy()
        self.gbest_f = float(self.pbest_f[best])

        self.iteration = 0
        self.evaluations = self.n
        self.history = [self.gbest_f]           # best fitness per iteration
        self.mean_history = [float(np.mean(self.fitness))]

    # ------------------------------------------------------------------ #
    @property
    def positions(self):
        return self.x

    @property
    def done(self):
        return self.iteration >= self.max_iter

    def current_w(self):
        if self.max_iter <= 1:
            return self.w_max
        t = min(self.iteration / (self.max_iter - 1), 1.0)
        return self.w_max - (self.w_max - self.w_min) * t

    def step(self):
        """Run one iteration and return a dict describing what happened.

        The dict contains every term of the velocity formula so the teaching
        tab can print the calculation with real numbers.
        """
        w = self.current_w()
        shape = (self.n, self.dim)
        if self.fixed_r is None:
            r1, r2 = self.rng.random(shape), self.rng.random(shape)
        else:
            r1 = np.full(shape, float(self.fixed_r))
            r2 = np.full(shape, float(self.fixed_r))

        x_old, v_old = self.x.copy(), self.v.copy()
        gbest_used = self.gbest_x.copy()
        pbest_used = self.pbest_x.copy()

        inertia = w * v_old
        cognitive = self.c1 * r1 * (self.pbest_x - x_old)
        social = self.c2 * r2 * (gbest_used - x_old)
        self.v = inertia + cognitive + social
        if self.v_max is not None:
            self.v = np.clip(self.v, -self.v_max, self.v_max)

        self.x = x_old + self.v
        if self.clip:
            self.x = np.clip(self.x, self.low, self.high)

        # Evaluate and update personal / global bests
        self.fitness = self.func(self.x)
        self.evaluations += self.n
        improved = self.fitness < self.pbest_f
        self.pbest_x[improved] = self.x[improved]
        self.pbest_f[improved] = self.fitness[improved]
        best = int(np.argmin(self.pbest_f))
        if self.pbest_f[best] < self.gbest_f:
            self.gbest_f = float(self.pbest_f[best])
            self.gbest_x = self.pbest_x[best].copy()

        self.iteration += 1
        self.history.append(self.gbest_f)
        self.mean_history.append(float(np.mean(self.fitness)))

        return {
            "iteration": self.iteration, "w": w, "r1": r1, "r2": r2,
            "x_old": x_old, "v_old": v_old,
            "pbest_used": pbest_used, "gbest_used": gbest_used,
            "inertia": inertia, "cognitive": cognitive, "social": social,
            "v_new": self.v.copy(), "x_new": self.x.copy(),
            "fitness": self.fitness.copy(), "improved": improved,
        }

    def run(self, callback=None):
        """Run until ``max_iter``; optional ``callback(self)`` after each step."""
        while not self.done:
            self.step()
            if callback:
                callback(self)
        return self.gbest_x, self.gbest_f
