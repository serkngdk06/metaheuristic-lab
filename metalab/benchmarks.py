"""Standard benchmark functions for testing optimisation algorithms.

Every function is vectorised: it takes an array ``P`` of shape
(n_points, dim) and returns an array of shape (n_points,).
All problems are *minimisation* problems.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Benchmark:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: tuple          # (low, high) applied to every dimension
    optimum: float         # known global minimum value
    description: str
    any_dim: bool = True   # False -> only defined for 2D


def sphere(P):
    return np.sum(P ** 2, axis=1)


def rastrigin(P):
    return 10 * P.shape[1] + np.sum(P ** 2 - 10 * np.cos(2 * np.pi * P), axis=1)


def ackley(P):
    d = P.shape[1]
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(P ** 2, axis=1) / d))
    b = -np.exp(np.sum(np.cos(2 * np.pi * P), axis=1) / d)
    return a + b + 20 + np.e


def rosenbrock(P):
    return np.sum(100 * (P[:, 1:] - P[:, :-1] ** 2) ** 2 + (1 - P[:, :-1]) ** 2, axis=1)


def griewank(P):
    i = np.arange(1, P.shape[1] + 1)
    return 1 + np.sum(P ** 2, axis=1) / 4000 - np.prod(np.cos(P / np.sqrt(i)), axis=1)


def himmelblau(P):
    x, y = P[:, 0], P[:, 1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


BENCHMARKS = {
    b.name: b
    for b in [
        Benchmark("Sphere", sphere, (-5.12, 5.12), 0.0,
                  "Tek tepeli, en basit fonksiyon. Minimum (0,0)."),
        Benchmark("Rastrigin", rastrigin, (-5.12, 5.12), 0.0,
                  "Çok sayıda yerel minimum. Algoritmaları tuzağa düşürür."),
        Benchmark("Ackley", ackley, (-5.0, 5.0), 0.0,
                  "Düz dış bölge + derin merkez çukuru. Minimum (0,0)."),
        Benchmark("Rosenbrock", rosenbrock, (-2.0, 2.0), 0.0,
                  "Muz şeklinde dar vadi. Minimum (1,1). Yakınsama zordur."),
        Benchmark("Griewank", griewank, (-10.0, 10.0), 0.0,
                  "Dalgalı yüzey, çok sayıda yerel minimum."),
        Benchmark("Himmelblau", himmelblau, (-5.0, 5.0), 0.0,
                  "4 eşit global minimum. Sürünün hangisini seçtiğini izleyin!",
                  any_dim=False),
    ]
}
