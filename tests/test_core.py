"""Unit tests for the algorithm core (no GUI needed).  Run:  pytest -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metalab.algorithms import PSO, GeneticAlgorithm  # noqa: E402
from metalab.benchmarks import BENCHMARKS, sphere  # noqa: E402
from metalab.expression import ExpressionError, parse_expression  # noqa: E402
from metalab.neural import MLP, BackpropTrainer, make_student_dataset, normalize  # noqa: E402


# ----------------------------------------------------------- benchmarks
@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_benchmarks_have_zero_minimum(name):
    b = BENCHMARKS[name]
    known = {"Rosenbrock": [1, 1], "Himmelblau": [3, 2]}.get(name, [0, 0])
    assert b.func(np.array([known], dtype=float))[0] == pytest.approx(b.optimum, abs=1e-9)


# ------------------------------------------------------------------ PSO
def test_pso_solves_sphere():
    pso = PSO(sphere, 5, (-5, 5), n_particles=30, max_iter=200, seed=0)
    _, f = pso.run()
    assert f < 1e-6


def test_pso_is_reproducible_with_seed():
    a = PSO(sphere, 3, (-5, 5), max_iter=20, seed=42).run()[1]
    b = PSO(sphere, 3, (-5, 5), max_iter=20, seed=42).run()[1]
    assert a == b


def test_pso_history_never_gets_worse():
    pso = PSO(BENCHMARKS["Rastrigin"].func, 2, (-5.12, 5.12), max_iter=50, seed=1)
    pso.run()
    assert np.all(np.diff(pso.history) <= 0)


def test_pso_lecture_example_finds_root():
    """Lecture example: |x^2 + 2x - 3| with particles 3, 7, 5 -> root x = 1."""
    f, _ = parse_expression("abs(x**2 + 2*x - 3)")
    pso = PSO(f, 1, (-50, 50), init_positions=[[3], [7], [5]], max_iter=100,
              w_max=0.7, w_min=0.4, v_max_ratio=None, seed=0)
    x, fx = pso.run()
    assert fx < 1e-3
    assert x[0] == pytest.approx(1.0, abs=1e-2)


def test_pso_velocity_formula_terms():
    pso = PSO(sphere, 2, (-5, 5), n_particles=4, max_iter=5, fixed_r=0.5,
              v_max_ratio=None, clip_to_bounds=False, seed=3)
    d = pso.step()
    expected = (d["w"] * d["v_old"]
                + pso.c1 * 0.5 * (d["pbest_used"] - d["x_old"])
                + pso.c2 * 0.5 * (d["gbest_used"] - d["x_old"]))
    np.testing.assert_allclose(d["v_new"], expected)
    np.testing.assert_allclose(d["x_new"], d["x_old"] + d["v_new"])


def test_inertia_decreases_linearly():
    pso = PSO(sphere, 2, (-1, 1), max_iter=11, w_max=0.9, w_min=0.4, seed=0)
    ws = []
    while not pso.done:
        ws.append(pso.step()["w"])
    assert ws[0] == pytest.approx(0.9)
    assert ws[-1] == pytest.approx(0.4)


# ------------------------------------------------------------------- GA
@pytest.mark.parametrize("selection", ["tournament", "roulette"])
def test_ga_improves(selection):
    ga = GeneticAlgorithm(sphere, 5, (-5, 5), pop_size=40, max_iter=150,
                          selection=selection, seed=0)
    start = ga.gbest_f
    ga.run()
    assert ga.gbest_f < start * 1e-2
    assert np.all(np.diff(ga.history) <= 0)  # elitism keeps the best


# ----------------------------------------------------------- expression
def test_expression_parses_2d():
    f, dim = parse_expression("x**2 + y**2")
    assert dim == 2
    assert f(np.array([[3.0, 4.0]]))[0] == pytest.approx(25)


@pytest.mark.parametrize("bad", ["__import__('os')", "x.real", "open('f')", "y + 1", "", "x +"])
def test_expression_rejects_unsafe_or_invalid(bad):
    with pytest.raises(ExpressionError):
        parse_expression(bad)


# -------------------------------------------------------- neural network
def test_mlp_parameter_count():
    assert MLP(2).n_params == 9          # the lecture network
    assert MLP(5).n_params == 4 * 5 + 1


def test_batch_forward_matches_single():
    mlp = MLP(3)
    X = np.random.default_rng(0).random((20, 2))
    P = np.random.default_rng(1).normal(size=(4, mlp.n_params))
    batch = mlp.forward_batch(X, P)
    for i in range(4):
        np.testing.assert_allclose(batch[i], mlp.forward(X, P[i]))


def test_backprop_gradient_matches_numeric():
    mlp = MLP(2)
    Xtr, ytr, *_ = make_student_dataset(50, seed=0)
    X = normalize(Xtr)
    bp = BackpropTrainer(mlp, X, ytr, lr=1e-9, seed=0)
    p0 = bp.p.copy()
    bp.step()
    analytic = (p0 - bp.p) / 1e-9
    eps = 1e-6
    numeric = np.array([(mlp.mse(X, ytr, p0 + e) - mlp.mse(X, ytr, p0 - e)) / (2 * eps)
                        for e in np.eye(mlp.n_params) * eps])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-6)


def test_pso_trains_network():
    mlp = MLP(2)
    Xtr, ytr, Xte, yte = make_student_dataset(200, noise=3, seed=1)
    X, Xt = normalize(Xtr), normalize(Xte)
    pso = PSO(mlp.fitness_function(X, ytr), mlp.n_params, (-10, 10),
              n_particles=30, max_iter=200, seed=0)
    pso.run()
    assert mlp.accuracy(Xt, yte, pso.gbest_x) > 0.85
