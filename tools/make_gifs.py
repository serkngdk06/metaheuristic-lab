"""Render the animated GIFs shown at the top of the README (no GUI needed).

    python tools/make_gifs.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from metalab.algorithms import PSO  # noqa: E402
from metalab.benchmarks import BENCHMARKS  # noqa: E402
from metalab.neural import MLP, make_student_dataset, normalize  # noqa: E402

OUT = os.path.join(ROOT, "assets")


def swarm_gif(name="Rastrigin", iters=60, path="demo_swarm.gif"):
    b = BENCHMARKS[name]
    pso = PSO(b.func, 2, b.bounds, n_particles=30, max_iter=iters, seed=3)
    g = np.linspace(*b.bounds, 200)
    X, Y = np.meshgrid(g, g)
    Z = b.func(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=80,
                                  gridspec_kw={"width_ratios": [1.2, 1]})
    ax.contourf(X, Y, np.log1p(Z), levels=40, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    scat = ax.scatter([], [], s=28, c="white", edgecolors="black", linewidths=0.7)
    star = ax.scatter([], [], s=260, marker="*", c="#facc15", edgecolors="black", zorder=5)
    line, = ax2.semilogy([], [], color="#2563eb", lw=2)
    ax2.set_xlim(0, iters)
    ax2.set_ylim(1e-9, max(pso.history) * 3)
    ax2.set_title("Best fitness (log)")
    ax2.set_xlabel("iteration")
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    def update(frame):
        if frame > 0:
            pso.step()
        scat.set_offsets(pso.positions)
        star.set_offsets([pso.gbest_x])
        ax.set_title(f"PSO on {name} — iteration {pso.iteration}")
        line.set_data(range(len(pso.history)), np.maximum(pso.history, 1e-12))
        return scat, star, line

    anim = FuncAnimation(fig, update, frames=iters + 1)
    anim.save(os.path.join(OUT, path), writer=PillowWriter(fps=10))
    plt.close(fig)
    print("saved", path)


def neural_gif(iters=120, path="demo_pso_nn.gif"):
    mlp = MLP(2)
    Xtr, ytr, _, _ = make_student_dataset(200, noise=4, seed=7)
    pso = PSO(mlp.fitness_function(normalize(Xtr), ytr), mlp.n_params, (-10, 10),
              n_particles=30, max_iter=iters, seed=2)
    g = np.linspace(0, 100, 80)
    GX, GY = np.meshgrid(g, g)
    grid = normalize(np.column_stack([GX.ravel(), GY.ravel()]))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=80,
                                  gridspec_kw={"width_ratios": [1.1, 1]})

    def update(frame):
        for _ in range(0 if frame == 0 else 2):
            if not pso.done:
                pso.step()
        ax.clear()
        prob = mlp.forward(grid, pso.gbest_x).reshape(GX.shape)
        ax.imshow(prob, origin="lower", extent=(0, 100, 0, 100), cmap="RdYlGn",
                  vmin=0, vmax=1, alpha=0.55, aspect="auto")
        ax.contour(GX, GY, prob, levels=[0.5], colors=["#2563eb"], linewidths=2)
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=12, c="#15803d")
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=12, c="#b91c1c")
        acc = mlp.accuracy(normalize(Xtr), ytr, pso.gbest_x)
        ax.set_title(f"Neural net trained by PSO — iter {pso.iteration}, acc {acc:.0%}")
        ax.set_xlabel("exam score")
        ax.set_ylabel("project score")
        ax2.clear()
        ax2.semilogy(pso.history, color="#2563eb", lw=2)
        ax2.set_xlim(0, iters)
        ax2.set_title("Training MSE (gbest)")
        ax2.set_xlabel("iteration")
        ax2.grid(alpha=0.3)
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=iters // 2 + 1)
    anim.save(os.path.join(OUT, path), writer=PillowWriter(fps=10))
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    swarm_gif()
    neural_gif()
