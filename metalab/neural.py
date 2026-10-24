"""Training a small neural network with PSO instead of backpropagation.

Network (same as the lecture example): 2 inputs -> H hidden (sigmoid) -> 1 output (sigmoid)

All weights and biases are flattened into ONE vector. That vector is a single
PSO particle, and the training error (MSE) is its fitness:

    particle = [W1 (2*H) | b1 (H) | W2 (H) | b2 (1)]      length = 4*H + 1

With H = 2 that is 9 numbers, i.e. PSO searches a 9-dimensional space.
"""

import numpy as np

DATASETS = {
    "weighted": "Ağırlıklı ort. ≥ 50  (0.4·sınav + 0.6·proje)",
    "threshold": "Barajlı: ortalama ≥ 50 ve her iki not ≥ 40",
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# --------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------- #
def make_student_dataset(n=200, noise=4.0, kind="weighted", test_ratio=0.2, seed=None):
    """Synthetic "exam + project -> pass/fail" data.

    Returns raw scores (0-100) and labels, already split into train / test.
    ``noise`` is the std of random points added to the average before
    labelling, which makes the classes overlap slightly (more realistic).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, (n, 2))
    avg = 0.4 * X[:, 0] + 0.6 * X[:, 1] + rng.normal(0, noise, n)
    if kind == "threshold":
        y = ((avg >= 50) & (X[:, 0] >= 40) & (X[:, 1] >= 40)).astype(float)
    else:
        y = (avg >= 50).astype(float)

    idx = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    test, train = idx[:n_test], idx[n_test:]
    return X[train], y[train], X[test], y[test]


def normalize(X):
    """Scale scores 0-100 -> 0-1.

    Without this, inputs like 85 and 90 push every hidden neuron into
    saturation (sigmoid(44) ~ 1) and the network cannot learn.
    """
    return np.asarray(X, dtype=float) / 100.0


# --------------------------------------------------------------------- #
#  Network
# --------------------------------------------------------------------- #
class MLP:
    """2-H-1 multilayer perceptron whose weights live in a flat vector."""

    def __init__(self, n_hidden=2, n_inputs=2):
        self.n_in = int(n_inputs)
        self.h = int(n_hidden)

    @property
    def n_params(self):
        return self.n_in * self.h + self.h + self.h + 1

    def param_names(self):
        names = [f"w{i + 1}{j + 1}" for i in range(self.n_in) for j in range(self.h)]
        names += [f"b{j + 1}" for j in range(self.h)]
        names += [f"v{j + 1}" for j in range(self.h)]
        names += ["b_çıkış"]
        return names

    def decode(self, p):
        """Flat particle -> (W1, b1, W2, b2)."""
        p = np.asarray(p, dtype=float)
        i = 0
        W1 = p[i:i + self.n_in * self.h].reshape(self.n_in, self.h)
        i += self.n_in * self.h
        b1 = p[i:i + self.h]
        i += self.h
        W2 = p[i:i + self.h]
        i += self.h
        b2 = p[i]
        return W1, b1, W2, b2

    def forward(self, X, p, return_hidden=False):
        W1, b1, W2, b2 = self.decode(p)
        hidden = sigmoid(X @ W1 + b1)          # hidden layer outputs
        out = sigmoid(hidden @ W2 + b2)        # network output in (0, 1)
        return (out, hidden) if return_hidden else out

    def forward_batch(self, X, P):
        """Evaluate MANY particles at once. P: (n_particles, n_params) -> (n_particles, n_samples)."""
        P = np.atleast_2d(P)
        k = self.n_in * self.h
        W1 = P[:, :k].reshape(-1, self.n_in, self.h)
        b1 = P[:, k:k + self.h]
        W2 = P[:, k + self.h:k + 2 * self.h]
        b2 = P[:, -1]
        hidden = sigmoid(np.einsum("si,pih->psh", X, W1) + b1[:, None, :])
        return sigmoid(np.einsum("psh,ph->ps", hidden, W2) + b2[:, None])

    def mse(self, X, y, p):
        return float(np.mean((self.forward(X, p) - y) ** 2))

    def accuracy(self, X, y, p):
        return float(np.mean((self.forward(X, p) >= 0.5) == (y >= 0.5)))

    def fitness_function(self, X, y):
        """Objective for PSO: ``f(P) -> MSE of every particle``."""
        def f(P):
            return np.mean((self.forward_batch(X, P) - y[None, :]) ** 2, axis=1)
        return f


# --------------------------------------------------------------------- #
#  Backpropagation baseline (for comparison)
# --------------------------------------------------------------------- #
class BackpropTrainer:
    """Full-batch gradient descent on the same MLP and the same MSE loss."""

    name = "Geri Yayılım"

    def __init__(self, mlp, X, y, lr=2.0, init_scale=1.0, seed=None):
        self.mlp, self.X, self.y = mlp, X, y
        self.lr = float(lr)
        rng = np.random.default_rng(seed)
        self.p = rng.uniform(-init_scale, init_scale, mlp.n_params)
        self.epoch = 0
        self.history = [mlp.mse(X, y, self.p)]

    def step(self, epochs=1):
        for _ in range(int(epochs)):
            W1, b1, W2, b2 = self.mlp.decode(self.p)
            out, hidden = self.mlp.forward(self.X, self.p, return_hidden=True)
            n = len(self.y)
            # dLoss/dOut for MSE, times sigmoid derivative
            d_out = 2 * (out - self.y) / n * out * (1 - out)
            g_W2 = hidden.T @ d_out
            g_b2 = d_out.sum()
            d_hidden = np.outer(d_out, W2) * hidden * (1 - hidden)
            g_W1 = self.X.T @ d_hidden
            g_b1 = d_hidden.sum(axis=0)
            grad = np.concatenate([g_W1.ravel(), g_b1, g_W2, [g_b2]])
            self.p = self.p - self.lr * grad
            self.epoch += 1
        loss = self.mlp.mse(self.X, self.y, self.p)
        self.history.append(loss)
        return loss
