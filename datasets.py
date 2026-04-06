"""Dataset generation utilities."""
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification


def get_dataset(name, n_samples=200, noise=0.15, seed=42):
    if name == 'Moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == 'Circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    elif name == 'XOR':
        np.random.seed(seed)
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += np.random.randn(n_samples, 2) * noise
    elif name == 'Spiral':
        np.random.seed(seed)
        n = n_samples // 2
        t1 = np.linspace(0, 3 * np.pi, n) + np.random.randn(n) * noise
        t2 = np.linspace(0, 3 * np.pi, n) + np.pi + np.random.randn(n) * noise
        X = np.vstack([
            np.column_stack([t1 * np.cos(t1), t1 * np.sin(t1)]),
            np.column_stack([t2 * np.cos(t2), t2 * np.sin(t2)])
        ]) / 3
        y = np.array([0]*n + [1]*n)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=2,
                                   n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, random_state=seed)
    y = y.reshape(-1, 1).astype(float)
    return X, y
