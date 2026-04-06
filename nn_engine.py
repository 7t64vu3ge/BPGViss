"""Neural network engine with full backpropagation tracking."""
import numpy as np

class Activation:
    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(a):
        return (a > 0).astype(float)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(a):
        return 1 - a ** 2


class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.1, activation='sigmoid', seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.num_layers = len(layer_sizes)
        self.activation_name = activation

        if activation == 'sigmoid':
            self.act_fn = Activation.sigmoid
            self.act_deriv = Activation.sigmoid_deriv
        elif activation == 'relu':
            self.act_fn = Activation.relu
            self.act_deriv = Activation.relu_deriv
        else:
            self.act_fn = Activation.tanh
            self.act_deriv = Activation.tanh_deriv

        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.history = {'loss': [], 'weights': [], 'gradients': [], 'activations': []}

    def forward(self, X):
        self.z_cache = []
        self.a_cache = [X.copy()]
        a = X
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_cache.append(z)
            if i == self.num_layers - 2:
                a = Activation.sigmoid(z)
            else:
                a = self.act_fn(z)
            self.a_cache.append(a)
        return a

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, y_true):
        m = y_true.shape[0]
        self.dw_cache = []
        self.db_cache = []
        self.delta_cache = []

        y_pred = self.a_cache[-1]
        delta = y_pred - y_true

        for i in range(self.num_layers - 2, -1, -1):
            dw = self.a_cache[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)
            self.dw_cache.insert(0, dw)
            self.db_cache.insert(0, db)
            self.delta_cache.insert(0, delta)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.act_deriv(self.a_cache[i])

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.dw_cache[i]
            self.biases[i] -= self.lr * self.db_cache[i]

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y)
        self.backward(y)

        self.history['loss'].append(loss)
        self.history['weights'].append([w.copy() for w in self.weights])
        self.history['gradients'].append([dw.copy() for dw in self.dw_cache])
        self.history['activations'].append([a.copy() for a in self.a_cache])

        self.update_weights()
        return loss

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            self.train_step(X, y)
        return self.history

    def get_gradient_norms(self):
        norms = []
        for grads in self.history['gradients']:
            norms.append([np.linalg.norm(g) for g in grads])
        return norms
