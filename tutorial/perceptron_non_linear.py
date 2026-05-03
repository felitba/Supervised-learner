import numpy as np

from tutorial.perceptron import Perceptron


def _tanh_derivative(x, beta_value=0.1):
    return beta_value * (1 - np.tanh(beta_value * x) ** 2)


class PerceptronNonLinear(Perceptron):

    #for this example, we will use the TANH(B * X) function as an activation function
    # where B is an arbitrary constant from > 0 to 10
    # the derivative from

    def __init__(self, learning_rate=0.1, epochs=20, epsilon=0.02):
        super().__init__(learning_rate, epochs,epsilon)
        self.beta_value = None

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        linear_output = np.dot(X, self.weights) + self.bias
        beta = 0.1 if self.beta_value is None else self.beta_value
        return np.tanh(beta * linear_output)

    def fit(self, X, y, beta_value=0.1, batch_size=1):
        self.beta_value = beta_value
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        for i in range(self.epochs):
            # print(f"Epoch: {i}")
            # print("Weights:", self.weights)
            # print("Bias:", self.bias)
            # print("")

            predictions = np.zeros_like(y, dtype=float)
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)

            # Initialize batch gradient accumulators
            grad_w = np.zeros_like(self.weights)
            grad_b = 0.0
            batch_count = 0

            for idx in indices:
                xi = X[idx]
                target = y[idx]

                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = np.tanh(self.beta_value * linear_output)
                predictions[idx] = y_pred

                # delta = (t - y) * tanh'(beta*z)
                delta = (target - y_pred) * _tanh_derivative(linear_output, self.beta_value)

                # Accumulate gradients for the batch
                grad_w += delta * xi
                grad_b += delta
                batch_count += 1

                # If batch is full, apply averaged update
                if batch_count == batch_size:
                    self.weights += self.lr * (grad_w / batch_count)
                    self.bias += self.lr * (grad_b / batch_count)

                    # Reset batch accumulators
                    grad_w.fill(0.0)
                    grad_b = 0.0
                    batch_count = 0

            # Apply any remaining partial batch (if n_samples % batch_size != 0)
            if batch_count > 0:
                self.weights += self.lr * (grad_w / batch_count)
                self.bias += self.lr * (grad_b / batch_count)


            err = mse(self,y, predictions)
            self.errors_per_epoch.append(err)
            # print(f"Best Error {err}")
            # print("")

            if err < self.epsilon:
                print(f"Method converged at epoch: {i}")
                break

            # inside fit(), after each epoch
            # plot_adaline_regression(
            #     X, y, self,
            #     f"TANH Regression Epoch {i + 1}",
            #     f"output/adaline_epoch_{i + 1}.png",
            #     xlim=(-6, 6),
            #     ylim=(-1.5,1.5),
            #     centered=True
            # )
        print(f"Finished training after {self.epochs} epochs with final error {self.errors_per_epoch[-1]:.4f}")


def mse(self, zeta, predictions) -> float:
    """E = (1/2N) Σ (ζ - O)²"""
    N = len(zeta) if hasattr(zeta, '__len__') else 1
    return (1 / (2 * N)) * np.sum((zeta - predictions) ** 2)

