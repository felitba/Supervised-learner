import numpy as np

from tutorial.perceptron import Perceptron


class PerceptronLinear(Perceptron):

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias


    def fit(self, X, y, batch_size=1):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()
        for i in range(self.epochs):
            # print(f"Epoch: {i}")
            # print("Weights:", self.weights)
            # print("Bias:", self.bias)
            predictions = np.zeros_like(y, dtype=float)
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)
            batch_updates = []
            batch_size_counter = 0
            
            for idx in indices:
                xi = X[idx]
                target = y[idx]

                linear_output = np.dot(xi, self.weights) + self.bias
                # we instead assign y_pred as the identity function
                # (directly equals weighted sum instead of {0,1})
                y_pred = linear_output
                predictions[idx] = y_pred


                update = self.lr * (target - y_pred)
                batch_updates.append(update)
                if batch_size_counter >= batch_size:
                    mean_batch_update = np.mean(batch_updates)
                    self.weights += mean_batch_update * xi
                    self.bias += mean_batch_update
                    batch_size_counter = 0
                    batch_updates = []
                else:
                    batch_size_counter += 1

            err = mse(self,y, predictions)
            self.errors_per_epoch.append(err)

            # print(f"Best Error {err}")
            # print("")

            if err < self.epsilon:
                print(f"Early stopping at epoch {i} with error {err}")
                return

            # import
            # inside fit(), after each epoch
            # plot_adaline_regression(
            #     X, y, self,
            #     f"ADALINE Regression Epoch {i + 1}",
            #     f"output/adaline_epoch_{i + 1}.png",
            #     xlim=(-6, 6),
            #     ylim=(-8, 18),
            #     centered=True
            # )
        print(f"Finished training after {self.epochs} epochs with final error {self.errors_per_epoch[-1]:.4f}")


def mse(self, zeta, predictions) -> float:
    """E = (1/2N) Σ (ζ - O)²"""
    N = len(zeta) if hasattr(zeta, '__len__') else 1
    return (1 / (2 * N)) * np.sum((zeta - predictions) ** 2)