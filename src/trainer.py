import numpy as np
from src.activation.activation import Array
from src.cost.cost import CostFunction
from src.optimizer.optimizer import Optimizer
from src.metric.metric import Metric
from src.config import ExperimentConfig
from src.network.model import Model


class Trainer:
    """Loop de entrenamiento genérico. No sabe nada de ejercicios ni de CSVs."""

    def __init__(self, cost_fn: CostFunction, optimizer: Optimizer, metrics: list[Metric], cfg: ExperimentConfig, regularization=False) -> None:
        self.cost_fn = cost_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.cfg = cfg
        self.ridge_alpha = 0.0001
        self.regularization = regularization

    def fit(self, model: Model, X_train: Array, zeta_train: Array, X_val: Array| None, zeta_val: Array|None) -> dict:
        """Entrena el modelo y devuelve el historial de errores por época."""
        if self.cfg.training_mode == "online":
            train_fn = self._train_epoch_online
        elif self.cfg.training_mode == "minibatch":
            train_fn = self._train_epoch_minibatch
        elif self.cfg.training_mode == "batch":
            train_fn = self._train_epoch_batch
        else:
            raise ValueError(f"training_mode desconocido: {self.cfg.training_mode!r}")

        train_errors, val_errors = [], []
        best_val_error = float('inf')
        best_weights = None
        patience = 50
        strikes = 0

        for epoch in range(self.cfg.epochs): # ← "for a fixed number of epochs" (Clase 11)
            train_errors.append(train_fn(model, X_train, zeta_train))
            # TODO: esto deberia ser opcional!!
            val_error = self._evaluate_loss(model, X_val, zeta_val)
            val_errors.append(val_error)

            if val_error < best_val_error:
                best_val_error = val_error
                best_weights = model.get_weights()  # save copy
                strikes = 0
            else:
                strikes += 1
                if strikes >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if train_errors[-1] < self.cfg.epsilon:
                break

        if best_weights is not None:
            model.set_weights(best_weights)

        return {"train_error": train_errors, "val_error": val_errors, "epochs": epoch + 1} # ← "if E < ε: break"  (Clase 11)

    def evaluate(self, model: Model, X: Array, zeta: Array) -> dict[str, float]:
        """Evaluación final — solo mide métricas, no toca los pesos."""
        predictions = np.array([model.forward(xi) for xi in X])
        return {metric.name(): metric.compute(zeta, predictions) for metric in self.metrics}

    def _train_epoch_online(self, model: Model, X: Array, zeta: Array) -> float:
        """Online: update después de cada muestra individual.

        for cada muestra:
            forward  → O    = model.forward(xi)
            backward → grad = cost.gradient(zi, O); model.backward(grad)
            update   → pesos actualizados con el gradiente de ESA muestra
            loss     → E    = cost(zi, O)
        """
        total_loss = 0.0
        for xi, zi in zip(X, zeta):
            # ── PASO 1: FORWARD ──────────────────────────────────────────────
            # Clase 10.1/10.2: O^μ = Θ(Σ xᵢ·wᵢ + w₀)   ("salida obtenida")
            # Clase 11:        O_j = Θ(Σ_k V_k^{M-1} · w_jk^M)  (feed forward pass)
            O    = model.forward(xi) # En este punto en un Perceptron Simple de cuaquier tipo, tenemos Output = θ(h)
            # ── PASO 2a: GRADIENTE EN LA SALIDA ──────────────────────────────
            # Clase 11 (regla de la cadena):  ∂E/∂W = ∂E/∂O · ∂O/∂h · ∂h/∂W
            #
            # `grad` es SOLO el primer factor:  grad = ∂E/∂O
            # Es el "punto de entrada" del backprop — cuánto cambió el error
            # respecto a la SALIDA de la red.  model.backward() se encarga
            # de multiplicar el resto de la cadena hacia atrás capa por capa.
            #
            # Conexión con perceptrón simple (Clase 10.2):
            #   Para MSE,  ∂E/∂O = -(ζ - O),  por eso Δw = η·(ζ-O)·x
            #   Aquí eso mismo queda separado en cost_fn.gradient + model.backward
            grad = self.cost_fn.gradient(zi, O)
            # ── PASO 2b: BACKWARD (backpropagation) ──────────────────────────
            # Clase 11: aplica la regla de la cadena de SALIDA hacia ENTRADA.
            # Para cada capa calcula  ∂E/∂W  usando el δ de la capa siguiente
            # (magnitud δ de Rumelhart, Hinton & Williams 1986).
            #
            #   δ^M     = ∂E/∂O · ∂O/∂h          (capa de salida)
            #   δ^{l}   = (Σ_j δ^{l+1} · w_j) · ∂V^l/∂h^l  (capas ocultas)
            #   ∂E/∂W^l = δ^l · V^{l-1}           (gradiente de los pesos)
            #
            # Perceptrón simple: con una sola capa no hay "retro"-propagación;
            # ∂E/∂W = -(ζ-O)·x  se calcula directo, sin necesidad de δ.
            model.zero_grads()
            model.backward(grad)
            # ── PASO 2c: ACTUALIZACIÓN DE PESOS ──────────────────────────────
            # Clase 10.1/10.2:  w ← w + Δw    con  Δw = η·(ζ-O)·x
            # Clase 11:         w ← w - η · ∂E/∂W   (gradiente descendente)
            #
            # model.get_grads() devuelve los ∂E/∂W acumulados en backward().
            # optimizer.update() aplica la regla de descenso (puede ser SGD, Adam, etc.)
            model.set_weights(self.optimizer.update(model.get_weights(), model.get_grads()))
            # ── PASO 3: ERROR ─────────────────────────────────────────────────
            # Clase 11:  E = f(x^μ_1, ..., x^μ_n);  si E < ε → convergencia
            # Acumulamos E sobre todas las muestras de la época.
            total_loss += self.cost_fn.compute(zi, O)
        return total_loss / len(X)

    def _train_epoch_minibatch(self, model: Model, X: Array, zeta: Array) -> float:
        """Minibatch: acumula gradientes sobre B muestras, update una vez por batch."""
        n = len(X)
        total_loss = 0.0
        indices = np.random.permutation(n)
        for start in range(0, n, self.cfg.batch_size):
            batch_idx = indices[start:start + self.cfg.batch_size]
            batch_size = len(batch_idx)
            model.zero_grads()

            for i in batch_idx:
                xi, zi = X[i], zeta[i]
                O = model.forward(xi)
                model.backward(self.cost_fn.gradient(zi, O))
                total_loss += self.cost_fn.compute(zi, O)

            avg_grads = [(gw / batch_size, gb / batch_size) for gw, gb in model.get_grads()]

            if self.regularization:
                weights = model.get_weights()

                l2_loss = 0.5 * self.ridge_alpha * sum(np.sum(w ** 2) for w, b in weights)
                total_loss += l2_loss * batch_size

                avg_grads = [
                    (gw + self.ridge_alpha * w, gb)
                    for (gw, gb), (w, b) in zip(avg_grads, weights)
                ]

            model.set_weights(self.optimizer.update(model.get_weights(), avg_grads))

        return total_loss / n

    def _train_epoch_batch(self, model: Model, X: Array, zeta: Array) -> float:
        """Batch: acumula gradientes sobre TODOS los datos, update una sola vez por época."""
        n = len(X)
        total_loss = 0.0
        model.zero_grads()
        for xi, zi in zip(X, zeta):
            O = model.forward(xi)
            model.backward(self.cost_fn.gradient(zi, O))

            # We are optionally using the ridge regression regularization
            l2 = 0
            if self.regularization:
                l2 = 0.5 * self.ridge_alpha * sum(np.sum(w ** 2) for (w, b) in model.get_weights())


            total_loss += self.cost_fn.compute(zi, O) + l2
        avg_grads = [(gw / n, gb / n) for gw, gb in model.get_grads()]
        model.set_weights(self.optimizer.update(model.get_weights(), avg_grads))
        return total_loss / n

    def _evaluate_loss(self, model: Model, X: Array, zeta: Array) -> float:
        """Mide la pérdida total sin tocar los pesos."""
        return sum(self.cost_fn.compute(zi, model.forward(xi)) for xi, zi in zip(X, zeta)) / len(X)
