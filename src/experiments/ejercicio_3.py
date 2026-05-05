import ast
import dataclasses

import numpy as np
import pandas as pd

from src.optimizer.gradient_descent import GradientDescent
from src.activation.identity import IdentityActivation
from src.activation.relu import ReLUActivation
from src.activation.logistic import LogisticActivation
from src.activation.soft_max import SoftMaxActivation
from analysis.plots import plot_error_curve
from src.optimizer.adam import AdamOptimizer
from src.optimizer.momentum import MomentumOptimizer

from src.data_management.preprocessing import one_hot_encode, standardize, normalize
from src.cost.categorical_cross_entropy import CategoricalCrossEntropyCost
from src.activation.tanh import TanhActivation
from src.config import ExperimentConfig
from src.data_management.dataset import Dataset
from src.metric.evaluate_mlp import evaluate_multiclass, print_report
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.trainer import Trainer


def _build_optimizer(cfg: ExperimentConfig):
    if cfg.optimizer == "adam":
        return AdamOptimizer(learning_rate=cfg.eta, beta1=cfg.adam_beta1, beta2=cfg.adam_beta2)
    if cfg.optimizer == "momentum":
        return MomentumOptimizer(learning_rate=cfg.eta, beta=cfg.momentum_beta)
    return GradientDescent(learning_rate=cfg.eta)


def _build_weight_initializer(act_name: str, n_inputs: int, n_neurons: int):
    """Returns a lambda that generates the perfectly scaled weight matrix."""
    rng = np.random.default_rng()
    act_name = act_name.lower()

    if act_name == "relu":
        # He Initialization (variance = 2.0 / n_inputs)
        return lambda: rng.standard_normal((n_inputs, n_neurons)) * np.sqrt(2.0 / n_inputs)
    else:
        # Xavier Initialization (variance = 1.0 / n_inputs) for Tanh, Softmax, etc.
        return lambda: rng.standard_normal((n_inputs, n_neurons)) * np.sqrt(1.0 / n_inputs)


def _build_activation(name: str, beta: float):
    name = name.lower()
    if name == "tanh":
        return TanhActivation(beta=beta)
    if name == "relu":
        return ReLUActivation()
    if name == "logistic":
        return LogisticActivation(beta=beta)
    if name == "softmax":
        return SoftMaxActivation()
    if name == "identity":
        return IdentityActivation()
    raise ValueError(f"Unknown activation: {name}")


def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(img)
    x_from = max(0, dx)
    x_to = min(28, 28 + dx)
    y_from = max(0, dy)
    y_to = min(28, 28 + dy)
    shifted[x_from:x_to, y_from:y_to] = img[x_from - dx:x_to - dx, y_from - dy:y_to - dy]
    return shifted


def augment_shift(X: np.ndarray, max_shift: int = 2) -> np.ndarray:
    X_img = X.reshape(-1, 28, 28)
    out = []
    for img in X_img:
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        out.append(_shift_image(img, dx, dy))
    return np.array(out).reshape(-1, 784)


def augment_noise(X: np.ndarray, std: float = 0.05, clip: tuple[float, float] | None = (0.0, 1.0)) -> np.ndarray:
    noise = np.random.normal(0.0, std, size=X.shape)
    X_aug = X + noise
    if clip is not None:
        X_aug = np.clip(X_aug, clip[0], clip[1])
    return X_aug


def grid_search(cfg: ExperimentConfig, X, zeta):
    etas = [0.0005]
    epochs_list = [500]
    architectures = [
        [784, 256, 64, 10],
    ]
    optimizers = ["adam"]
    activations = ["relu"]
    patience = [30]

    results = []
    combo_n = 0
    total = len(etas) * len(epochs_list) * len(architectures) * len(optimizers)

    for patience in patience:
        for eta in etas:
            for epochs in epochs_list:
                for arch in architectures:
                    for opt in optimizers:
                        for act in activations:
                            combo_n += 1
                            print(f"\n({combo_n}/{total}) eta={eta} epochs={epochs} arch={arch} opt={opt}, act={act}, patience={patience}")

                            cfg_variant = dataclasses.replace(
                                cfg, eta=eta, epochs=epochs, architecture=arch, optimizer=opt
                            )

                            hidden_act = _build_activation(act, cfg_variant.beta)
                            output_act = _build_activation("softmax", cfg_variant.beta)

                            layers = []
                            for i in range(len(arch) - 2):
                                n_in = arch[i]
                                n_out = arch[i + 1]
                                layers.append(NeuronLayer(
                                    n_inputs=n_in,
                                    n_neurons=n_out,
                                    activation=hidden_act,
                                    weight_initializator=_build_weight_initializer(act, n_in, n_out)
                                ))

                            n_in_out = arch[-2]
                            n_out_out = arch[-1]
                            layers.append(NeuronLayer(
                                n_inputs=n_in_out,
                                n_neurons=n_out_out,
                                activation=output_act,
                                weight_initializator=_build_weight_initializer("softmax", n_in_out, n_out_out)
                            ))
                            model = MultilayerPerceptron(layers)

                            # split, then augment only train
                            train_ds, val_ds, _, _ = Dataset(X=X, zeta=zeta).split(
                                train=cfg_variant.split_train,
                                val=cfg_variant.split_val,
                                test=cfg_variant.split_test,
                                seed=cfg_variant.seed,
                            )

                            X_aug = augment_noise(train_ds.X, std=0.05)
                            train_ds = Dataset(
                                X=np.concatenate([train_ds.X, X_aug], axis=0),
                                zeta=np.concatenate([train_ds.zeta, train_ds.zeta], axis=0)
                            )

                            trainer = Trainer(
                                cost_fn=CategoricalCrossEntropyCost(),
                                optimizer=_build_optimizer(cfg_variant),
                                metrics=[],
                                cfg=cfg_variant,
                                regularization=True,
                                patience=patience,
                            )

                            history = trainer.fit(model, train_ds.X, train_ds.zeta, val_ds.X, val_ds.zeta)

                            # evaluate accuracy on val
                            val_outputs = np.array([model.forward(xi) for xi in val_ds.X])
                            val_preds = np.argmax(val_outputs, axis=1)
                            val_true = np.argmax(val_ds.zeta, axis=1)
                            val_accuracy = float(np.mean(val_preds == val_true))

                            plot_error_curve(history, output_path=f"output/experiment/ej3/learning_curve_{arch}_{opt}_{act}_{eta}_{epochs}_{patience}.png")

                            results.append(({"eta": eta, "epochs": epochs, "arch": arch, "opt": opt, "act": act, "patience": patience}, val_accuracy))
                            print(f"→ val_accuracy={val_accuracy:.4f}")

    best_params, best_acc = max(results, key=lambda x: x[1])
    print("\nBest combo:", best_params, "val_accuracy=", best_acc)
    return best_params, best_acc


def run(cfg: ExperimentConfig) -> None:
    # SETUP
    df_train = pd.read_csv(cfg.data_path)
    X = np.array(df_train["image"].apply(ast.literal_eval).tolist())

    if cfg.preprocessing == "standardize":
        X = standardize(X)
    elif cfg.preprocessing == "normalize":
        X = normalize(X)

    zeta = one_hot_encode(df_train["label"].values, n_classes=10)

    # 1) GRID SEARCH to find best hyperparams
    best_params, _ = grid_search(cfg, X, zeta)

    # 2) Build model with best hyperparams
    cfg_best = dataclasses.replace(
        cfg,
        eta=best_params["eta"],
        epochs=best_params["epochs"],
        architecture=best_params["arch"],
        optimizer=best_params["opt"],
        activation=best_params["act"],
    )

    hidden_act = _build_activation(best_params["act"], cfg_best.beta)
    output_act = _build_activation("softmax", cfg_best.beta)

    layers = []
    for i in range(len(cfg_best.architecture) - 2):
        layers.append(NeuronLayer(
            n_inputs=cfg_best.architecture[i],
            n_neurons=cfg_best.architecture[i + 1],
            activation=hidden_act,
        ))
    layers.append(NeuronLayer(
        n_inputs=cfg_best.architecture[-2],
        n_neurons=cfg_best.architecture[-1],
        activation=output_act,
    ))
    model_template = MultilayerPerceptron(layers)

    # 3) Split
    train_ds, val_ds, _, _ = Dataset(X=X, zeta=zeta).split(
        train=cfg_best.split_train,
        val=cfg_best.split_val,
        test=cfg_best.split_test,
        seed=cfg_best.seed,
    )

    # augment
    X_aug_noise = augment_noise(train_ds.X, std=0.05)
    X_aug_shift = augment_shift(train_ds.X, max_shift=2)
    train_ds = Dataset(
        X=np.concatenate([train_ds.X, X_aug_noise, X_aug_shift], axis=0),
        zeta=np.concatenate([train_ds.zeta, train_ds.zeta, train_ds.zeta], axis=0)
    )
    print(f"Train size after augmentation: {len(train_ds.X)}")

    # 4) Train final model
    final_model = model_template.clone()
    final_trainer = Trainer(
        cost_fn=CategoricalCrossEntropyCost(),
        optimizer=_build_optimizer(cfg_best),
        metrics=[],
        cfg=cfg_best,
        patience=30, # same as grid
        regularization=True,
    )
    history = final_trainer.fit(
        final_model,
        train_ds.X, train_ds.zeta,
        val_ds.X,   val_ds.zeta,
    )
    plot_error_curve(history, output_path="output/experiment/ej3/last_learning_curve.png")

    for i in range(0, len(history['train_error']), 10):
        print(f"Epoch {i:3d}: {history['train_error'][i]:.6f}")

    print(f"\nFirst 5 train errors: {[round(e, 6) for e in history['train_error'][:5]]}")
    print(f"Last  5 train errors: {[round(e, 6) for e in history['train_error'][-5:]]}")

    sample_output_after = final_model.forward(train_ds.X[0])
    print(f"\nRaw output after training:  {np.round(sample_output_after, 4)}")
    print(f"Argmax after training:      {np.argmax(sample_output_after)}")
    print(f"True label of sample 0:     {np.argmax(train_ds.zeta[0])}")

    # 5) Evaluate on digits_test.csv
    df_test = pd.read_csv("data/digits_test.csv")
    X_test  = np.array(df_test["image"].apply(ast.literal_eval).tolist())
    y_test  = df_test["label"].values

    if cfg.preprocessing == "standardize":
        X_test = standardize(X_test)
    elif cfg.preprocessing == "normalize":
        X_test = normalize(X_test)

    confusion, metrics = evaluate_multiclass(final_model, X_test, y_test)
    print_report(confusion, metrics)