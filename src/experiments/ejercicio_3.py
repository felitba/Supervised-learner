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



def grid_search(cfg: ExperimentConfig, X, zeta):
    etas = [0.0005]
    epochs_list = [75]
    architectures = [
        [784, 256, 128, 10],
    ]
    optimizers = ["adam"]
    activations = ["relu"]

    results = []
    combo_n = 0
    total = len(etas) * len(epochs_list) * len(architectures) * len(optimizers)

    for eta in etas:
        for epochs in epochs_list:
            for arch in architectures:
                for opt in optimizers:
                    for act in activations:
                        combo_n += 1
                        print(f"\n({combo_n}/{total}) eta={eta} epochs={epochs} arch={arch} opt={opt}, act={act}")

                        cfg_variant = dataclasses.replace(
                            cfg, eta=eta, epochs=epochs, architecture=arch, optimizer=opt
                        )

                        hidden_act = _build_activation(act, cfg_variant.beta)
                        output_act = _build_activation("softmax", cfg_variant.beta)

                        layers = []
                        for i in range(len(arch) - 2):
                            layers.append(NeuronLayer(
                                n_inputs=arch[i],
                                n_neurons=arch[i + 1],
                                activation=hidden_act,
                            ))
                        layers.append(NeuronLayer(
                            n_inputs=arch[-2],
                            n_neurons=arch[-1],
                            activation=output_act,
                        ))
                        model = MultilayerPerceptron(layers)

                        train_ds, val_ds, _, _ = Dataset(X=X, zeta=zeta).split(
                            train=cfg_variant.split_train,
                            val=cfg_variant.split_val,
                            test=cfg_variant.split_test,
                            seed=cfg_variant.seed,
                        )

                        trainer = Trainer(
                            cost_fn=CategoricalCrossEntropyCost(),
                            optimizer=_build_optimizer(cfg_variant),  # reuse from ejercicio_2
                            metrics=[],
                            cfg=cfg_variant,
                            regularization=False,
                        )

                        history = trainer.fit(model, train_ds.X, train_ds.zeta, val_ds.X, val_ds.zeta)

                        # evaluate accuracy on val
                        val_outputs = np.array([model.forward(xi) for xi in val_ds.X])
                        val_preds = np.argmax(val_outputs, axis=1)
                        val_true = np.argmax(val_ds.zeta, axis=1)
                        val_accuracy = float(np.mean(val_preds == val_true))

                        results.append(({"eta": eta, "epochs": epochs, "arch": arch, "opt": opt, "act":act}, val_accuracy))
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
    model = MultilayerPerceptron(layers)

    # 3) Split + Train final model
    train_ds, val_ds, test_ds, _ = Dataset(X=X, zeta=zeta).split(
        train=cfg_best.split_train,
        val=cfg_best.split_val,
        test=cfg_best.split_test,
        seed=cfg_best.seed,
    )

    # DEBUG 2 — check model output before any training
    sample_output = model.forward(train_ds.X[0])
    print(f"\nRaw output before training: {np.round(sample_output, 4)}")
    print(f"Argmax before training:     {np.argmax(sample_output)}")
    print(f"True label of sample 0:     {np.argmax(train_ds.zeta[0])}")

    # DEBUG 3 — check gradient on first sample
    O    = model.forward(train_ds.X[0])
    grad = CategoricalCrossEntropyCost().gradient(train_ds.zeta[0], O)
    print(f"\nGradient on first sample:   {np.round(grad, 4)}")
    print(f"Gradient max abs value:     {np.abs(grad).max():.6f}")

    # Train
    from src.experiments.ejercicio_2 import _build_optimizer  # reuse helper

    # ...
    trainer = Trainer(
        cost_fn=CategoricalCrossEntropyCost(),
        optimizer=_build_optimizer(cfg_best),
        metrics=[],
        cfg=cfg_best,
        regularization=False,
    )

    history = trainer.fit(
        model,
        train_ds.X, train_ds.zeta,
        val_ds.X,   val_ds.zeta,
    )

    for i in range(0, len(history['train_error']), 10):
        print(f"Epoch {i:3d}: {history['train_error'][i]:.6f}")

    # DEBUG 4 — check loss movement
    print(f"\nFirst 5 train errors: {[round(e, 6) for e in history['train_error'][:5]]}")
    print(f"Last  5 train errors: {[round(e, 6) for e in history['train_error'][-5:]]}")

    # DEBUG 5 — check model output after training
    sample_output_after = model.forward(train_ds.X[0])
    print(f"\nRaw output after training:  {np.round(sample_output_after, 4)}")
    print(f"Argmax after training:      {np.argmax(sample_output_after)}")
    print(f"True label of sample 0:     {np.argmax(train_ds.zeta[0])}")

    plot_error_curve(history, output_path="output/experiment/ej3/learning_curve.png")

    # 4) Evaluate on digits_test.csv
    df_test = pd.read_csv("data/digits_test.csv")
    X_test  = np.array(df_test["image"].apply(ast.literal_eval).tolist())
    y_test  = df_test["label"].values

    confusion, metrics = evaluate_multiclass(model, X_test, y_test)
    print_report(confusion, metrics)