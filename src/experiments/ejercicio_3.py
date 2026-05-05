import ast
import dataclasses

import numpy as np
import pandas as pd

from data_management.splitter import k_fold_split
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
    etas = [0.1]
    epochs_list = [500]
    architectures = [

        [784, 256, 128, 10],
    ]
    optimizers = ["momentum"]
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

    # 2) Build model template with best hyperparams
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
    model_template = MultilayerPerceptron(layers)

    # 3) K-FOLD on full digits.csv with best hyperparams
    print("Running K-Fold...")
    dataset = Dataset(X=X, zeta=zeta)
    avg_acc, fold_accs = _run_kfold_multiclass(cfg_best, dataset, model_template, k=5)
    print(f"\nBest-params k-fold avg_accuracy={avg_acc:.4f}")

    # 4) Optional: train final model on all digits.csv
    final_model = model_template.clone()
    final_trainer = Trainer(
        cost_fn=CategoricalCrossEntropyCost(),
        optimizer=_build_optimizer(cfg_best),
        metrics=[],
        cfg=cfg_best,
    )
    history = final_trainer.fit(
        final_model,
        dataset.X, dataset.zeta,
        X_val=None, zeta_val=None,
    )
    plot_error_curve(history, output_path="output/experiment/ej3/learning_curve.png")

    # 5) Evaluate on digits_test.csv
    df_test = pd.read_csv("data/digits_test.csv")
    X_test  = np.array(df_test["image"].apply(ast.literal_eval).tolist())
    y_test  = df_test["label"].values

    confusion, metrics = evaluate_multiclass(final_model, X_test, y_test)
    print_report(confusion, metrics)

def _run_kfold_multiclass(
    cfg: ExperimentConfig,
    dataset: Dataset,
    model_template: MultilayerPerceptron,
    k: int = 5,
) -> tuple[float, list[float]]:
    splits = k_fold_split(dataset, k=k, seed=cfg.seed)
    fold_accuracies = []

    for i, (train_ds, val_ds) in enumerate(splits):
        fold_model = model_template.clone()

        trainer = Trainer(
            cost_fn=CategoricalCrossEntropyCost(),
            optimizer=_build_optimizer(cfg),
            metrics=[],
            cfg=cfg,
        )
        trainer.fit(
            fold_model,
            X_train=train_ds.X, zeta_train=train_ds.zeta,
            X_val=val_ds.X,     zeta_val=val_ds.zeta,
        )

        # accuracy on val fold
        val_outputs = np.array([fold_model.forward(xi) for xi in val_ds.X])
        val_preds = np.argmax(val_outputs, axis=1)
        val_true = np.argmax(val_ds.zeta, axis=1)
        val_accuracy = float(np.mean(val_preds == val_true))
        fold_accuracies.append(val_accuracy)

        print(f"Fold {i + 1}/{k}: val_accuracy={val_accuracy:.4f}")

    avg_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else float("nan")
    return avg_accuracy, fold_accuracies