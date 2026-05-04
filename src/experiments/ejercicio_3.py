import ast
import numpy as np
import pandas as pd

from analysis.plots import plot_error_curve
from src.data_management.preprocessing import one_hot_encode
from src.cost.categorical_cross_entropy import CategoricalCrossEntropyCost
from src.activation.logistic import LogisticActivation
from src.activation.tanh import TanhActivation
from src.config import ExperimentConfig
from src.data_management.dataset import Dataset
from src.metric.evaluate_mlp import evaluate_multiclass, print_report
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer


def run(cfg: ExperimentConfig) -> None:
    # SETUP
    df_train = pd.read_csv(cfg.data_path)
    X = np.array(df_train["image"].apply(ast.literal_eval).tolist())
    zeta = one_hot_encode(df_train["label"].values, n_classes=10)

    # Architecture from config
    layers = []
    for i in range(len(cfg.architecture) - 2):
        layers.append(NeuronLayer(
            n_inputs=cfg.architecture[i],
            n_neurons=cfg.architecture[i + 1],
            activation=TanhActivation(beta=1.0)
        ))
    layers.append(NeuronLayer(
        n_inputs=cfg.architecture[-2],
        n_neurons=cfg.architecture[-1],
        activation = LogisticActivation(beta=1.0)
    ))
    model = MultilayerPerceptron(layers)

    # Split
    train_ds, val_ds, test_ds, _ = Dataset(X=X, zeta=zeta).split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )

    # Train
    trainer = Trainer(
        cost_fn   = CategoricalCrossEntropyCost(),
        optimizer = GradientDescent(learning_rate=cfg.eta),
        metrics   = [],
        cfg       = cfg,
    )

    history = trainer.fit(
        model,
        train_ds.X, train_ds.zeta,
        val_ds.X,   val_ds.zeta,
    )

    plot_error_curve(history, output_path="output/experiment/ej3/learning_curve.png")

    # Evaluate on digits_test.csv
    df_test = pd.read_csv("data/digits_test.csv")
    X_test  = np.array(df_test["image"].apply(ast.literal_eval).tolist())
    y_test  = df_test["label"].values

    confusion, metrics = evaluate_multiclass(model, X_test, y_test)
    print_report(confusion, metrics)