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

    # DEBUG 1 — check data loaded correctly
    print(f"X shape: {X.shape}, zeta shape: {zeta.shape}")
    print(f"X min: {X.min():.3f}, max: {X.max():.3f}")
    print(f"Label distribution: {np.sum(zeta, axis=0).astype(int)}")

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
        activation=LogisticActivation(beta=1.0)
    ))
    model = MultilayerPerceptron(layers)

    # Split
    train_ds, val_ds, test_ds, _ = Dataset(X=X, zeta=zeta).split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
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

    # EVALUATE on digits_test.csv
    df_test = pd.read_csv("data/digits_test.csv")
    X_test  = np.array(df_test["image"].apply(ast.literal_eval).tolist())
    y_test  = df_test["label"].values

    confusion, metrics = evaluate_multiclass(model, X_test, y_test)
    print_report(confusion, metrics)