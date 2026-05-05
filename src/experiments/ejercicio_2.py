import dataclasses
import os
import ast
import numpy as np
import pandas as pd

from src.data_management.preprocessing import one_hot_encode
from src.data_management.splitter import k_fold_split
from src.data_management.dataset import Dataset
from src.metric.classify_data_mlp import classify_data_mlp
from src.metric.classify_data import classify_data
from src.metric.f1 import F1Metric
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.optimizer.adam import AdamOptimizer
from src.optimizer.momentum import MomentumOptimizer
from src.activation.tanh import TanhActivation
from src.activation.logistic import LogisticActivation
from src.activation.relu import ReLUActivation
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.config import ExperimentConfig
from src.trainer import Trainer
from analysis.plots import plot_error_curve, plot_confusion_matrix_multiclass


def _build_optimizer(cfg: ExperimentConfig):
    if cfg.optimizer == "adam":
        return AdamOptimizer(learning_rate=cfg.eta, beta1=cfg.adam_beta1, beta2=cfg.adam_beta2)
    if cfg.optimizer == "momentum":
        return MomentumOptimizer(learning_rate=cfg.eta, beta=cfg.momentum_beta)
    return GradientDescent(learning_rate=cfg.eta)


def _build_activation(cfg: ExperimentConfig):
    if cfg.activation == "logistic":
        return LogisticActivation(beta=cfg.beta)
    if cfg.activation == "relu":
        return ReLUActivation()
    return TanhActivation(beta=cfg.beta)


def _build_model(cfg: ExperimentConfig) -> MultilayerPerceptron:
    """Construye el MLP dinámicamente a partir de cfg.architecture.

    cfg.architecture es una lista completa incluyendo la entrada, p.ej. [784, 64, 32, 10].
    Cada par consecutivo define una capa: n_inputs=architecture[i], n_neurons=architecture[i+1].
    """
    layers = []
    for i in range(len(cfg.architecture) - 1):
        layers.append(NeuronLayer(
            n_inputs=cfg.architecture[i],
            n_neurons=cfg.architecture[i + 1],
            activation=_build_activation(cfg),
        ))
    return MultilayerPerceptron(layers)


def _compute_accuracy(
    model: MultilayerPerceptron,
    X: np.ndarray,
    int_labels: np.ndarray,
) -> float:
    """Fracción de predicciones correctas. int_labels son etiquetas enteras (no one-hot)."""
    # Dato no menor, usamos accuracy y no algo como F1 aca ya que en este caso no nos interesan los falsos positivos
    # Ya que no son "Representativos" de la eficiencia de mi modelo, ponele. Nos interesan mas simplemente el hit o miss.
    # En este set de datos, a diferencia del anterior no tenemos una categoria abrumantemente dominante como "no fraude"
    # Que eran la GRANDISIMA mayoria. Aca no nos importa el falso positivo, etc.
    # Ojo igual, esto no nos dice todas las cosas que queremos saber, tambien usamos la matriz de confusion para ver exactamente
    # DONDE se confunde la red!
    #Espero que sirva para entender por que usamos esto :)
    outputs     = np.array([model.forward(xi) for xi in X])
    predictions = np.argmax(outputs, axis=1)
    return float(np.mean(predictions == int_labels))


# TP3 K-Fold — digits.csv used for param+hyperparam tuning
def _run_kfold(
    cfg: ExperimentConfig,
    dataset: Dataset,
    int_labels: np.ndarray,
    model_template: MultilayerPerceptron,
    k: int = 5,
    run_label: str = "",
    output_dir: str = "output/experiment/ej2/kfold/default",
) -> tuple[float, list[float]]:
    """Ejecuta k-fold cross-validation y devuelve (avg_accuracy, per_fold_accuracies).

    dataset.zeta — etiquetas one-hot (para la función de costo MSE).
    int_labels   — etiquetas enteras originales (para accuracy).
    """
    # Llamamos k_fold_split dos veces con el mismo seed: una sobre el dataset principal
    # (one-hot zeta) y otra sobre un dataset auxiliar que lleva int_labels como zeta.
    # El mismo seed produce la misma permutación, garantizando que los folds estén alineados.
    splits       = k_fold_split(dataset, k=k, seed=cfg.seed)
    label_splits = k_fold_split(
        Dataset(int_labels.reshape(-1, 1), int_labels), k=k, seed=cfg.seed
    )

    fold_accuracies = []

    for i, ((train_ds, val_ds), (_, val_labels_ds)) in enumerate(zip(splits, label_splits)):
        val_int_labels = val_labels_ds.zeta.astype(int)

        # Pesos frescos cada fold para evitar contaminación entre folds
        fold_model = model_template.clone()

        trainer = Trainer(
            cost_fn=MSECost(),
            optimizer=_build_optimizer(cfg),
            metrics=[],
            cfg=cfg,
        )
        history = trainer.fit(
            fold_model,
            X_train=train_ds.X, zeta_train=train_ds.zeta,
            X_val=val_ds.X,     zeta_val=val_ds.zeta,
        )

        #TODO: evaluar
        val_accuracy = _compute_accuracy(fold_model, val_ds.X, val_int_labels)
        fold_accuracies.append(val_accuracy)

        fold_hparams = (
            f"eta={cfg.eta} | epochs={cfg.epochs} | "
            f"arch={cfg.architecture} | fold={i + 1}/{k}"
        )
        os.makedirs(output_dir, exist_ok=True)
        plot_error_curve(
            history,
            output_path=f"{output_dir}/fold_{i}_error.png",
            hparams_str=fold_hparams,
        )

        prefix = f"  [{run_label}] " if run_label else "  "
        print(
            f"{prefix}Fold {i + 1}/{k}: "
            f"train_error={history['train_error'][-1]:.4f}  "
            f"val_accuracy={val_accuracy:.4f}"
        )

    avg_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else float("nan")
    return avg_accuracy, fold_accuracies


def _generalization_study(
    cfg: ExperimentConfig,
    dataset: Dataset,
    int_labels: np.ndarray,
    model_template: MultilayerPerceptron,
) -> None:
    """Ejercicio 2: estudio de generalización con digits.csv y evaluación final en digits_test.csv."""

    # Paso 1: digits.csv entero es para tuning — NO separamos test set aquí.
    # digits_test.csv es el held-out final, se toca SOLO en el paso 4.

    # Paso 2: Grid search — tasa de aprendizaje, arquitectura, optimizador (requisito mínimo TP3).
    etas          = [0.001, 0.1]
    # Arquitecturas: [entrada, oculta(s), salida] — entrada=784 píxeles, salida=10 clases.
    architectures = [
        [784, 32, 10],
        [784, 128, 64, 10],
    ]
    optimizers    = ["gradient_descent", "adam"]

    total_combos = len(etas) * len(architectures) * len(optimizers)
    print(f"\n{'=' * 60}")
    print(f"  FASE 2 — Grid Search + K-Fold ({total_combos} combinaciones)")
    print(f"{'=' * 60}")

    results = []
    combo_n = 0
    for eta in etas:
        for arch in architectures:
            for opt in optimizers:
                combo_n += 1
                print(f"\n  ({combo_n}/{total_combos}) eta={eta}  arch={arch}  optimizer={opt}")
                cfg_variant   = dataclasses.replace(cfg, eta=eta, architecture=arch, optimizer=opt)
                model_variant = _build_model(cfg_variant)
                combo_dir = (
                    f"output/experiment/ej2/kfold/"
                    f"combo_{combo_n:02d}_eta{eta}_{opt}_{'x'.join(str(n) for n in arch)}"
                )
                avg_accuracy, _ = _run_kfold(
                    cfg_variant, dataset, int_labels, model_variant,
                    run_label=f"{combo_n}/{total_combos}",
                    output_dir=combo_dir,
                )
                results.append(({"eta": eta, "architecture": arch, "optimizer": opt}, avg_accuracy))
                print(
                    f"  → Grid [eta={eta} arch={arch} optimizer={opt}] "
                    f"→ avg_accuracy={avg_accuracy:.4f}"
                )

    # Seleccionamos por MAYOR accuracy (no menor error)
    best_params, best_accuracy = max(results, key=lambda x: x[1])
    print(f"\n{'─' * 60}")
    print(
        f"  Mejor combinación: eta={best_params['eta']}  "
        f"arch={best_params['architecture']}  optimizer={best_params['optimizer']}"
    )
    print(f"  avg_accuracy={best_accuracy:.4f}")
    print(f"{'─' * 60}")

    # Paso 3: Entrenar modelo final en TODO digits.csv con los mejores parámetros
    cfg_best = dataclasses.replace(
        cfg,
        eta=best_params["eta"],
        architecture=best_params["architecture"],
        optimizer=best_params["optimizer"],
    )

    final_model   = _build_model(cfg_best)
    final_trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=_build_optimizer(cfg_best),
        metrics=[],
        cfg=cfg_best,
    )
    history_final = final_trainer.fit(
        final_model, dataset.X, dataset.zeta, X_val=None, zeta_val=None
    )

    os.makedirs("output/experiment/ej2/final", exist_ok=True)
    final_hparams = (
        f"eta={best_params['eta']} | arch={best_params['architecture']} | optimizer={best_params['optimizer']}"
    )
    plot_error_curve(
        history_final,
        output_path="output/experiment/ej2/final/final_training_error.png",
        hparams_str=final_hparams,
    )

    # Paso 4: Evaluación final en digits_test.csv — SOLO AQUÍ se toca el test set
    df_test         = pd.read_csv("data/digits_test.csv")
    X_test          = np.array(df_test["image"].apply(ast.literal_eval).tolist())
    int_labels_test = df_test["label"].values.astype(int)

    test_accuracy = _compute_accuracy(final_model, X_test, int_labels_test)
    print(f"\n  Accuracy en test: {test_accuracy * 100:.2f}%")

    # GENERALIZATION
    test_outputs = np.array([final_model.forward(xi) for xi in X_test])
    confusion    = classify_data_mlp(int_labels_test, test_outputs)

    print("\nConfusion Matrix")
    print("Rows = True class")
    print("Cols = Predicted class\n")
    print("     " + " ".join(f"{i:5d}" for i in range(10)))
    for i, row in enumerate(confusion):
        print(f"{i:3d} | " + " ".join(f"{int(v):5d}" for v in row))

    # TODO: Ejercicio 2a: reportar esta accuracy y la matriz de confusión en el informe
    plot_confusion_matrix_multiclass(
        confusion,
        output_path="output/experiment/ej2/final/confusion_matrix.png",
        hparams_str=final_hparams,
    )


def run(cfg: ExperimentConfig) -> None:
    """Ejercicio 2 — Clasificación de dígitos manuscritos (0-9).

    digits.csv: usado en su totalidad para búsqueda de hiperparámetros (grid search + k-fold).
    digits_test.csv: held-out final, se toca únicamente en la evaluación final (Paso 4).
    """
    # SET UP
    df = pd.read_csv(cfg.data_path)
    X  = np.array(df["image"].apply(ast.literal_eval).tolist())

    # Etiquetas enteras — se guardan ANTES de one-hot para poder calcular accuracy
    int_labels = df["label"].values

    # One-hot encoding para la función de costo (MSE sobre 10 salidas)
    zeta = one_hot_encode(int_labels, n_classes=10)

    dataset = Dataset(X=X, zeta=zeta)

    # Arquitectura dinámica desde cfg.architecture (por defecto en config: [784, 64, 32, 10])
    model_template = _build_model(cfg)

    _generalization_study(cfg, dataset, int_labels, model_template)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY — implementación original, conservada para referencia
# ─────────────────────────────────────────────────────────────────────────────

def _run_legacy(cfg: ExperimentConfig) -> None:
    """DEPRECATED — original implementation, kept for reference.
    See run() for the current implementation."""

    #SET UP
    df = pd.read_csv(cfg.data_path)
    X = np.array(df["image"].apply(ast.literal_eval).tolist())
    zeta = _one_hot_legacy(df["label"].values)
    dataset = Dataset(X=X, zeta=zeta)

    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=cfg.architecture[0], n_neurons=cfg.architecture[1], activation=TanhActivation(beta=1.0)),
        NeuronLayer(n_inputs=cfg.architecture[1], n_neurons=cfg.architecture[2], activation=TanhActivation(beta=1.0)),
    ])

    trainer_mlp = Trainer(
        cost_fn=MSECost(), optimizer=GradientDescent(learning_rate=cfg.eta), metrics=[], cfg=cfg,
    )

    train_dataset, val_dataset, test_dataset, [train_index, val_index, test_index] = dataset.split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )

    # LEARN
    # TODO EVALUATION

    history = trainer_mlp.fit(
        model, train_dataset.X, train_dataset.zeta, val_dataset.X, val_dataset.zeta
    )

    print(f"[DEBUG ej2] Training finished.")
    print(f"Error final: {history['train_error'][-1]:.4f}")


    # GENERALIZATION

    df = pd.read_csv("data/digits_test.csv")
    X = np.array(df["image"].apply(ast.literal_eval).tolist())
    zeta = df["label"].values
    dataset_2 = Dataset(X=X, zeta=zeta)

    test_output_dataset_2 = model.forward(dataset_2.X)
    confusion = classify_data_mlp(
        dataset_2.zeta, test_output_dataset_2)

    print("Confusion Matrix")
    print("Rows = True class")
    print("Cols = Predicted class\n")

    print("     " + " ".join(f"{i:5d}" for i in range(10)))

    for i, row in enumerate(confusion):
        print(f"{i:3d} | " + " ".join(f"{int(v):5d}" for v in row))


def _one_hot_legacy(labels, n_classes=10):
    """DEPRECATED — original implementation, kept for reference.
    See one_hot_encode from src.data_management.preprocessing for the current implementation."""
    y = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y
