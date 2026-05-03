import dataclasses
import numpy as np

from src.data_management.preprocessing import fit_normalize
from src.data_management.loader import load_csv
from src.data_management.splitter import k_fold_split
from src.data_management.dataset import Dataset
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.identity import IdentityActivation
from src.activation.tanh import TanhActivation
from src.activation.step import StepActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_regression, plot_error_curve, plot_learning_comparison, plot_threshold_sweep, plot_confusion_matrix
from src.metric.classify_data import classify_data
from src.metric.f1 import F1Metric


def _learning_study(cfg: ExperimentConfig, dataset: Dataset) -> tuple[dict, dict]:
    """Ejercicio 1a/1b: Compara capacidad de aprendizaje lineal vs no lineal.

    Entrena sobre el dataset COMPLETO normalizado sin split — esto es un
    estudio de capacidad, no de generalización.
    """
    # Normalizar dataset completo — en un capacity study esto es aceptable
    norm_X,    = fit_normalize(dataset.X)
    norm_zeta, = fit_normalize(dataset.zeta)

    n_inputs = dataset.X.shape[1]

    # Config fija para la fase de comparación de capacidad (independiente de cfg)
    EPOCHS = 200
    ETA = 0.01
    MODE = "online"
    study_cfg = dataclasses.replace(cfg, epochs=EPOCHS, training_mode=MODE)

    # --- Perceptrón lineal (ADALINE) ---
    # X_val=None es intencional — esta fase no usa validación
    model_linear = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=IdentityActivation()),
    ])
    trainer_linear = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=ETA),
        metrics=[],
        cfg=study_cfg,
    )
    history_linear = trainer_linear.fit(
        model_linear, norm_X, norm_zeta, X_val=None, zeta_val=None
    )

    layer = model_linear.layers[0]
    print("\n" + "=" * 55)
    print("  FASE 1 — Capacidad de aprendizaje (dataset completo)")
    print("=" * 55)
    print(f"  Modelo:      ADALINE Lineal")
    print(f"  Épocas:      {history_linear['epochs']}")
    #TODO: fix this. son varios pesos, y no layer.weights[0, 0].
    print(f"  Peso final:  w={layer.weights[0, 0]:.4f}   (esperado ≈ 2.0)")
    print(f"  Bias final:  w₀={layer.bias[0]:.4f}   (esperado ≈ 5.0)")
    # TODO> para esta etapa, no usamos val. Como esta funcion de grafico depende de val, la comento por ahora
    print(f"  Error final: {history_linear['train_error'][-1]:.4f}")

    # --- Perceptrón no lineal (Tanh) ---
    # X_val=None es intencional — esta fase no usa validación
    model_nonlinear = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=TanhActivation(beta=cfg.beta)),
    ])
    trainer_nonlinear = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=ETA),
        metrics=[],
        cfg=study_cfg,
    )
    history_nonlinear = trainer_nonlinear.fit(
        model_nonlinear, norm_X, norm_zeta, X_val=None, zeta_val=None
    )
    print(f"  Modelo:      Perceptrón No Lineal (Tanh, β={cfg.beta})")
    print(f"  Error final: {history_nonlinear['train_error'][-1]:.4f}")

    # TODO Ejercicio 1a/1b: analyze underfitting and saturation from these curves to select the best perceptron for Part 2
    plot_learning_comparison(
        history_linear, history_nonlinear,
        output_path="output/experiment/ej1/learning_comparison.png",
    )
    print("  Gráficos guardados en output/experiment/ej1/")

    return history_linear, history_nonlinear


# TP3 K-Fold Cross Validation — see slides on model selection
def _run_kfold(cfg: ExperimentConfig, dataset: Dataset, model_template: MultilayerPerceptron, run_label: str = "") -> float:
    """Runs k-fold cross validation (k=5) and returns avg validation error."""
    # TODO: make k configurable via cfg (add k_folds field to ExperimentConfig)
    k = 5
    splits = k_fold_split(dataset, k=k, seed=cfg.seed)
    # TODO: extend k_fold_split to return indices and remove this coupling.
    val_errors_final = []

    for i, (train_ds, val_ds) in enumerate(splits):
        # Compute normalization params ONLY from train_ds — never from val or test
        norm_train_X,    norm_val_X    = fit_normalize(train_ds.X,    val_ds.X)
        norm_train_zeta, norm_val_zeta = fit_normalize(train_ds.zeta, val_ds.zeta)

        # Fresh model weights each fold to avoid weight contamination from previous folds
        fold_model = model_template.clone()

        #TODO: entrenar
        trainer = Trainer(
            cost_fn=MSECost(),
            optimizer=GradientDescent(learning_rate=cfg.eta),
            metrics=[],
            cfg=cfg,
        )
        history = trainer.fit(
            fold_model,
            X_train=norm_train_X, zeta_train=norm_train_zeta,
            X_val=norm_val_X, zeta_val=norm_val_zeta,
        )

        #TODO: evaluar
        plot_error_curve(history, output_path=f"output/experiment/ej1/fold_{i}_error.png")
        if history["val_error"]:
            prefix = f"  [{run_label}] " if run_label else "  "
            print(f"{prefix}Fold {i+1}/{k}: train={history['train_error'][-1]:.4f}  val={history['val_error'][-1]:.4f}")
            val_errors_final.append(history["val_error"][-1])

    avg_val_error = sum(val_errors_final) / len(val_errors_final) if val_errors_final else float("nan")
    return avg_val_error


def _generalization_study(cfg: ExperimentConfig, dataset: Dataset, model: MultilayerPerceptron) -> None:
    """Ejercicio 1c: Estudia generalización usando k-fold cross-validation.

    Recibe dataset SIN normalizar — la normalización se computa por fold
    usando solo los datos de entrenamiento de ese fold.
    """
    #TODO: pensar en otras maneras de splitting como k-fold.

    # Step 1: Split dataset into train_val (80%) and test (20%) — test is NOT touched until step 4
    train_val_ds, _, test_ds, split_indices = dataset.split(train=0.8, val=0.0, test=0.2, seed=cfg.seed)
    test_idx = split_indices[2]

    # Load fraud labels for final classify_data call — aligned with test indices
    fraud_labels_full = load_csv(cfg.data_path, target_column="flagged_fraud").zeta
    test_fraud_labels = fraud_labels_full[test_idx]

    # Step 2: Grid search over eta, epochs, beta
    # TODO: consider reducing grid size if runtime is too slow
    etas   = [0.001, 0.01, 0.1]
    epochs = [100, 200, 500]
    betas  = [0.5, 1.0, 2.0]

    total_combos = len(etas) * len(epochs) * len(betas)
    print(f"\n{'=' * 55}")
    print(f"  FASE 2 — Grid Search + K-Fold ({total_combos} combinaciones)")
    print(f"{'=' * 55}")

    results = []
    combo_n = 0
    for eta in etas:
        for ep in epochs:
            for beta in betas:
                combo_n += 1
                print(f"\n  ({combo_n}/{total_combos}) eta={eta}  epochs={ep}  beta={beta}")
                cfg_variant = dataclasses.replace(cfg, eta=eta, epochs=ep, beta=beta)
                avg_val_error = _run_kfold(cfg_variant, train_val_ds, model, run_label=f"{combo_n}/{total_combos}")
                params = {"eta": eta, "epochs": ep, "beta": beta}
                results.append((params, avg_val_error))
                print(f"  → avg_val_error={avg_val_error:.4f}")

    best_params, best_val_error = min(results, key=lambda x: x[1])
    print(f"\n{'─' * 55}")
    print(f"  Mejor combinacion: eta={best_params['eta']}  epochs={best_params['epochs']}  beta={best_params['beta']}")
    print(f"  avg_val_error={best_val_error:.4f}")
    print(f"{'─' * 55}")

    # Step 3: Train final model on full train_val with best params (no folds here — this is the final model)
    cfg_best = dataclasses.replace(cfg, eta=best_params["eta"], epochs=best_params["epochs"], beta=best_params["beta"])

    norm_train_val_X,    norm_test_X    = fit_normalize(train_val_ds.X,    test_ds.X)
    norm_train_val_zeta, = fit_normalize(train_val_ds.zeta)

    final_model = model.clone()
    final_trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=cfg_best.eta),
        metrics=[],
        cfg=cfg_best,
    )
    final_trainer.fit(final_model, norm_train_val_X, norm_train_val_zeta, X_val=None, zeta_val=None)

    # Step 4: Evaluate final model on test set — sweep thresholds to find best F1
    # TODO Ejercicio 1c: select best model and recommend fraud detection threshold

    #TODO: aca donde clasifico, debo usar el output del NUEVO MODELO.
    # estoy usando el viejo, porque todavia no esta implementado el nuevo.
    # test_new_model_output_dataset = NEW_MODEL.forward(test_dataset.X)
    test_predictions = np.array([final_model.forward(xi) for xi in norm_test_X])

    best_threshold = 0.5
    best_f1 = -1.0
    best_result = None
    threshold_results = []

    for threshold in np.arange(0.1, 1.0, 0.1):
        # TODO: add this as parameter in config. We assume that prob. >= threshold is a positive classification
        [false_pos, false_neg, true_pos, true_neg] = classify_data(
            test_fraud_labels, test_predictions, threshold=round(threshold, 1)
        )
        f1 = F1Metric().compute(false_pos, false_neg, true_pos, true_neg)
        threshold_results.append({
            "threshold": round(threshold, 1),
            "tp": true_pos, "tn": true_neg, "fp": false_pos, "fn": false_neg,
            "f1": f1,
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(threshold, 1)
            best_result = (false_pos, false_neg, true_pos, true_neg)

    false_pos, false_neg, true_pos, true_neg = best_result

    header = f"{'threshold':>9} | {'TP':>6} | {'TN':>6} | {'FP':>6} | {'FN':>6} | {'F1':>8}"
    print(f"\n{'=' * 55}")
    print(f"  FASE 3 — Evaluación final en test set (threshold sweep)")
    print(f"{'=' * 55}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for r in threshold_results:
        marker = "  ← best" if r["threshold"] == best_threshold else ""
        print(f"  {r['threshold']:>9.1f} | {int(r['tp']):>6} | {int(r['tn']):>6} | {int(r['fp']):>6} | {int(r['fn']):>6} | {r['f1']:>8.4f}{marker}")

    plot_threshold_sweep(threshold_results, best_threshold, "output/experiment/ej1/threshold_sweep.png")
    plot_confusion_matrix(true_pos, true_neg, false_pos, false_neg, best_threshold, "output/experiment/ej1/confusion_matrix.png")
    # TODO Ejercicio 1c: report this threshold as recommendation to CompanyX


def run(cfg: ExperimentConfig) -> None:
    """Ejercicio 1 — Detección de fraude.

    Entrena perceptrón simple lineal vs no lineal. Compara generalización.
    """
    target_columns = cfg.target_column
    excluded_columns = ["flagged_fraud"]

    dataset = load_csv(cfg.data_path, target_column=target_columns, columns_to_ignore=excluded_columns)

    # Parte 1: comparar capacidad de aprendizaje lineal vs no lineal (Ejercicio 1a/1b)
    _learning_study(cfg, dataset)

    # Parte 2: estudio de generalización con el modelo no lineal (Tanh)
    # Tanh se prefiere sobre lineal: la salida acotada es más apropiada para
    # estimación de probabilidad de fraude que la salida lineal no acotada.
    # TODO: Verify this choice against Part 1 curves before final submission
    n_inputs = dataset.X.shape[1]
    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=TanhActivation(beta=cfg.beta)),
    ])
    _generalization_study(cfg, dataset, model)
