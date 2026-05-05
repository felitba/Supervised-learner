import numpy as np
from src.activation.activation import Array
from src.network.model import Model
from src.metric.classify_data_mlp import classify_data_mlp


def evaluate_multiclass(model: Model, X: Array, y_true: Array) -> tuple[np.ndarray, dict]:
    """
    Single forward pass → confusion matrix → all metrics.

    Args:
        model:  trained MultilayerPerceptron
        X:      input images, shape (n_samples, 784)
        y_true: true integer labels, shape (n_samples,)  e.g. [3, 7, 0, ...]

    Returns:
        confusion: (10, 10) int array
        metrics:   dict with accuracy, f1, precision, recall, and per-class breakdowns
    """
    predictions = model.forward(X)
    confusion   = classify_data_mlp(y_true, predictions)  # builds confusion matrix

    n_classes   = confusion.shape[0]
    total       = np.sum(confusion)
    accuracy    = np.trace(confusion) / total if total > 0 else 0.0

    precisions, recalls, f1s, supports = [], [], [], []

    for i in range(n_classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp   # amount of "predicted i, was not i"
        fn = np.sum(confusion[i, :]) - tp   # amount of "was i, predicted other"

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        support   = int(np.sum(confusion[i, :]))          # total real samples of class i

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    metrics = {
        "accuracy":         accuracy,
        "f1_macro":         float(np.mean(f1s)),
        "precision_macro":  float(np.mean(precisions)),
        "recall_macro":     float(np.mean(recalls)),
        "per_class": [
            {
                "digit":     i,
                "precision": precisions[i],
                "recall":    recalls[i],
                "f1":        f1s[i],
                "support":   supports[i],
                "correct":   int(confusion[i, i]),
            }
            for i in range(n_classes)
        ],
    }

    return confusion, metrics


def print_report(confusion: np.ndarray, metrics: dict, goal: float = None) -> None:
    """
    Prints a clean terminal report: overall metrics, per-class table, confusion matrix.

    Args:
        confusion: output of evaluate_multiclass
        metrics:   output of evaluate_multiclass
        goal:      optional accuracy target (e.g. 0.98 for exercise 3)
    """
    W = 62
    SEP  = "─" * W
    SEP2 = "═" * W


    print(f"\n{SEP2}")
    print(f"  EVALUATION REPORT")
    print(SEP2)

    acc = metrics["accuracy"]
    f1  = metrics["f1_macro"]
    pre = metrics["precision_macro"]
    rec = metrics["recall_macro"]

    print(f"  Accuracy          {acc:.4f}   ({acc*100:.2f}%)")
    print(f"  F1  (macro)       {f1:.4f}")
    print(f"  Precision (macro) {pre:.4f}")
    print(f"  Recall    (macro) {rec:.4f}")

    if goal is not None:
        met    = acc >= goal
        symbol = "✓" if met else "✗"
        status = "REACHED" if met else "NOT YET"
        print(f"  Goal ≥ {goal:.0%}        {symbol}  {status}  (gap: {(acc - goal)*100:+.2f}%)")

    print(f"\n{SEP}")
    print(f"  {'Digit':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Correct':>8}  {'Total':>6}")
    print(SEP)

    for pc in metrics["per_class"]:
        flag = "  ←" if pc["f1"] < 0.90 else ""
        print(
            f"  {pc['digit']:>5}  "
            f"{pc['precision']:>10.4f}  "
            f"{pc['recall']:>8.4f}  "
            f"{pc['f1']:>8.4f}  "
            f"{pc['correct']:>8}  "
            f"{pc['support']:>6}"
            f"{flag}"
        )