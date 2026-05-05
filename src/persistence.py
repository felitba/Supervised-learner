import json
import pathlib
import numpy as np
import dataclasses
from src.config import ExperimentConfig
from src.network.multilayer_perceptron import MultilayerPerceptron


def save_run(
        run_id: str,
        cfg: ExperimentConfig,
        model: MultilayerPerceptron,
        history: dict,
        metrics: dict[str, float],
) -> None:
    """Guarda en results/<run_id>/: config.json, weights.npz, history.json, metrics.json"""
    out_dir = pathlib.Path(f"results/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # Save history and metrics
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save weights and biases
    weights_dict = {}
    for i, (w, b) in enumerate(model.get_weights()):
        weights_dict[f"layer_{i}_w"] = w
        weights_dict[f"layer_{i}_b"] = b

    np.savez(out_dir / "weights.npz", **weights_dict)


def load_run(run_id: str) -> dict:
    """Carga todo lo guardado por save_run.

    Returns:
        dict con claves: config, weights, history, metrics
    """
    path = pathlib.Path(f"results/{run_id}")

    # Load config from file
    with open(path / "config.json", "r") as f:
        cfg_data = json.load(f)
        config = ExperimentConfig(**cfg_data)

    # Load history and metrics
    with open(path / "history.json", "r") as f:
        history = json.load(f)

    with open(path / "metrics.json", "r") as f:
        metrics = json.load(f)

    # Load weights and biases
    data = np.load(path / "weights.npz")
    weights = []
    i = 0
    while f"layer_{i}_w" in data:
        weights.append((data[f"layer_{i}_w"], data[f"layer_{i}_b"]))
        i += 1

    return {
        "config": config,
        "weights": weights,
        "history": history,
        "metrics": metrics
    }