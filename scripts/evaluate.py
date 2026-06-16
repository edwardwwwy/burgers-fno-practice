from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.burgers import load_burgers_splits
from src.evaluation.evaluate import evaluate_split, load_model_checkpoint
from src.training.train import save_metrics_summary
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the Burgers test split.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--model",
        choices=("fno", "baseline"),
        default="fno",
        help="Which trained model checkpoint to evaluate.",
    )
    return parser.parse_args()


def model_paths(project_root: Path, paths: dict, model_name: str) -> tuple[Path, Path]:
    if model_name == "fno":
        checkpoint_file = paths["checkpoint_file"]
        metrics_file = paths["eval_metrics_file"]
    elif model_name == "baseline":
        checkpoint_file = paths["baseline_checkpoint_file"]
        metrics_file = paths["baseline_eval_metrics_file"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return (
        project_root / paths["checkpoint_dir"] / checkpoint_file,
        project_root / paths["output_dir"] / metrics_file,
    )


def maybe_write_comparison(project_root: Path, paths: dict) -> None:
    output_dir = project_root / paths["output_dir"]
    fno_metrics_path = output_dir / paths["eval_metrics_file"]
    baseline_metrics_path = output_dir / paths["baseline_eval_metrics_file"]

    if not fno_metrics_path.exists() or not baseline_metrics_path.exists():
        return

    with fno_metrics_path.open("r", encoding="utf-8") as file:
        fno_summary = json.load(file)
    with baseline_metrics_path.open("r", encoding="utf-8") as file:
        baseline_summary = json.load(file)

    comparison = {
        "fno": fno_summary["test"],
        "baseline": baseline_summary["test"],
    }

    comparison_path = output_dir / paths["comparison_metrics_file"]
    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print("Model comparison:")
    print(f"  FNO      mse={comparison['fno']['mse']:.6f}, rel_l2={comparison['fno']['relative_l2']:.6f}")
    print(
        "  Baseline "
        f"mse={comparison['baseline']['mse']:.6f}, "
        f"rel_l2={comparison['baseline']['relative_l2']:.6f}"
    )
    print(f"Saved comparison metrics to {comparison_path}")


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(1)

    paths = config["paths"]
    dataset_path = PROJECT_ROOT / paths["data_dir"] / paths["dataset_file"]
    checkpoint_path, eval_metrics_path = model_paths(PROJECT_ROOT, paths, args.model)

    device = torch.device(config.get("device", "cpu"))
    splits = load_burgers_splits(dataset_path)
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, model_name=args.model)

    test_inputs, test_targets = splits["test"]
    test_metrics = evaluate_split(
        model=model,
        inputs=test_inputs,
        targets=test_targets,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )

    summary = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "best_validation": checkpoint.get("best_metrics", {}),
        "test": test_metrics,
    }
    save_metrics_summary(summary, eval_metrics_path)

    print(f"Test metrics ({args.model}):")
    print(f"  mse: {test_metrics['mse']:.6f}")
    print(f"  relative_l2: {test_metrics['relative_l2']:.6f}")
    print(f"Saved evaluation metrics to {eval_metrics_path}")
    maybe_write_comparison(PROJECT_ROOT, paths)


if __name__ == "__main__":
    main()
