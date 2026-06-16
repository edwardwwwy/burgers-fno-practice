from pathlib import Path
import argparse
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
from src.evaluation.evaluate import evaluate_split, load_fno_checkpoint
from src.training.train import save_metrics_summary
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FNO1d checkpoint on the test split.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(1)

    paths = config["paths"]
    dataset_path = PROJECT_ROOT / paths["data_dir"] / paths["dataset_file"]
    checkpoint_path = PROJECT_ROOT / paths["checkpoint_dir"] / paths["checkpoint_file"]
    eval_metrics_path = PROJECT_ROOT / paths["output_dir"] / paths["eval_metrics_file"]

    device = torch.device(config.get("device", "cpu"))
    splits = load_burgers_splits(dataset_path)
    model, checkpoint = load_fno_checkpoint(checkpoint_path, device)

    test_inputs, test_targets = splits["test"]
    test_metrics = evaluate_split(
        model=model,
        inputs=test_inputs,
        targets=test_targets,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_validation": checkpoint.get("best_metrics", {}),
        "test": test_metrics,
    }
    save_metrics_summary(summary, eval_metrics_path)

    print("Test metrics:")
    print(f"  mse: {test_metrics['mse']:.6f}")
    print(f"  relative_l2: {test_metrics['relative_l2']:.6f}")
    print(f"Saved evaluation metrics to {eval_metrics_path}")


if __name__ == "__main__":
    main()
