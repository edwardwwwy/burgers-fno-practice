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
from src.training.train import save_checkpoint, save_metrics_summary, train_fno
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNO1d on synthetic 1D Burgers data.")
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
    train_metrics_path = PROJECT_ROOT / paths["output_dir"] / paths["train_metrics_file"]

    device = torch.device(config.get("device", "cpu"))
    splits = load_burgers_splits(dataset_path)

    model, history, best_metrics = train_fno(config, splits, device)
    save_checkpoint(model, config, history, best_metrics, checkpoint_path)

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_validation": best_metrics,
        "final_epoch": history[-1],
    }
    save_metrics_summary(summary, train_metrics_path)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved training metrics to {train_metrics_path}")
    print("Best validation metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
