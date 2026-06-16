from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.burgers import generate_burgers_splits, save_burgers_splits
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 1D Burgers train/val/test data.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    seed_everything(int(config["seed"]))

    paths = config["paths"]
    dataset_path = PROJECT_ROOT / paths["data_dir"] / paths["dataset_file"]

    splits = generate_burgers_splits(config)
    save_burgers_splits(splits, dataset_path)

    print(f"Saved Burgers dataset to {dataset_path}")
    for split_name, (inputs, targets) in splits.items():
        print(f"{split_name}: inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}")


if __name__ == "__main__":
    main()
