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

from src.utils.config import load_config
from src.utils.reproducibility import seed_everything
from src.visualization.plot_results import create_result_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create result plots and a short Burgers benchmark summary.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--sample-index", type=int, default=0, help="Test sample index to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(1)

    created_paths = create_result_report(
        config=config,
        project_root=PROJECT_ROOT,
        sample_index=args.sample_index,
    )

    print("Saved result report artifacts:")
    for path in created_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
