from pathlib import Path
import os
import sys

# Some Windows/Anaconda environments load more than one OpenMP runtime when
# NumPy and PyTorch are installed from different channels. The smoke test is a
# tiny CPU-only sanity check, so allow it and keep the thread count small.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.burgers import generate_burgers_data
from src.models.fno1d import FNO1d
from src.utils.config import load_config
from src.utils.metrics import relative_l2_error
from src.utils.reproducibility import seed_everything


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    seed_everything(int(config["seed"]))
    torch.set_num_threads(1)

    device = torch.device("cpu")
    data_config = config["data"]
    smoke_config = config["smoke"]

    inputs, targets = generate_burgers_data(
        n_samples=int(smoke_config["n_samples"]),
        n_grid=int(smoke_config["n_grid"]),
        viscosity=float(data_config["viscosity"]),
        time_horizon=float(data_config["time_horizon"]),
        time_steps=int(smoke_config["time_steps"]),
        initial_condition_modes=int(data_config["initial_condition_modes"]),
        seed=int(config["seed"]),
    )
    inputs = inputs.to(device)
    targets = targets.to(device)

    model = FNO1d(
        modes=int(smoke_config["modes"]),
        width=int(smoke_config["width"]),
        layers=int(smoke_config["layers"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    loss_fn = nn.MSELoss()

    prediction = model(inputs)
    initial_loss = loss_fn(prediction, targets)
    initial_rel_l2 = relative_l2_error(prediction, targets)

    if prediction.shape != targets.shape:
        raise AssertionError(f"Expected prediction shape {targets.shape}, got {prediction.shape}")
    if not torch.isfinite(initial_loss):
        raise AssertionError("Initial loss is not finite")
    if not torch.isfinite(initial_rel_l2):
        raise AssertionError("Initial relative L2 error is not finite")

    for _ in range(int(smoke_config["train_steps"])):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_prediction = model(inputs)
        final_loss = loss_fn(final_prediction, targets)
        final_rel_l2 = relative_l2_error(final_prediction, targets)

    if not torch.isfinite(final_loss):
        raise AssertionError("Final loss is not finite")
    if not torch.isfinite(final_rel_l2):
        raise AssertionError("Final relative L2 error is not finite")

    print("Smoke test passed.")
    print(f"data shape: inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}")
    print(f"prediction shape: {tuple(final_prediction.shape)}")
    print(f"initial loss: {initial_loss.item():.6f}")
    print(f"final loss: {final_loss.item():.6f}")
    print(f"final relative L2 error: {final_rel_l2.item():.6f}")


if __name__ == "__main__":
    main()
