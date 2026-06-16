from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.training.train import (
    make_baseline_from_config,
    make_dataloader,
    make_fno_from_config,
    run_epoch,
)


def load_model_checkpoint(
    path: str | Path,
    device: torch.device,
    model_name: str,
) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the {model_name} model first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    if model_name == "fno":
        model = make_fno_from_config(config).to(device)
    elif model_name == "baseline":
        model = make_baseline_from_config(config).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def load_fno_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    return load_model_checkpoint(path, device, model_name="fno")


def evaluate_split(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataloader = make_dataloader(inputs, targets, batch_size=batch_size, shuffle=False)
    return run_epoch(model, dataloader, nn.MSELoss(), device)
