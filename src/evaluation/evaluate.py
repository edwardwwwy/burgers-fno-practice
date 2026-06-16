from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.training.train import make_dataloader, make_fno_from_config, run_epoch


def load_fno_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run scripts/train_fno.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = make_fno_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_split(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataloader = make_dataloader(inputs, targets, batch_size=batch_size, shuffle=False)
    return run_epoch(model, dataloader, nn.MSELoss(), device)
