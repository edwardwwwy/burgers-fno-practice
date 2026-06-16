from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.burgers import BurgersDataset
from src.models.baseline_cnn import BaselineCNN1d
from src.models.fno1d import FNO1d
from src.utils.metrics import relative_l2_error


def make_dataloader(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = BurgersDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_fno_from_config(config: dict) -> FNO1d:
    model_config = config["model"]
    return FNO1d(
        modes=int(model_config["modes"]),
        width=int(model_config["width"]),
        layers=int(model_config["layers"]),
    )


def make_baseline_from_config(config: dict) -> BaselineCNN1d:
    baseline_config = config["baseline"]
    return BaselineCNN1d(
        channels=int(baseline_config["channels"]),
        layers=int(baseline_config["layers"]),
        kernel_size=int(baseline_config["kernel_size"]),
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one train or evaluation epoch.

    If optimizer is provided, model parameters are updated. Otherwise this runs
    without gradients. Tensors keep the project convention [batch, n_grid].
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_rel_l2 = 0.0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)

        rel_l2 = relative_l2_error(predictions.detach(), targets)
        total_loss += float(loss.item()) * batch_size
        total_rel_l2 += float(rel_l2.item()) * batch_size
        total_examples += batch_size

    return {
        "mse": total_loss / total_examples,
        "relative_l2": total_rel_l2 / total_examples,
    }


def train_model(
    model: nn.Module,
    config: dict,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float]]:
    """Train an operator model and return the best-validation checkpoint state."""
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])

    train_loader = make_dataloader(*splits["train"], batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(*splits["val"], batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    loss_fn = nn.MSELoss()

    history = []
    best_state = None
    best_val_mse = float("inf")
    best_metrics = {}

    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_relative_l2": train_metrics["relative_l2"],
            "val_mse": val_metrics["mse"],
            "val_relative_l2": val_metrics["relative_l2"],
        }
        history.append(epoch_metrics)

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_metrics = epoch_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch {epoch:03d} | "
            f"train mse {train_metrics['mse']:.6f} | "
            f"val mse {val_metrics['mse']:.6f} | "
            f"val rel_l2 {val_metrics['relative_l2']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_metrics


def train_fno(
    config: dict,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[FNO1d, list[dict[str, float]], dict[str, float]]:
    """Train FNO1d and return the model, epoch history, and best validation metrics."""
    model = make_fno_from_config(config)
    trained_model, history, best_metrics = train_model(model, config, splits, device)
    return trained_model, history, best_metrics


def train_baseline(
    config: dict,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[BaselineCNN1d, list[dict[str, float]], dict[str, float]]:
    """Train the simple 1D CNN baseline on the same data splits as FNO."""
    model = make_baseline_from_config(config)
    trained_model, history, best_metrics = train_model(model, config, splits, device)
    return trained_model, history, best_metrics


def save_checkpoint(
    model: nn.Module,
    config: dict,
    history: list[dict[str, float]],
    best_metrics: dict[str, float],
    path: str | Path,
    model_name: str,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )


def save_metrics_summary(summary: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
