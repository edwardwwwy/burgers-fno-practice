from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.burgers import solve_burgers_periodic
from src.evaluation.evaluate import evaluate_split, load_model_checkpoint
from src.training.train import save_metrics_summary


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def collect_test_metrics(
    config: dict,
    project_root: Path,
    fno_model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    paths = config["paths"]
    output_dir = project_root / paths["output_dir"]
    fno_metrics = read_json(output_dir / paths["eval_metrics_file"])
    baseline_metrics = read_json(output_dir / paths["baseline_eval_metrics_file"])

    batch_size = int(config["training"]["batch_size"])
    if fno_metrics is None:
        fno_test = evaluate_split(fno_model, test_inputs, test_targets, batch_size, device)
    else:
        fno_test = fno_metrics["test"]

    if baseline_metrics is None:
        baseline_test = evaluate_split(baseline_model, test_inputs, test_targets, batch_size, device)
    else:
        baseline_test = baseline_metrics["test"]

    comparison = {"fno": fno_test, "baseline": baseline_test}
    save_metrics_summary(comparison, output_dir / paths["comparison_metrics_file"])
    return comparison


def plot_prediction_comparison(
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    fno_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    path: Path,
    sample_index: int = 0,
) -> None:
    """Plot one test example with predictions and pointwise absolute errors."""
    path.parent.mkdir(parents=True, exist_ok=True)

    u0 = test_inputs[sample_index].cpu().numpy()
    target = test_targets[sample_index].cpu().numpy()
    fno = fno_prediction[sample_index].cpu().numpy()
    baseline = baseline_prediction[sample_index].cpu().numpy()

    n_grid = u0.shape[0]
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(x, u0, label="initial u(x, 0)", color="tab:gray", linestyle="--")
    axes[0].plot(x, target, label="ground truth u(x, T)", color="black", linewidth=2)
    axes[0].plot(x, fno, label="FNO prediction", color="tab:blue")
    axes[0].plot(x, baseline, label="baseline prediction", color="tab:orange")
    axes[0].set_ylabel("u")
    axes[0].set_title("1D Burgers Operator Learning Prediction")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, np.abs(fno - target), label="|FNO - truth|", color="tab:blue")
    axes[1].plot(x, np.abs(baseline - target), label="|baseline - truth|", color="tab:orange")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("pointwise absolute error")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fno_rollout_comparison(
    test_inputs: torch.Tensor,
    fno_model: torch.nn.Module,
    config: dict,
    device: torch.device,
    path: Path,
    sample_index: int = 0,
    rollout_steps: int = 3,
) -> None:
    """Plot repeated one-step FNO predictions against numerical Burgers solves."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    viscosity = float(data_config["viscosity"])
    time_horizon = float(data_config["time_horizon"])
    time_steps = int(data_config["time_steps"])

    true_state = test_inputs[sample_index : sample_index + 1].cpu().numpy()
    predicted_state = test_inputs[sample_index : sample_index + 1].to(device)

    n_grid = true_state.shape[1]
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    fig, axes = plt.subplots(rollout_steps, 1, figsize=(9, 2.6 * rollout_steps), sharex=True)
    axes = np.atleast_1d(axes)

    fno_model.eval()
    for step, ax in enumerate(axes, start=1):
        true_state = solve_burgers_periodic(
            u0=true_state,
            viscosity=viscosity,
            time_horizon=time_horizon,
            time_steps=time_steps,
        )
        with torch.no_grad():
            predicted_state = fno_model(predicted_state).detach()

        prediction = predicted_state.cpu().numpy()
        mse = float(np.mean((prediction - true_state) ** 2))

        ax.plot(x, true_state[0], label=f"numerical truth t={step}T", color="black", linewidth=2)
        ax.plot(x, prediction[0], label=f"FNO rollout t={step}T", color="tab:blue")
        ax.set_ylabel("u")
        ax.set_title(f"Repeated FNO one-step rollout, step {step}, MSE={mse:.6f}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_loss_curves(
    fno_history: list[dict[str, float]],
    baseline_history: list[dict[str, float]],
    path: Path,
) -> bool:
    """Plot train/validation MSE curves when checkpoint histories are available."""
    if not fno_history or not baseline_history:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, history, color in (
        ("FNO", fno_history, "tab:blue"),
        ("Baseline", baseline_history, "tab:orange"),
    ):
        epochs = [row["epoch"] for row in history]
        train_mse = [row["train_mse"] for row in history]
        val_mse = [row["val_mse"] for row in history]
        ax.plot(epochs, train_mse, color=color, linestyle="--", label=f"{label} train MSE")
        ax.plot(epochs, val_mse, color=color, label=f"{label} val MSE")

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Training and Validation Loss Curves")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_result_summary(
    comparison: dict[str, dict[str, float]],
    prediction_plot: Path,
    rollout_plot: Path,
    loss_curve: Path,
    loss_curve_written: bool,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Burgers FNO Result Summary",
        "",
        "| model | test MSE | test relative L2 |",
        "| --- | ---: | ---: |",
        f"| FNO | {comparison['fno']['mse']:.6f} | {comparison['fno']['relative_l2']:.6f} |",
        (
            f"| Baseline CNN | {comparison['baseline']['mse']:.6f} | "
            f"{comparison['baseline']['relative_l2']:.6f} |"
        ),
        "",
        "Generated artifacts:",
        f"- Prediction comparison: `{prediction_plot.name}`",
        f"- Multi-step FNO rollout comparison: `{rollout_plot.name}`",
    ]

    if loss_curve_written:
        lines.append(f"- Loss curves: `{loss_curve.name}`")
    else:
        lines.append("- Loss curves: not generated because checkpoint histories were unavailable")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_result_report(config: dict, project_root: Path, sample_index: int = 0) -> list[Path]:
    paths = config["paths"]
    output_dir = project_root / paths["output_dir"]
    dataset_path = project_root / paths["data_dir"] / paths["dataset_file"]
    fno_checkpoint_path = project_root / paths["checkpoint_dir"] / paths["checkpoint_file"]
    baseline_checkpoint_path = project_root / paths["checkpoint_dir"] / paths["baseline_checkpoint_file"]

    require_file(dataset_path, "Run python scripts/generate_data.py --config configs/default.yaml first.")
    require_file(fno_checkpoint_path, "Run python scripts/train_fno.py --config configs/default.yaml first.")
    require_file(
        baseline_checkpoint_path,
        "Run python scripts/train_baseline.py --config configs/default.yaml first.",
    )

    from src.data.burgers import load_burgers_splits

    device = torch.device(config.get("device", "cpu"))
    splits = load_burgers_splits(dataset_path)
    test_inputs, test_targets = splits["test"]

    fno_model, fno_checkpoint = load_model_checkpoint(fno_checkpoint_path, device, model_name="fno")
    baseline_model, baseline_checkpoint = load_model_checkpoint(
        baseline_checkpoint_path,
        device,
        model_name="baseline",
    )

    fno_model.eval()
    baseline_model.eval()
    with torch.no_grad():
        fno_prediction = fno_model(test_inputs.to(device)).cpu()
        baseline_prediction = baseline_model(test_inputs.to(device)).cpu()

    comparison = collect_test_metrics(
        config=config,
        project_root=project_root,
        fno_model=fno_model,
        baseline_model=baseline_model,
        test_inputs=test_inputs,
        test_targets=test_targets,
        device=device,
    )

    prediction_plot = output_dir / paths["prediction_plot_file"]
    rollout_plot = output_dir / paths["rollout_plot_file"]
    loss_curve = output_dir / paths["loss_curve_file"]
    summary_path = output_dir / paths["result_summary_file"]

    plot_prediction_comparison(
        test_inputs=test_inputs,
        test_targets=test_targets,
        fno_prediction=fno_prediction,
        baseline_prediction=baseline_prediction,
        path=prediction_plot,
        sample_index=sample_index,
    )
    plot_fno_rollout_comparison(
        test_inputs=test_inputs,
        fno_model=fno_model,
        config=config,
        device=device,
        path=rollout_plot,
        sample_index=sample_index,
    )
    loss_curve_written = plot_loss_curves(
        fno_history=fno_checkpoint.get("history", []),
        baseline_history=baseline_checkpoint.get("history", []),
        path=loss_curve,
    )
    write_result_summary(
        comparison=comparison,
        prediction_plot=prediction_plot,
        rollout_plot=rollout_plot,
        loss_curve=loss_curve,
        loss_curve_written=loss_curve_written,
        path=summary_path,
    )

    created = [prediction_plot, rollout_plot, summary_path, output_dir / paths["comparison_metrics_file"]]
    if loss_curve_written:
        created.append(loss_curve)
    return created
