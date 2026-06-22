# burgers-fno-practice

A small, research-style PyTorch project for learning solution operators of the 1D viscous Burgers equation with a Fourier Neural Operator (FNO).A Codex-assisted PyTorch implementation and benchmark of Fourier Neural Operators for 1D Burgers’ equation.

This repository is designed as a practice project before building a more serious 1D Schrödinger equation FNO benchmark. The emphasis is on a complete and reproducible workflow rather than a large-scale benchmark: synthetic data generation, FNO training, a simple CNN baseline, evaluation, plots, and a fast CPU smoke test.

## Motivation

Neural operators learn mappings between function spaces. Instead of predicting a single scalar label, the model learns an operator such as:

```text
initial field -> future field
```

The 1D viscous Burgers equation is a useful practice problem because it is simple enough to simulate locally, but still contains nonlinear advection and diffusion. This makes it a good stepping stone before moving to more delicate PDE benchmarks.

## Problem Definition

The target PDE is the 1D viscous Burgers equation:

```text
u_t + u u_x = nu u_xx
```

The domain is:

```text
x in [0, 1]
```

with periodic boundary conditions.

The first-version operator learning task is:

```text
u(x, 0) -> u(x, T)
```

The project generates smooth random periodic initial conditions, evolves them with a small numerical solver, and trains models to predict the final state.

## Why Fourier Neural Operator

Fourier Neural Operators are a natural fit for PDE operator learning because they mix information in the frequency domain. For periodic 1D problems, the FFT-based spectral convolution is especially convenient:

- low Fourier modes capture smooth global structure
- mode truncation gives a compact operator layer
- the model can learn a mapping between full discretized functions

This project compares the FNO against a simple local 1D CNN baseline. The CNN baseline is intentionally modest: it helps show whether the spectral model is adding value without turning the repository into a large benchmark suite.

## Project Workflow

The repository supports this workflow:

1. Generate synthetic train, validation, and test data.
2. Train the FNO1d model.
3. Train the simple 1D CNN baseline.
4. Evaluate both models on the same test split.
5. Generate result plots and a summary report.
6. Run a fast CPU smoke test for sanity checking.

## File Structure

```text
configs/default.yaml              # Main experiment config
src/data/burgers.py               # Synthetic Burgers data generation and loading
src/models/fno1d.py               # Minimal readable FNO1d implementation
src/models/baseline_cnn.py        # Simple 1D CNN baseline
src/training/train.py             # Shared training loop and checkpoint utilities
src/evaluation/evaluate.py        # Checkpoint loading and test-set evaluation
src/visualization/plot_results.py # Plot and report generation
src/utils/config.py               # YAML config loading
src/utils/metrics.py              # MSE/relative L2 utilities
src/utils/reproducibility.py      # Seeding utilities
scripts/generate_data.py          # Data generation entry point
scripts/train_fno.py              # FNO training entry point
scripts/train_baseline.py         # Baseline training entry point
scripts/evaluate.py               # Evaluation entry point
scripts/plot_results.py           # Visualization/report entry point
scripts/smoke_test.py             # Fast CPU smoke test
tests/test_smoke.py               # Pytest smoke test wrapper
data/.gitkeep                     # Generated data directory placeholder
outputs/.gitkeep                  # Generated outputs directory placeholder
checkpoints/.gitkeep              # Generated checkpoint directory placeholder
```

Generated data, checkpoints, metrics, plots, and cache files are ignored by Git.

## Installation

Create and activate your preferred Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The default configuration is intentionally CPU-friendly.

## Reproduce the Workflow

Run commands from the repository root.

Generate synthetic Burgers data:

```bash
python scripts/generate_data.py --config configs/default.yaml
```

Train the FNO model:

```bash
python scripts/train_fno.py --config configs/default.yaml
```

Train the CNN baseline:

```bash
python scripts/train_baseline.py --config configs/default.yaml
```

Evaluate both models:

```bash
python scripts/evaluate.py --config configs/default.yaml --model fno
python scripts/evaluate.py --config configs/default.yaml --model baseline
```

Generate plots and result summary:

```bash
python scripts/plot_results.py --config configs/default.yaml
```

Open the guided notebook:

```text
notebooks/burgers_fno_demo.ipynb
```

The notebook explains the Burgers PDE background, the operator-learning task, the
FNO1d model, the generated dataset, evaluation metrics, prediction plots, and a
multi-step qualitative FNO rollout.

Run the fast smoke test:

```bash
python scripts/smoke_test.py
pytest tests/test_smoke.py
```

## Expected Outputs

After running the full workflow, generated files appear under `data/`, `checkpoints/`, and `outputs/`.

Typical generated files include:

```text
data/burgers_1d.npz
checkpoints/fno1d.pt
checkpoints/baseline_cnn.pt
outputs/fno_train_metrics.json
outputs/fno_eval_metrics.json
outputs/baseline_train_metrics.json
outputs/baseline_eval_metrics.json
outputs/model_comparison_metrics.json
outputs/result_summary.md
outputs/prediction_comparison.png
outputs/fno_rollout_comparison.png
outputs/loss_curves.png
```

These files are intentionally not committed. They can be regenerated from the commands above.

## Reproducibility Notes

- Experiment settings live in `configs/default.yaml`.
- Scripts seed Python, NumPy, and PyTorch through `src/utils/reproducibility.py`.
- The default dataset and models are small so the full workflow can run quickly on CPU.
- Metrics include MSE and relative L2 error.
- FNO and baseline evaluation use the same generated test split.

## Limitations

This is a practice benchmark, not a final scientific result.

- The dataset is synthetic and intentionally small.
- The numerical solver is simple and configured for small CPU experiments.
- The first task predicts only `u(x, T)`, not the full trajectory `u(x, t)`.
- Hyperparameters are chosen for readability and speed, not optimal performance.
- The CNN baseline is deliberately simple.
- There is no experiment tracking framework or large-scale sweep.

## Next Step

The next research step is to use this repository as a template for a 1D Schrödinger equation FNO benchmark. The useful pieces to carry forward are:

- reproducible config-driven scripts
- synthetic or curated PDE dataset generation
- clear train/validation/test split handling
- FNO implementation and baseline comparison
- metrics, plots, and smoke tests

For the Schrödinger benchmark, the main changes would be the PDE solver, data representation, model output target, and domain-specific evaluation metrics.
