---
name: pde-fno-project
description: Use this skill when building or improving a research-style PyTorch PDE/FNO project, especially a 1D Burgers equation Fourier Neural Operator practice project.
---

# PDE/FNO Project Skill

## Project Goal

Use this skill to guide the construction of a complete, reproducible PyTorch project for operator learning on the 1D viscous Burgers equation.

The main model should be a Fourier Neural Operator (FNO), specifically a minimal and readable FNO1d implementation.

This repository is a practice project before building a later FNO benchmark for the 1D Schrodinger equation. Favor clarity, reproducibility, and small verifiable steps over broad feature coverage.

## Scientific Task

### PDE

The target partial differential equation is the 1D viscous Burgers equation:

```text
u_t + u u_x = nu u_xx
```

### Domain

```text
x in [0, 1]
```

### Boundary Condition

Use a periodic boundary condition.

### First Version Learning Task

Learn the operator mapping:

```text
u(x, 0) -> u(x, T)
```

### Future Extension

Extend the project later to full trajectory prediction:

```text
u(x, 0) -> u(x, t)
```

## Expected Components

The project should eventually contain:

- synthetic Burgers data generation
- minimal FNO1d model
- simple 1D CNN baseline
- training pipeline
- evaluation pipeline
- visualization
- smoke test
- config file
- reproducibility utilities
- GitHub-ready README

## Expected File Structure

The intended project structure is:

```text
configs/default.yaml
src/data/burgers.py
src/models/fno1d.py
src/models/baseline_cnn.py
src/training/train.py
src/evaluation/evaluate.py
src/visualization/plot_results.py
src/utils/config.py
src/utils/metrics.py
src/utils/reproducibility.py
scripts/generate_data.py
scripts/train_fno.py
scripts/train_baseline.py
scripts/evaluate.py
scripts/smoke_test.py
tests/test_smoke.py
data/.gitkeep
outputs/.gitkeep
checkpoints/.gitkeep
```

Do not create this full structure until the user explicitly approves an implementation plan.

## Implementation Rules

- Inspect the repository first.
- Summarize what already exists before implementing.
- Propose a short implementation plan before making substantial changes.
- Wait for user approval before large changes.
- Implement in small verifiable steps.
- Prioritize a CPU-friendly smoke test.
- Keep code beginner-readable.
- Avoid over-engineering.
- Do not commit generated data, outputs, checkpoints, model weights, or local caches.
- Keep scripts runnable from the repository root.

## Implementation Order

Use this recommended order when the user asks to build the project:

1. Update `.gitignore` and `requirements.txt`.
2. Create folder structure and config.
3. Implement reproducibility utilities and metrics.
4. Implement tiny data generation.
5. Implement FNO1d.
6. Implement smoke test.
7. Implement full data generation script.
8. Implement training pipeline.
9. Implement evaluation pipeline.
10. Implement baseline CNN.
11. Implement visualization.
12. Update README.
13. Run smoke test and fix bugs.

## First-Version Simplifications

For the first working version:

- Predict only `u(x, T)`, not the full trajectory.
- Use a small dataset by default.
- Use CPU-friendly hyperparameters.
- Use simple plots.
- Use simple config loading.
- Do not add an experiment tracking framework.
- Do not rely on an external PDE dataset.

## Key Risks

Pay special attention to:

- numerical solver stability
- tensor shape mistakes
- FFT mode truncation errors
- CPU runtime becoming too long
- `.gitignore` accidentally hiding `.gitkeep`
- checkpoint paths not existing
- evaluation expecting checkpoints that do not exist yet
- generated data accidentally committed

## Expected Final Commands

The final project should support these commands from the repository root:

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/generate_data.py --config configs/default.yaml
python scripts/train_fno.py --config configs/default.yaml
python scripts/train_baseline.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```
