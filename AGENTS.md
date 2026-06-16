# AGENTS.md

## Purpose

This repository is for building a clean, reproducible, research-style coding project.

Use this file as the general working guide for agentic coding in this repository.

## Workflow

Before making major changes:

1. Inspect the current repository structure.
2. Summarize what already exists.
3. Propose a short implementation plan.
4. Wait for user approval before creating or modifying many files.
5. Make changes in small, verifiable steps.

After each major step, report:

* files created or modified
* purpose of the changes
* command to run for verification
* recommended next step

## Code quality

Prefer:

* simple and readable code
* clear file and function names
* minimal dependencies
* scripts that can be run from the repository root
* comments for nontrivial logic

Avoid:

* over-engineering
* unnecessary abstractions
* large unexplained code blocks
* changing many unrelated files at once

## Testing

For nontrivial code changes, add or update a small smoke test.

The smoke test should be fast and should verify that the main workflow runs without crashing.

## File safety

Do not commit generated data, model checkpoints, large outputs, cache files, virtual environments, or local machine-specific files.

Common files and directories to ignore include:

```text
__pycache__/
.pytest_cache/
.venv/
.ipynb_checkpoints/
data/*.npz
outputs/*
checkpoints/*
*.pt
*.pth
```

Use placeholder files such as `.gitkeep` when empty directories should be preserved in Git.
