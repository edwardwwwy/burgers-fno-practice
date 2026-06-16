# burgers-fno-practice
A PyTorch implementation and benchmark of Fourier Neural Operators for 1D PDEs.

## Current workflow

```bash
python scripts/generate_data.py --config configs/default.yaml
python scripts/train_fno.py --config configs/default.yaml
python scripts/train_baseline.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --model fno
python scripts/evaluate.py --config configs/default.yaml --model baseline
```
