# src/training

This package contains reusable training and evaluation logic.

## Files

- `dataset.py`: graph label/split validation and dataset helpers.
- `train.py`: training loop, early stopping, optimizer/scheduler handling.
- `evaluate.py`: split evaluation and prediction collection.
- `metrics.py`: metric computation and plots for history, split bars, and
  confusion matrices.

## Output Contract

Training functions return a dictionary containing:

- trained model state
- predictions as a DataFrame
- metric dictionaries for train/validation/test splits
- training history

Script wrappers save those outputs under `outputs/.../models/<run_name>/`.
