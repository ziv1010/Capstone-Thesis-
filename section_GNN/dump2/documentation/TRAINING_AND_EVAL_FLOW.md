# TRAINING AND EVALUATION FLOW

This document details exactly how the `HeteroLegalOutcomeGNN` is optimized and evaluated across epochs.

## 1. High-Level Flow 

Training is orchestrating entirely by `scripts/train_gnn.py`.

```mermaid
graph TD
    classDef step fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef logic fill:#e1f5fe,stroke:#0288d1;
    classDef loss fill:#ffebee,stroke:#c62828;

    START[Initialize run_dir & load cache.pt]:::step
    SEED[set_global_seed]:::logic
    MODEL[Initialize HeteroLegalOutcomeGNN]:::step
    OPT[AdamW Optimizer]:::logic
    
    EPOCH{For each Epoch}:::step
    TRAIN_PASS[Forward Pass -> logits]:::step
    LOSS[Compute CrossEntropyLoss]:::loss
    BACKPROP[loss.backward() & optimizer.step()]:::logic
    
    EVAL_PASS[Validation Forward Pass]:::step
    METRIC_VAL[Calculate Macro F1 from probs]:::logic
    EARLY_STOP{If val.macro_f1 > best?}:::logic
    
    SAVE_STATE[Save Best State]:::step
    PATIENCE_INC[Increase patience_counter]:::logic
    
    BREAK{Early Stopping limit hit?}:::logic
    
    END[Final Artifact Plotting]:::step
    
    START --> SEED
    SEED --> MODEL
    MODEL --> OPT
    OPT --> EPOCH
    
    EPOCH --> TRAIN_PASS
    TRAIN_PASS --> LOSS
    LOSS --> BACKPROP
    BACKPROP --> EVAL_PASS
    
    EVAL_PASS --> METRIC_VAL
    METRIC_VAL --> EARLY_STOP
    
    EARLY_STOP -- Yes --> SAVE_STATE
    EARLY_STOP -- No --> PATIENCE_INC
    
    SAVE_STATE --> BREAK
    PATIENCE_INC --> BREAK
    
    BREAK -- Yes --> END
    BREAK -- No --> EPOCH
```

## 2. Masking Architecture

Because GNNs perform message passing, the *entire* graph is passed into PyTorch on every forward pass (there are no mini-batches in the default configuration).

**How train/val/test splits work on a single graph:**
1. The `HeteroData` object contains three boolean arrays under `data["case"].train_mask`, `.val_mask`, `.test_mask`.
2. When the model outputs `logits` (which has shape `[num_cases, num_classes]`), the network performs array slicing.
3. The loss is *only* computed over `logits[train_mask]`.
   ```python
   loss = F.cross_entropy(logits[train_mask], y_case[train_mask], weight=class_weights)
   ```
4. This means nodes in the validation set implicitly influence the embeddings of the training set if there are shared structural nodes. (In a `local_star_only` transductive ablation, they remain strictly independent). 

## 3. Loss & Optimizer

- **Loss Function:** PyTorch `CrossEntropyLoss`.
- **Handling Imbalance:** The target dataset predicts legal outcome. Appellants often win or lose at extremely skewed rates (e.g. 70/30). A raw neural network would just learn to predict "Win" linearly to minimize loss.
  - The framework computes `class_weight="balanced"` globally over the training inputs via Scikit-Learn.
  - Rare classes incur massive loss penalties when predicted incorrectly.
- **Optimizer:** `AdamW` ensures regularization. (Learning Rate defaults to `0.001`, `weight_decay=1e-5`).

## 4. Evaluation Strategy (Metrics)

Evaluated via `src/training/evaluate.py -> metrics.py`.
The most critical metric is **Macro F1**. Why?
Because in a 70/30 split, Accuracy implies you can just guess class A all day, resulting in 70% accuracy (which implies learning). Macro F1, however, computes the local F1 score corresponding to the minority class (30%) and the majority (70%) entirely separately, and then averages them.
If your model only guesses Class A, its Macro F1 will hover horribly around 0.41.

`train.py` checkpoints models relying **exclusively** on validation Macro F1.

```python
is_better_run = (
    run_val_macro_f1 > best_val_macro_f1
    or (run_val_macro_f1 == best_val_macro_f1 and run_val_accuracy > best_val_accuracy)
)
```

## 5. Artifact Generation

Upon training completion, `scripts/train_gnn.py` invokes charting routines inside `metrics.py`:
1. `save_split_metric_bar_plot`: Bar charts tracking generic F1 and accuracy across the 3 splits cleanly.
2. `save_training_history_plot`: A 2-tier graph explicitly monitoring over-fitting. Training loss descends smoothly. If Val-Macro-F1 curves off a cliff at Epoch 15 while Train climbs to 0.99, early-stopping kicks in.
3. `predictions.csv`: Complete diagnostic dump containing every file's true label index, predicted label, and maximum SoftMax floating point confidence. Highly readable.
