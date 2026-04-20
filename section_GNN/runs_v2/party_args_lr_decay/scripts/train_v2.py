from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

import sys
from pathlib import Path
_SECTION_GNN = Path(__file__).resolve().parents[3]
if str(_SECTION_GNN) not in sys.path:
    sys.path.insert(0, str(_SECTION_GNN))

from src.models.hetero_gnn import HeteroLegalOutcomeGNN
from src.training.evaluate import evaluate_split


def _build_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor | None:
    unique = np.unique(y_train)
    if len(unique) <= 1:
        return None
    weights = np.ones((num_classes,), dtype=np.float32)
    observed_weights = compute_class_weight(class_weight="balanced", classes=unique, y=y_train)
    for cls_idx, cls_weight in zip(unique.tolist(), observed_weights.tolist()):
        weights[int(cls_idx)] = float(cls_weight)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    data: Any,
    label_names: list[str],
    cfg: dict[str, Any],
    seed: int,
    logger: Any | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    input_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}
    num_classes = len(label_names)
    model = HeteroLegalOutcomeGNN(
        metadata=data.metadata(),
        input_dims=input_dims,
        out_dim=num_classes,
        cfg=cfg.get("model", {}),
    ).to(device)

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    # LR scheduler (ReduceLROnPlateau on val_macro_f1)
    scheduler = None
    if train_cfg.get("lr_scheduler") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("lr_scheduler_factor", 0.5)),
            patience=int(train_cfg.get("lr_scheduler_patience", 8)),
            min_lr=float(train_cfg.get("lr_min", 1e-6)),
        )

    train_mask = data["case"].train_mask
    val_mask = data["case"].val_mask
    test_mask = data["case"].test_mask
    y_case = data["case"].y

    class_weights = None
    if str(train_cfg.get("class_weight", "balanced")) == "balanced":
        class_weights = _build_class_weights(y_case[train_mask].detach().cpu().numpy(), num_classes=num_classes)
        if class_weights is not None:
            class_weights = class_weights.to(device)

    best_state = deepcopy(model.state_dict())
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience = int(train_cfg.get("early_stopping_patience", 10))
    use_early_stopping = bool(train_cfg.get("use_early_stopping", True))
    epochs = int(train_cfg.get("epochs", 90))
    log_every = int(train_cfg.get("log_every", 5))
    patience_counter = 0
    history: list[dict[str, Any]] = []
    current_lr = float(train_cfg.get("lr", 1e-3))

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(logits[train_mask], y_case[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval, embeddings = model(data.x_dict, data.edge_index_dict)
            probabilities = torch.softmax(logits_eval, dim=-1)
            predictions = probabilities.argmax(dim=-1)

        train_metrics = evaluate_split(y_true=y_case[train_mask].detach().cpu().numpy(), y_pred=predictions[train_mask].detach().cpu().numpy(), y_proba=probabilities[train_mask].detach().cpu().numpy(), label_names=label_names)
        val_metrics = evaluate_split(y_true=y_case[val_mask].detach().cpu().numpy(), y_pred=predictions[val_mask].detach().cpu().numpy(), y_proba=probabilities[val_mask].detach().cpu().numpy(), label_names=label_names)
        train_macro_f1 = float(train_metrics.get("macro_f1", 0.0))
        train_accuracy = float(train_metrics.get("accuracy", 0.0))
        val_macro_f1 = float(val_metrics.get("macro_f1", 0.0))
        val_accuracy = float(val_metrics.get("accuracy", 0.0))

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_macro_f1)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr and logger is not None:
                logger.info("LR decayed: %.6f → %.6f at epoch=%d", prev_lr, new_lr, epoch)
            current_lr = new_lr

        is_best = False
        if val_macro_f1 >= best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "best_val_macro_f1_so_far": best_val_macro_f1,
            "patience_counter": patience_counter,
            "lr": current_lr,
            "is_best": is_best,
        })

        if logger is not None and (epoch == 1 or epoch % log_every == 0):
            logger.info(
                "epoch=%d/%d train_loss=%.4f train_macro_f1=%.4f val_macro_f1=%.4f best_val_macro_f1=%.4f patience_counter=%d lr=%.6f",
                epoch, epochs, float(loss.item()), train_macro_f1, val_macro_f1, best_val_macro_f1, patience_counter, current_lr,
            )

        if use_early_stopping and patience_counter >= patience:
            if logger is not None:
                logger.info("Early stopping at epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, hidden = model(data.x_dict, data.edge_index_dict)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

    predictions_np = predictions.detach().cpu().numpy()
    probabilities_np = probabilities.detach().cpu().numpy()
    y_np = y_case.detach().cpu().numpy()

    split_masks = {"train": train_mask.detach().cpu().numpy(), "val": val_mask.detach().cpu().numpy(), "test": test_mask.detach().cpu().numpy()}
    split_metrics = {split_name: evaluate_split(y_true=y_np[mask], y_pred=predictions_np[mask], y_proba=probabilities_np[mask], label_names=label_names) for split_name, mask in split_masks.items()}

    prediction_frame = pd.DataFrame({
        "case_id": list(data["case"].case_id),
        "file_name": list(data["case"].file_name),
        "raw_label": list(data["case"].raw_label),
        "split": ["train" if split_masks["train"][idx] else "val" if split_masks["val"][idx] else "test" for idx in range(len(data["case"].case_id))],
        "target_index": y_np.tolist(),
        "target_label": [label_names[idx] for idx in y_np.tolist()],
        "pred_index": predictions_np.tolist(),
        "pred_label": [label_names[idx] for idx in predictions_np.tolist()],
        "confidence": probabilities_np.max(axis=1).tolist(),
    })

    return {
        "model_state_dict": best_state,
        "metrics": {
            "train": split_metrics["train"],
            "val": split_metrics["val"],
            "test": split_metrics["test"],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "epochs_completed": len(history),
            "early_stopping_enabled": use_early_stopping,
            "early_stopping_patience": patience,
            "history": history,
        },
        "predictions_df": prediction_frame,
        "logits": logits.detach().cpu(),
        "probabilities": probabilities.detach().cpu(),
        "case_embeddings": hidden["case"].detach().cpu(),
    }
