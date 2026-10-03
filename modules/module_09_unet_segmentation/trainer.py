"""
trainer.py

Module 9: Forest Segmentation Using U-Net
Training and evaluation loops for the U-Net architecture.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses_metrics import CombinedBCEDiceLoss, compute_iou_dice

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains the ForestUNet with Combined BCE + Dice Loss and Adam optimizer.
    """
    model.to(device)
    criterion = CombinedBCEDiceLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_iou": [],
        "train_dice": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": []
    }

    best_val_iou = 0.0
    best_weights = None
    t0_start = time.time()

    print(f"Training ForestUNet on device: [{device}] for {epochs} epochs...", flush=True)

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_iou_sum, train_dice_sum = 0.0, 0.0
        n_train_batches = len(train_loader)

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds_bin = (torch.sigmoid(logits) >= 0.5).float()
            iou, dice = compute_iou_dice(preds_bin, y_batch)
            train_iou_sum += iou
            train_dice_sum += dice

        ep_train_loss = train_loss / n_train_batches
        ep_train_iou = (train_iou_sum / n_train_batches) * 100.0
        ep_train_dice = (train_dice_sum / n_train_batches) * 100.0

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_iou_sum, val_dice_sum = 0.0, 0.0
        n_val_batches = len(val_loader)

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item()
                preds_bin = (torch.sigmoid(logits) >= 0.5).float()
                iou, dice = compute_iou_dice(preds_bin, y_batch)
                val_iou_sum += iou
                val_dice_sum += dice

        ep_val_loss = val_loss / max(1, n_val_batches)
        ep_val_iou = (val_iou_sum / max(1, n_val_batches)) * 100.0
        ep_val_dice = (val_dice_sum / max(1, n_val_batches)) * 100.0

        history["train_loss"].append(ep_train_loss)
        history["train_iou"].append(ep_train_iou)
        history["train_dice"].append(ep_train_dice)
        history["val_loss"].append(ep_val_loss)
        history["val_iou"].append(ep_val_iou)
        history["val_dice"].append(ep_val_dice)

        if ep_val_iou > best_val_iou:
            best_val_iou = ep_val_iou
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch:>2}/{epochs:>2}] | Train Loss: {ep_train_loss:.4f}, IoU: {ep_train_iou:>5.1f}%, Dice: {ep_train_dice:>5.1f}% | Val Loss: {ep_val_loss:.4f}, Val IoU: {ep_val_iou:>5.1f}%, Val Dice: {ep_val_dice:>5.1f}%", flush=True)

    total_time = time.time() - t0_start
    print(f"U-Net Training completed in {total_time:.2f} seconds. Best Val IoU: {best_val_iou:.2f}%\n", flush=True)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def evaluate_unet(model: nn.Module, test_loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """Evaluates U-Net on test set, computing IoU, Dice, Precision, Recall, and Pixel Accuracy."""
    model.eval()
    model.to(device)

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    t0 = time.time()

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds_bin = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            y_true = y_batch.numpy().astype(int)

            pred_flat = preds_bin.flatten()
            true_flat = y_true.flatten()

            total_tp += int(np.sum((pred_flat == 1) & (true_flat == 1)))
            total_fp += int(np.sum((pred_flat == 1) & (true_flat == 0)))
            total_tn += int(np.sum((pred_flat == 0) & (true_flat == 0)))
            total_fn += int(np.sum((pred_flat == 0) & (true_flat == 1)))

    total_time_ms = (time.time() - t0) * 1000.0
    latency_per_tile_ms = total_time_ms / max(1, len(test_loader.dataset))

    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_dice": float(f1),
        "iou": float(iou),
        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
        "latency_per_tile_ms": float(latency_per_tile_ms)
    }
