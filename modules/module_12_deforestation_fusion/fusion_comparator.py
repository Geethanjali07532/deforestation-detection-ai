"""
fusion_comparator.py

Module 12: Deep Learning-Based Deforestation Detection Fusion Strategies
Comparative framework benchmarking the 4 multi-temporal fusion paradigms:
  1. Early Fusion (EF)
  2. Late Fusion (LF)
  3. Siamese Differencing (Siam-Diff)
  4. Siamese Concatenation (Siam-Concat)
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.module_11_siamese_change_detection.losses import CompoundChangeLoss, compute_change_metrics

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def count_parameters(model: nn.Module) -> int:
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_single_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 4,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[nn.Module, float]:
    """Trains a single multi-temporal fusion change detection network."""
    model.to(device)
    criterion = CompoundChangeLoss(alpha=0.5, pos_weight=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for t1_b, t2_b, mask_b in train_loader:
            t1_b, t2_b, mask_b = t1_b.to(device), t2_b.to(device), mask_b.to(device)
            optimizer.zero_grad()
            logits = model(t1_b, t2_b)
            loss = criterion(logits, mask_b)
            loss.backward()
            optimizer.step()

    train_time = time.time() - t0
    return model, train_time


def evaluate_single_fusion_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluates a multi-temporal fusion change detection model on the test set."""
    model.eval()
    model.to(device)

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    t0 = time.time()

    with torch.no_grad():
        for t1_b, t2_b, mask_b in test_loader:
            t1_b, t2_b = t1_b.to(device), t2_b.to(device)
            logits = model(t1_b, t2_b)
            preds_bin = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int).flatten()
            gt = mask_b.numpy().astype(int).flatten()

            total_tp += int(np.sum((preds_bin == 1) & (gt == 1)))
            total_fp += int(np.sum((preds_bin == 1) & (gt == 0)))
            total_tn += int(np.sum((preds_bin == 0) & (gt == 0)))
            total_fn += int(np.sum((preds_bin == 0) & (gt == 1)))

    infer_time_ms = (time.time() - t0) * 1000.0
    latency_per_tile = infer_time_ms / max(1, len(test_loader.dataset))

    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    dice = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "latency_ms": float(latency_per_tile)
    }


def compare_fusion_strategies(
    models_dict: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 4,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[pd.DataFrame, Dict[str, nn.Module]]:
    """
    Trains and benchmarks all 4 fusion strategies under identical data and optimization rules.
    Returns: (leaderboard_df, trained_models_dict)
    """
    records = []
    trained_models = {}

    for name, model in models_dict.items():
        print(f"Training {name:<22} ({count_parameters(model):,} parameters)...", end=" ", flush=True)
        trained_m, train_time = train_single_fusion_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )
        trained_models[name] = trained_m

        metrics = evaluate_single_fusion_model(trained_m, test_loader, device=device)
        print(f"Done in {train_time:.1f}s | Test IoU: {metrics['iou']*100:.2f}%, Dice: {metrics['dice']*100:.2f}%", flush=True)

        records.append({
            "Fusion Strategy": name,
            "Parameters": f"{count_parameters(model):,}",
            "Mean IoU": metrics["iou"],
            "Dice Score": metrics["dice"],
            "Pixel Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Train Time (s)": round(train_time, 2),
            "Latency (ms)": round(metrics["latency_ms"], 2)
        })

    df = pd.DataFrame(records).sort_values(by="Mean IoU", ascending=False).reset_index(drop=True)
    return df, trained_models
