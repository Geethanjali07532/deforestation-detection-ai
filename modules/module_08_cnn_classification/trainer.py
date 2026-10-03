"""
trainer.py

Module 8: CNN-Based Land Cover Classification
Provides clean PyTorch training and evaluation loops for satellite patch models.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def train_land_cover_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 8,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains the CNN model using CrossEntropyLoss and Adam optimizer.
    Returns: (best_model, training_history)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_weights = None

    print(f"Training CNN on device: [{device}] for {epochs} epochs...")
    t0_start = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # 2. Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch:>2}/{epochs:>2}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:>5.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:>5.2f}%")

    total_time = time.time() - t0_start
    print(f"Training completed in {total_time:.2f} seconds. Best Val Accuracy: {best_val_acc:.2f}%\n")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates trained model on unseen test patches.
    Returns accuracy, confusion matrix, and classification report.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    t0 = time.time()

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    total_inference_time_ms = (time.time() - t0) * 1000.0
    latency_per_patch_ms = total_inference_time_ms / max(1, len(all_targets))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
        "latency_per_patch_ms": float(latency_per_patch_ms),
        "total_test_samples": len(all_targets)
    }
