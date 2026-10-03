"""
mc_dropout_evaluator.py

Module 18: Model Evaluation, Robustness & Uncertainty Quantification
Implements Bayesian Monte Carlo Dropout (MCDO) to quantify epistemic model uncertainty
during multi-temporal deforestation change detection (Gal & Ghahramani, 2016).
"""

import os
import sys
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MCDropoutChangeDetector(nn.Module):
    """
    Siamese Change Detection Network with integrated spatial dropout layers
    designed specifically for test-time Monte Carlo uncertainty sampling.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1, base_features: int = 16, dropout_p: float = 0.25):
        super().__init__()
        f = base_features
        self.dropout_p = dropout_p

        # Dual shared encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )

        # Decoder head
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(f * 4, f * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(f * 2, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        return e1, e2, b

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1_1, e2_1, b_1 = self._encode(t1)
        e1_2, e2_2, b_2 = self._encode(t2)

        diff_b = torch.abs(b_1 - b_2)
        diff_2 = torch.abs(e2_1 - e2_2)
        diff_1 = torch.abs(e1_1 - e1_2)

        d2 = self.up2(diff_b)
        d2 = torch.cat([d2, diff_2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, diff_1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


def estimate_epistemic_uncertainty(
    model: nn.Module,
    t1_tensor: torch.Tensor,
    t2_tensor: torch.Tensor,
    num_passes: int = 10,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs T stochastic forward passes with active dropout to compute:
      - mean_prob (mu): Bayesian expectation of deforestation probability
      - std_prob (sigma): Epistemic uncertainty (standard deviation)
      - entropy: Predictive information entropy: -p*log(p) - (1-p)*log(1-p)
    """
    model.eval()
    # Force dropout layers into training mode during test evaluation
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    model.to(device)
    t1_tensor = t1_tensor.to(device)
    t2_tensor = t2_tensor.to(device)

    pass_probs = []

    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(t1_tensor, t2_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            pass_probs.append(probs)

    pass_probs = np.array(pass_probs)  # (T, H, W)

    mean_prob = np.mean(pass_probs, axis=0)
    std_prob = np.std(pass_probs, axis=0)

    # Predictive entropy
    p_safe = np.clip(mean_prob, 1e-7, 1.0 - 1e-7)
    entropy = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))

    return mean_prob, std_prob, entropy
