"""
Module 9: Forest Segmentation Using U-Net
Deep Semantic Segmentation:
  - Encoder-Decoder Architecture with Skip Connections
  - Custom 5-Band Multispectral U-Net
  - Combined BCE + Dice Loss for Imbalance Handling
  - Pixel-Level Forest vs. Non-Forest Mask Prediction
  - Evaluation via IoU (Jaccard) and Dice Score
"""

from .unet_model import ForestUNet
from .losses_metrics import DiceLoss, CombinedBCEDiceLoss, compute_iou_dice
from .segmentation_dataset import ForestSegmentationDataset, create_segmentation_dataloaders
from .trainer import train_unet, evaluate_unet

__all__ = [
    "ForestUNet",
    "DiceLoss",
    "CombinedBCEDiceLoss",
    "compute_iou_dice",
    "ForestSegmentationDataset",
    "create_segmentation_dataloaders",
    "train_unet",
    "evaluate_unet"
]
