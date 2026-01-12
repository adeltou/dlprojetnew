"""
Package Models pour le projet RDD2022
Contient les architectures de segmentation sémantique
"""

from .unet_scratch import UNetScratch, create_unet_model
from .yolo_pretrained import YOLODetection, create_yolo_model, create_yolo_data_yaml
from .hybrid_model import HybridModel, create_hybrid_model
from .model_utils import (
    dice_coefficient,
    dice_loss,
    iou_metric,
    combined_loss,
    focal_loss,
    DiceCoefficient,
    IoUMetric
)

__all__ = [
    # U-Net
    'UNetScratch',
    'create_unet_model',

    # YOLO
    'YOLODetection',
    'create_yolo_model',
    'create_yolo_data_yaml',

    # Hybrid
    'HybridModel',
    'create_hybrid_model',

    # Métriques et Loss
    'dice_coefficient',
    'dice_loss',
    'iou_metric',
    'combined_loss',
    'focal_loss',
    'DiceCoefficient',
    'IoUMetric'
]