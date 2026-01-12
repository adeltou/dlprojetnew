"""
Fonctions utilitaires pour les modèles de segmentation
Métriques : IoU, Dice Coefficient
Loss functions : Dice Loss, Combined Loss
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Import de la configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import METRICS_CONFIG, NUM_CLASSES


# ============================================================================
# MÉTRIQUES DE SEGMENTATION
# ============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calcule le coefficient de Dice (F1-score pour la segmentation)
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        y_true: Ground truth mask (batch_size, H, W, num_classes) ou (batch_size, H, W)
        y_pred: Predicted mask (batch_size, H, W, num_classes)
        smooth: Valeur pour éviter division par zéro
        
    Returns:
        Dice coefficient (scalaire entre 0 et 1)
    """
    # Aplatir les tenseurs
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    # Intersection et union
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    
    # Coefficient de Dice
    dice = (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )
    
    return dice


def dice_loss(y_true, y_pred):
    """
    Loss basée sur le coefficient de Dice
    Loss = 1 - Dice
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        
    Returns:
        Dice loss (scalaire)
    """
    return 1 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Calcule l'IoU (Intersection over Union) / Jaccard Index
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        smooth: Valeur pour éviter division par zéro
        
    Returns:
        IoU score (scalaire entre 0 et 1)
    """
    # Aplatir les tenseurs
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    # Intersection
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    
    # Union
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    
    # IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def mean_iou(y_true, y_pred, num_classes=NUM_CLASSES + 1):
    """
    Calcule l'IoU moyen sur toutes les classes
    
    Args:
        y_true: Ground truth (batch_size, H, W, num_classes)
        y_pred: Predictions (batch_size, H, W, num_classes)
        num_classes: Nombre de classes
        
    Returns:
        Mean IoU (scalaire)
    """
    iou_scores = []
    
    for class_id in range(num_classes):
        # Extraire les masques pour cette classe
        y_true_class = y_true[..., class_id]
        y_pred_class = y_pred[..., class_id]
        
        # Calculer l'IoU pour cette classe
        iou = iou_metric(y_true_class, y_pred_class)
        iou_scores.append(iou)
    
    # Moyenne des IoU
    return tf.reduce_mean(iou_scores)


# ============================================================================
# FONCTIONS DE LOSS
# ============================================================================

def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Loss combinée : alpha * Dice Loss + (1-alpha) * Categorical Crossentropy
    
    Cette combinaison permet de :
    - Dice Loss : Optimiser le chevauchement des masques
    - Crossentropy : Optimiser la classification pixel par pixel
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        alpha: Poids de la Dice loss (entre 0 et 1)
        
    Returns:
        Loss combinée
    """
    # Dice loss
    dice = dice_loss(y_true, y_pred)
    
    # Categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    cce = tf.reduce_mean(cce)
    
    # Combinaison
    return alpha * dice + (1 - alpha) * cce


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss pour gérer le déséquilibre des classes
    
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        alpha: Facteur de pondération
        gamma: Facteur de focalisation
        
    Returns:
        Focal loss
    """
    # Clip predictions pour éviter log(0)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred)
    
    # Focal term
    weight = alpha * tf.pow((1 - y_pred), gamma)
    
    # Focal loss
    loss = weight * cross_entropy
    
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# ============================================================================
# MÉTRIQUES PERSONNALISÉES POUR KERAS
# ============================================================================

class DiceCoefficient(keras.metrics.Metric):
    """
    Métrique Dice Coefficient pour Keras
    """
    
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        dice = dice_coefficient(y_true, y_pred)
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice_sum / self.count
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)


class IoUMetric(keras.metrics.Metric):
    """
    Métrique IoU pour Keras
    """
    
    def __init__(self, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = iou_metric(y_true, y_pred)
        self.iou_sum.assign_add(iou)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.iou_sum / self.count
    
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)


# ============================================================================
# FONCTIONS UTILITAIRES POUR L'ÉVALUATION
# ============================================================================

def calculate_class_iou(y_true, y_pred, class_id):
    """
    Calcule l'IoU pour une classe spécifique
    
    Args:
        y_true: Ground truth mask (H, W) ou (H, W, num_classes)
        y_pred: Predicted mask (H, W, num_classes)
        class_id: ID de la classe
        
    Returns:
        IoU pour cette classe
    """
    # Si y_pred est categorical, prendre l'argmax
    if len(y_pred.shape) == 3:
        y_pred_class = np.argmax(y_pred, axis=-1)
    else:
        y_pred_class = y_pred
    
    # Si y_true est categorical, prendre l'argmax
    if len(y_true.shape) == 3:
        y_true_class = np.argmax(y_true, axis=-1)
    else:
        y_true_class = y_true
    
    # Masques binaires pour cette classe
    true_mask = (y_true_class == class_id).astype(np.float32)
    pred_mask = (y_pred_class == class_id).astype(np.float32)
    
    # Intersection et union
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask) - intersection
    
    # IoU
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_metrics_per_class(y_true, y_pred, num_classes=NUM_CLASSES + 1):
    """
    Calcule les métriques (IoU, Dice) pour chaque classe
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        num_classes: Nombre de classes
        
    Returns:
        Dictionnaire {class_id: {'iou': X, 'dice': Y}}
    """
    metrics = {}
    
    for class_id in range(num_classes):
        iou = calculate_class_iou(y_true, y_pred, class_id)
        
        # Dice = 2 * IoU / (1 + IoU)
        if iou > 0:
            dice = 2 * iou / (1 + iou)
        else:
            dice = 0.0
        
        metrics[class_id] = {
            'iou': iou,
            'dice': dice
        }
    
    return metrics


# ============================================================================
# TESTS
# ============================================================================

def test_metrics():
    """
    Fonction de test des métriques
    """
    print("=" * 80)
    print("TEST DES MÉTRIQUES")
    print("=" * 80)
    
    # Créer des tensors de test
    batch_size = 2
    height = 128
    width = 128
    num_classes = NUM_CLASSES + 1
    
    # Ground truth et prédictions aléatoires
    y_true = tf.random.uniform((batch_size, height, width, num_classes))
    y_true = tf.one_hot(tf.argmax(y_true, axis=-1), num_classes)
    
    y_pred = tf.random.uniform((batch_size, height, width, num_classes))
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    print(f"\n📊 Shapes:")
    print(f"  y_true: {y_true.shape}")
    print(f"  y_pred: {y_pred.shape}")
    
    # Test Dice Coefficient
    print(f"\n🎲 Dice Coefficient:")
    dice = dice_coefficient(y_true, y_pred)
    print(f"  Dice: {dice.numpy():.4f}")
    
    # Test IoU
    print(f"\n📐 IoU Metric:")
    iou = iou_metric(y_true, y_pred)
    print(f"  IoU: {iou.numpy():.4f}")
    
    # Test Combined Loss
    print(f"\n⚖️  Combined Loss:")
    loss = combined_loss(y_true, y_pred, alpha=0.5)
    print(f"  Loss: {loss.numpy():.4f}")
    
    # Test Focal Loss
    print(f"\n🎯 Focal Loss:")
    f_loss = focal_loss(y_true, y_pred)
    print(f"  Focal Loss: {f_loss.numpy():.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS DES MÉTRIQUES SONT PASSÉS!")
    print("=" * 80)


if __name__ == "__main__":
    test_metrics()
