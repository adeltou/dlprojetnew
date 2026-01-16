"""
Module de Calcul des Métriques d'Évaluation - VERSION CORRIGÉE
Corrige les problèmes de mapping YOLO et de calcul des métriques
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix


# ============================================================================
# DÉTECTION ET GESTION DU TYPE DE MODÈLE
# ============================================================================

def is_yolo_model(model) -> bool:
    """
    Vérifie si le modèle est un modèle YOLO (Ultralytics)
    
    Args:
        model: Le modèle à vérifier
        
    Returns:
        True si c'est un modèle YOLO, False sinon
    """
    model_type = str(type(model))
    return 'ultralytics' in model_type.lower() or 'yolo' in model_type.lower()


def predict_with_yolo(model, batch_images: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Effectue des prédictions avec un modèle YOLO et convertit au format attendu
    
    CRITIQUE: YOLO nécessite:
    - Images en uint8 (0-255), pas normalisées
    - Conversion des class IDs YOLO (0,1,2,4) vers masque (0,1,2,3,4)
    
    Args:
        model: Modèle YOLO (Ultralytics)
        batch_images: Batch d'images NORMALISÉES (N, H, W, C) en float32 [0-1]
        target_size: Taille cible des masques de sortie
        
    Returns:
        Masques prédits (N, H, W) avec class IDs corrects (0,1,2,3,4)
    """
    predictions = []
    
    # Mapping YOLO class ID → Mask class ID
    # YOLO: 0=longitudinal, 1=transversal, 2=crocodile, 4=pothole
    # Mask: 0=background, 1=longitudinal, 2=transversal, 3=crocodile, 4=pothole
    YOLO_TO_MASK = {
        0: 1,  # longitudinal
        1: 2,  # transversal  
        2: 3,  # crocodile
        4: 4   # pothole
    }
    
    for img in batch_images:
        # 1. Dénormaliser l'image pour YOLO (0-1 → 0-255)
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 2. Redimensionner à 640x640 pour YOLO
        img_yolo = cv2.resize(img_uint8, (640, 640))
        
        # 3. Prédire avec YOLO - seuil de confiance bas pour détecter plus
        results = model.predict(
            source=img_yolo,
            conf=0.10,  # Seuil très bas pour détecter plus d'objets
            iou=0.45,
            imgsz=640,
            save=False,
            verbose=False
        )
        
        # 4. Créer masque de segmentation vide (background = 0)
        mask = np.zeros(target_size, dtype=np.uint8)
        
        # 5. Remplir le masque avec les détections YOLO
        if results[0].masks is not None and len(results[0].masks) > 0:
            for idx in range(len(results[0].masks)):
                # Récupérer le masque YOLO et la classe
                yolo_mask = results[0].masks.data[idx].cpu().numpy()
                yolo_class = int(results[0].boxes.cls[idx].cpu().numpy())
                
                # Redimensionner le masque YOLO à la taille cible
                yolo_mask_resized = cv2.resize(yolo_mask, target_size, interpolation=cv2.INTER_NEAREST)
                
                # Convertir YOLO class ID → Mask class ID
                mask_class = YOLO_TO_MASK.get(yolo_class, 0)
                
                # Appliquer au masque (les zones détectées)
                mask[yolo_mask_resized > 0.5] = mask_class
        
        predictions.append(mask)
    
    return np.array(predictions)


# ============================================================================
# ÉVALUATION GLOBALE DU MODÈLE
# ============================================================================

def evaluate_model_on_dataset(model,
                              images: np.ndarray,
                              masks_true: np.ndarray,
                              batch_size: int = 8,
                              model_name: str = None) -> Dict:
    """
    Évalue un modèle sur un dataset complet
    
    Cette fonction prend un modèle entraîné et calcule toutes les métriques
    importantes. VERSION CORRIGÉE pour gérer YOLO correctement.
    
    Args:
        model: Modèle Keras ou YOLO à évaluer
        images: Images normalisées (N, H, W, C) [0-1]
        masks_true: Masques vrais (N, H, W) avec class IDs
        batch_size: Taille des batches pour l'évaluation
        model_name: Nom du modèle (pour l'affichage)
        
    Returns:
        Dict contenant toutes les métriques
    """
    display_name = model_name if model_name else "modèle"
    num_samples = len(images)
    
    print(f"\n📊 Évaluation de {display_name} sur {num_samples} images...")
    print("-" * 80)
    
    # Déterminer le type de modèle
    is_yolo = is_yolo_model(model)
    if is_yolo:
        print(f"  🔍 Modèle YOLO détecté - utilisation de la prédiction YOLO spéciale")
    
    # Initialiser les accumulateurs
    all_y_true = []
    all_y_pred = []
    
    total_iou = 0
    total_dice = 0
    total_pixel_acc = 0
    
    # Évaluer batch par batch
    num_batches = (num_samples + batch_size - 1) // batch_size
    debug_printed = False

    for batch_idx in range(num_batches):
        # Indices de début et fin pour ce batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Extraire le batch
        batch_images = images[start_idx:end_idx]
        batch_masks_true = masks_true[start_idx:end_idx]
        
        # Prédiction selon le type de modèle
        if is_yolo:
            # YOLO: prédiction spéciale
            batch_masks_pred = predict_with_yolo(model, batch_images, target_size=(256, 256))
        else:
            # Keras (U-Net, Hybrid): prédiction standard
            batch_predictions = model.predict(batch_images, verbose=False)
            batch_masks_pred = np.argmax(batch_predictions, axis=-1)
        
        # Debug: afficher les informations du premier batch
        if not debug_printed:
            print(f"\n  🔍 DEBUG - Premier batch:")
            print(f"     - Shape images: {batch_images.shape}")
            print(f"     - Shape masques vrais: {batch_masks_true.shape}")
            print(f"     - Shape masques prédits: {batch_masks_pred.shape}")
            print(f"     - Classes uniques dans masques vrais: {np.unique(batch_masks_true)}")
            print(f"     - Classes uniques dans prédictions: {np.unique(batch_masks_pred)}")
            
            # Compter les pixels par classe
            for cls_id in range(5):
                true_pixels = np.sum(batch_masks_true == cls_id)
                pred_pixels = np.sum(batch_masks_pred == cls_id)
                print(f"     - Classe {cls_id}: {true_pixels} pixels vrais, {pred_pixels} pixels prédits")
            
            debug_printed = True
        
        # Accumuler pour la matrice de confusion globale
        all_y_true.extend(batch_masks_true.flatten())
        all_y_pred.extend(batch_masks_pred.flatten())
        
        # Calculer les métriques pour ce batch
        for i in range(len(batch_images)):
            # IoU pour cette image
            iou = calculate_iou_single(batch_masks_true[i], batch_masks_pred[i])
            total_iou += iou
            
            # Dice pour cette image
            dice = calculate_dice_single(batch_masks_true[i], batch_masks_pred[i])
            total_dice += dice
            
            # Pixel accuracy
            pixel_acc = calculate_pixel_accuracy_single(batch_masks_true[i], batch_masks_pred[i])
            total_pixel_acc += pixel_acc
        
        # Afficher la progression
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Batch {batch_idx + 1}/{num_batches} traité...")
    
    # Calculer les moyennes globales
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_pixel_acc = total_pixel_acc / num_samples
    
    print(f"\n✅ Évaluation terminée sur {num_samples} images")
    print(f"  - IoU moyen: {avg_iou:.4f}")
    print(f"  - Dice moyen: {avg_dice:.4f}")
    print(f"  - Pixel Accuracy: {avg_pixel_acc:.4f}")
    
    # Calculer la matrice de confusion
    conf_matrix = calculate_confusion_matrix(all_y_true, all_y_pred)
    
    # Calculer les métriques par classe
    per_class_metrics = calculate_class_metrics(all_y_true, all_y_pred)
    
    # Construire le dictionnaire de résultats
    results = {
        'global': {
            'iou': float(avg_iou),
            'dice': float(avg_dice),
            'pixel_accuracy': float(avg_pixel_acc)
        },
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'num_samples': num_samples
    }
    
    return results


# ============================================================================
# CALCUL DES MÉTRIQUES INDIVIDUELLES
# ============================================================================

def calculate_iou_single(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le Mean IoU pour une seule paire de masques
    
    L'IoU (Intersection over Union) mesure le chevauchement entre deux masques.
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        Mean IoU score entre 0 et 1
    """
    iou_per_class = []
    
    # Calculer l'IoU pour chaque classe présente
    for class_id in range(5):  # 0, 1, 2, 3, 4
        # Masques binaires pour cette classe
        true_binary = (mask_true == class_id)
        pred_binary = (mask_pred == class_id)
        
        # Intersection et Union
        intersection = np.sum(true_binary & pred_binary)
        union = np.sum(true_binary | pred_binary)
        
        # Calculer IoU seulement si la classe existe
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)
    
    # Retourner la moyenne des IoU
    return np.mean(iou_per_class) if iou_per_class else 0.0


def calculate_dice_single(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le Mean Dice coefficient pour une seule paire de masques
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        Mean Dice coefficient entre 0 et 1
    """
    dice_per_class = []
    
    # Calculer le Dice pour chaque classe présente
    for class_id in range(5):  # 0, 1, 2, 3, 4
        # Masques binaires pour cette classe
        true_binary = (mask_true == class_id)
        pred_binary = (mask_pred == class_id)
        
        # Intersection et somme des cardinalités
        intersection = np.sum(true_binary & pred_binary)
        sum_cardinality = np.sum(true_binary) + np.sum(pred_binary)
        
        # Calculer Dice seulement si la classe existe
        if sum_cardinality > 0:
            dice = (2.0 * intersection) / sum_cardinality
            dice_per_class.append(dice)
    
    # Retourner la moyenne des Dice
    return np.mean(dice_per_class) if dice_per_class else 0.0


def calculate_pixel_accuracy_single(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule la précision pixel par pixel
    
    Args:
        mask_true: Masque ground truth (H, W)
        mask_pred: Masque prédit (H, W)
        
    Returns:
        Accuracy entre 0 et 1
    """
    correct = np.sum(mask_true == mask_pred)
    total = mask_true.size
    
    return correct / total if total > 0 else 0


# ============================================================================
# MATRICE DE CONFUSION
# ============================================================================

def calculate_confusion_matrix(y_true: List, y_pred: List) -> np.ndarray:
    """
    Calcule la matrice de confusion pour la segmentation
    
    Args:
        y_true: Liste des labels vrais (tous les pixels)
        y_pred: Liste des labels prédits (tous les pixels)
        
    Returns:
        Matrice de confusion (5, 5) pour les 5 classes
    """
    # Convertir en arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculer la matrice de confusion pour les 5 classes
    conf_matrix = confusion_matrix(
        y_true, 
        y_pred, 
        labels=[0, 1, 2, 3, 4]  # Background, Longitudinal, Transversal, Crocodile, Pothole
    )
    
    return conf_matrix


# ============================================================================
# MÉTRIQUES PAR CLASSE
# ============================================================================

def calculate_class_metrics(y_true: List, y_pred: List) -> Dict:
    """
    Calcule les métriques détaillées pour chaque classe
    
    Args:
        y_true: Labels vrais
        y_pred: Labels prédits
        
    Returns:
        Dict {class_id: {'iou': X, 'dice': Y, 'precision': Z, ...}}
    """
    # Convertir en arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # Pour chaque classe (0, 1, 2, 3, 4)
    for class_id in range(5):
        # Créer des masques binaires pour cette classe
        true_binary = (y_true == class_id).astype(int)
        pred_binary = (y_pred == class_id).astype(int)
        
        # Calculer TP, FP, FN, TN
        tp = np.sum((true_binary == 1) & (pred_binary == 1))  # True Positives
        fp = np.sum((true_binary == 0) & (pred_binary == 1))  # False Positives
        fn = np.sum((true_binary == 1) & (pred_binary == 0))  # False Negatives
        tn = np.sum((true_binary == 0) & (pred_binary == 0))  # True Negatives
        
        # Calculer IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        
        # Calculer Dice
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Calculer Précision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculer Rappel
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculer F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Stocker les métriques
        metrics[class_id] = {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(true_binary))  # Nombre de pixels de cette classe
        }
    
    return metrics


if __name__ == "__main__":
    print("Module de métriques d'évaluation - VERSION CORRIGÉE")
    print("Ce module gère correctement:")
    print("  ✓ Les prédictions YOLO avec mapping de classes")
    print("  ✓ Le calcul des métriques de segmentation")
    print("  ✓ Les matrices de confusion")