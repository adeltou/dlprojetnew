"""
Interface Streamlit pour le projet de Deep Learning
Détection et Segmentation des Dommages Routiers
Master 2 HPC - 2025-2026

Version améliorée avec:
- Correction des incohérences de classes
- Support du modèle de segmentation YOLO
- Post-traitement des masques
- Meilleure visualisation
"""

import streamlit as st
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import time
from scipy import ndimage

# Configuration de la page
st.set_page_config(
    page_title="Projet Deep Learning - Dommages Routiers",
    page_icon="🛣️",
    layout="wide"
)

# Chemins des fichiers
BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "results" / "figures"
LOGS_DIR = BASE_DIR / "results" / "logs"
MODELS_DIR = BASE_DIR / "results" / "models"

# Créer le dossier models s'il n'existe pas
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ajouter le répertoire racine au path
sys.path.insert(0, str(BASE_DIR))

# Fichiers de métriques
UNET_METRICS = LOGS_DIR / "unet_100img_20260109_232107_metrics.json"
YOLO_METRICS = LOGS_DIR / "yolo_training_results.json"
HYBRID_METRICS = LOGS_DIR / "hybrid_100img_20260110_231216_metrics.json"

# Configuration - CORRIGÉE pour cohérence avec le dataset RDD2022
IMG_SIZE = (256, 256)
# Le dataset RDD2022 utilise les classes 0, 1, 2, 4 (sans la classe 3)
# Pour la segmentation, on ajoute le background comme classe 0
# Donc on remape: background=0, D00=1, D10=2, D20=3, D40=4
NUM_CLASSES = 5  # 4 types de dommages + 1 background

# Classes de dommages - CORRIGÉES
# Mapping: 0=Background, 1=D00(Longitudinale), 2=D10(Transversale), 3=D20(Crocodile), 4=D40(Nid-de-poule)
CLASS_NAMES = {
    0: "Background",
    1: "Fissure longitudinale (D00)",
    2: "Fissure transversale (D10)",
    3: "Fissure crocodile (D20)",
    4: "Nid-de-poule (D40)"
}

# Couleurs pour la visualisation (RGB) - Plus distinctives
CLASS_COLORS = {
    0: (0, 0, 0),         # Noir pour background
    1: (255, 50, 50),     # Rouge vif pour longitudinale
    2: (50, 255, 50),     # Vert vif pour transversale
    3: (50, 50, 255),     # Bleu vif pour crocodile
    4: (255, 255, 50)     # Jaune vif pour nid-de-poule
}

# Couleurs alternatives plus visibles pour l'overlay
CLASS_COLORS_OVERLAY = {
    0: (0, 0, 0),
    1: (255, 0, 0),       # Rouge pur
    2: (0, 255, 0),       # Vert pur
    3: (0, 100, 255),     # Bleu-cyan
    4: (255, 200, 0)      # Orange-jaune
}


# =============================================================================
# FONCTIONS DE CHARGEMENT DES MODÈLES
# =============================================================================

@st.cache_resource
def load_unet_model():
    """Charge le modèle U-Net"""
    try:
        from models.unet_scratch import create_unet_model
        model = create_unet_model(
            input_shape=(256, 256, 3),
            num_classes=NUM_CLASSES,
            compile_model=False
        )
        # Chercher les poids sauvegardés
        weights_path = MODELS_DIR / "unet_best.h5"
        if weights_path.exists():
            model.load_weights(str(weights_path))
            return model, True
        return model, False
    except Exception as e:
        st.warning(f"Impossible de charger U-Net: {e}")
        return None, False


@st.cache_resource
def load_hybrid_model():
    """Charge le modèle Hybride"""
    try:
        from models.hybrid_model import create_hybrid_model
        model = create_hybrid_model(
            input_shape=(256, 256, 3),
            num_classes=NUM_CLASSES,
            compile_model=False
        )
        # Chercher les poids sauvegardés
        weights_path = MODELS_DIR / "hybrid_best.h5"
        if weights_path.exists():
            model.load_weights(str(weights_path))
            return model, True
        return model, False
    except Exception as e:
        st.warning(f"Impossible de charger Hybrid: {e}")
        return None, False


@st.cache_resource
def load_yolo_model():
    """Charge le modèle YOLO - Préférence pour le modèle de segmentation"""
    try:
        from ultralytics import YOLO

        # Priorité 1: Modèle YOLO entraîné sur RDD2022
        trained_path = MODELS_DIR / "yolo_best.pt"
        if trained_path.exists():
            model = YOLO(str(trained_path))
            return model, True, "trained"

        # Priorité 2: Modèle de segmentation pré-entraîné (meilleur pour les masques)
        seg_path = BASE_DIR / "yolov8n-seg.pt"
        if seg_path.exists():
            model = YOLO(str(seg_path))
            return model, False, "segmentation"

        # Priorité 3: Modèle de détection pré-entraîné
        pretrained_path = BASE_DIR / "yolov8n.pt"
        if pretrained_path.exists():
            model = YOLO(str(pretrained_path))
            return model, False, "detection"

        return None, False, None
    except Exception as e:
        st.warning(f"Impossible de charger YOLO: {e}")
        return None, False, None


# =============================================================================
# FONCTIONS DE PRÉDICTION
# =============================================================================

def preprocess_image(image, target_size=IMG_SIZE):
    """Prétraite l'image pour les modèles"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convertir en RGB si nécessaire
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Redimensionner
    image_resized = cv2.resize(image, target_size)

    # Normaliser
    image_normalized = image_resized.astype(np.float32) / 255.0

    return image_resized, image_normalized


def predict_unet(model, image_normalized):
    """Effectue une prédiction avec U-Net"""
    # Ajouter la dimension batch
    input_tensor = np.expand_dims(image_normalized, axis=0)

    # Prédiction
    start_time = time.time()
    prediction = model.predict(input_tensor, verbose=0)
    inference_time = time.time() - start_time

    # Récupérer le masque de segmentation
    mask = np.argmax(prediction[0], axis=-1)
    confidence = np.max(prediction[0], axis=-1)

    return mask, confidence, inference_time


def predict_hybrid(model, image_normalized):
    """Effectue une prédiction avec le modèle Hybride"""
    # Ajouter la dimension batch
    input_tensor = np.expand_dims(image_normalized, axis=0)

    # Prédiction
    start_time = time.time()
    prediction = model.predict(input_tensor, verbose=0)
    inference_time = time.time() - start_time

    # Récupérer le masque de segmentation
    mask = np.argmax(prediction[0], axis=-1)
    confidence = np.max(prediction[0], axis=-1)

    return mask, confidence, inference_time


def predict_yolo(model, image, model_type="detection"):
    """
    Effectue une prédiction avec YOLO

    Args:
        model: Modèle YOLO
        image: Image à analyser
        model_type: "segmentation", "detection" ou "trained"
    """
    start_time = time.time()

    # Prédiction avec YOLO
    results = model.predict(source=image, conf=0.25, verbose=False)
    inference_time = time.time() - start_time

    # Initialiser le masque et la carte de confiance
    mask = np.zeros(IMG_SIZE, dtype=np.uint8)
    confidence_map = np.zeros(IMG_SIZE, dtype=np.float32)

    for result in results:
        # Vérifier si c'est un modèle de segmentation avec des masques
        if hasattr(result, 'masks') and result.masks is not None:
            # Modèle de segmentation - extraire les masques
            masks_data = result.masks.data.cpu().numpy()
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for i, (seg_mask, cls, conf) in enumerate(zip(masks_data, classes, confs)):
                    # Redimensionner le masque à la taille cible
                    seg_mask_resized = cv2.resize(seg_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

                    # Pour les modèles pré-entraînés COCO, on simule la détection de dommages
                    # En production, utiliser un modèle entraîné sur RDD2022
                    if model_type in ["segmentation", "detection"]:
                        # Simulation: mapper certaines classes COCO à des dommages
                        # Ceci est temporaire - idéalement utiliser un modèle entraîné
                        damage_class = simulate_damage_class_from_coco(cls, seg_mask_resized)
                    else:
                        # Modèle entraîné: utiliser le mapping direct
                        damage_class = cls + 1 if cls < 4 else 4

                    # Appliquer le masque là où la valeur est > 0.5
                    mask_binary = seg_mask_resized > 0.5
                    mask[mask_binary] = damage_class
                    confidence_map[mask_binary] = conf

        # Si pas de masques, utiliser les bounding boxes
        elif result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            h, w = IMG_SIZE
            for (x1, y1, x2, y2), cls, conf in zip(xyxyn, classes, confs):
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)

                # Assurer que les coordonnées sont valides
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)

                if model_type in ["segmentation", "detection"]:
                    # Modèle pré-entraîné: simulation
                    damage_class = (cls % 4) + 1  # Classes 1-4
                else:
                    # Modèle entraîné: mapping direct
                    damage_class = cls + 1 if cls < 4 else 4

                # Créer un masque elliptique au lieu d'un rectangle
                # pour un résultat plus réaliste
                center_x = (px1 + px2) // 2
                center_y = (py1 + py2) // 2
                axis_x = (px2 - px1) // 2
                axis_y = (py2 - py1) // 2

                if axis_x > 0 and axis_y > 0:
                    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y),
                               0, 0, 360, int(damage_class), -1)
                    cv2.ellipse(confidence_map, (center_x, center_y), (axis_x, axis_y),
                               0, 0, 360, conf, -1)

    return mask, confidence_map, inference_time


def simulate_damage_class_from_coco(coco_class, mask):
    """
    Simule une classe de dommage à partir d'une détection COCO
    Note: Cette fonction est utilisée uniquement pour la démonstration
    avec des modèles pré-entraînés non entraînés sur RDD2022
    """
    # Mapping basé sur les caractéristiques de forme du masque
    if mask is not None and np.any(mask > 0):
        # Calculer le ratio largeur/hauteur
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            height = coords[0].max() - coords[0].min() + 1
            width = coords[1].max() - coords[1].min() + 1
            aspect_ratio = width / max(height, 1)

            # Classification basée sur la forme
            if aspect_ratio > 3:
                return 1  # Fissure longitudinale (allongée horizontalement)
            elif aspect_ratio < 0.3:
                return 2  # Fissure transversale (allongée verticalement)
            elif np.sum(mask > 0) > 1000:
                return 4  # Nid-de-poule (grande surface)
            else:
                return 3  # Fissure crocodile (forme complexe)

    # Par défaut, rotation entre les classes
    return (coco_class % 4) + 1


def mask_to_colored_image(mask, use_overlay_colors=False):
    """Convertit un masque en image colorée"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    colors = CLASS_COLORS_OVERLAY if use_overlay_colors else CLASS_COLORS

    for class_id, color in colors.items():
        colored[mask == class_id] = color

    return colored


def overlay_mask_on_image(image, mask, alpha=0.6):
    """Superpose le masque coloré sur l'image originale avec meilleure visibilité"""
    colored_mask = mask_to_colored_image(mask, use_overlay_colors=True)

    # Créer l'overlay
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)

    # Garder l'image originale là où il n'y a pas de détection
    detection_mask = mask > 0
    result = image.copy()
    result[detection_mask] = overlay[detection_mask]

    # Ajouter des contours pour mieux visualiser les zones détectées
    for class_id in range(1, NUM_CLASSES):
        class_mask = (mask == class_id).astype(np.uint8)
        if np.any(class_mask):
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = CLASS_COLORS_OVERLAY[class_id]
            cv2.drawContours(result, contours, -1, color, 2)

    return result


def postprocess_mask(mask, min_area=50, apply_morphology=True):
    """
    Post-traitement du masque pour améliorer la qualité

    Args:
        mask: Masque de segmentation
        min_area: Surface minimale pour garder une région
        apply_morphology: Appliquer des opérations morphologiques
    """
    processed_mask = mask.copy()

    if apply_morphology:
        # Appliquer des opérations morphologiques par classe
        for class_id in range(1, NUM_CLASSES):
            class_mask = (mask == class_id).astype(np.uint8)

            if np.any(class_mask):
                # Fermeture pour combler les petits trous
                kernel = np.ones((3, 3), np.uint8)
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

                # Ouverture pour enlever le bruit
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

                # Supprimer les petites régions
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < min_area:
                        class_mask[labels == i] = 0

                # Mettre à jour le masque
                processed_mask[class_mask > 0] = class_id

    return processed_mask


def enhance_detection_with_edge_analysis(image, mask):
    """
    Améliore la détection en utilisant l'analyse des contours de l'image

    Args:
        image: Image originale (RGB)
        mask: Masque prédit par le modèle
    """
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Détection de contours avec Canny
    edges = cv2.Canny(gray, 50, 150)

    # Dilater les contours
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Combiner avec le masque existant
    enhanced_mask = mask.copy()

    # Pour les zones où il y a des contours forts mais pas de détection,
    # on peut suggérer une détection potentielle (avec précaution)
    # Note: Ceci est une heuristique, pas une vraie amélioration de modèle

    return enhanced_mask


def create_detection_summary(mask):
    """
    Crée un résumé des détections dans le masque

    Returns:
        dict: Résumé des détections par classe
    """
    summary = {}

    for class_id in range(1, NUM_CLASSES):
        class_mask = (mask == class_id).astype(np.uint8)
        pixel_count = np.sum(class_mask)

        if pixel_count > 0:
            # Trouver les composantes connexes
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )

            regions = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                centroid = centroids[i]
                regions.append({
                    'area': area,
                    'bbox': (x, y, w, h),
                    'centroid': centroid
                })

            summary[class_id] = {
                'class_name': CLASS_NAMES[class_id],
                'total_pixels': pixel_count,
                'percentage': (pixel_count / mask.size) * 100,
                'num_regions': num_labels - 1,
                'regions': regions
            }

    return summary


def heuristic_crack_detection(image):
    """
    Détection heuristique de fissures basée sur le traitement d'image
    Cette méthode utilise des techniques classiques de vision par ordinateur
    pour détecter les potentiels dommages routiers.

    Args:
        image: Image RGB (H, W, 3)

    Returns:
        mask: Masque de détection (H, W) avec les classes
        confidence: Carte de confiance
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    confidence = np.zeros((h, w), dtype=np.float32)

    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. Améliorer le contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Détection de contours avec Canny multi-échelle
    edges_low = cv2.Canny(enhanced, 30, 100)
    edges_high = cv2.Canny(enhanced, 50, 150)
    edges = cv2.bitwise_or(edges_low, edges_high)

    # 3. Détection des zones sombres (potentielles fissures)
    # Les fissures apparaissent souvent comme des zones plus sombres
    mean_val = np.mean(gray)
    dark_threshold = mean_val * 0.7
    dark_regions = (gray < dark_threshold).astype(np.uint8) * 255

    # 4. Combiner les détections
    combined = cv2.bitwise_or(edges, dark_regions)

    # 5. Opérations morphologiques
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)

    # Fermeture pour connecter les fissures proches
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small)

    # Ouverture pour enlever le bruit
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)

    # 6. Analyse des composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )

    # 7. Classifier les régions détectées
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w_box, h_box = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # Filtrer les petites régions (bruit)
        if area < 50:
            continue

        # Calculer les caractéristiques de la région
        region_mask = (labels == i).astype(np.uint8)
        aspect_ratio = w_box / max(h_box, 1)

        # Classifier selon la forme
        if area > 2000 and aspect_ratio > 0.5 and aspect_ratio < 2:
            # Grande zone compacte -> Nid-de-poule
            class_id = 4
            conf = 0.4
        elif aspect_ratio > 3:
            # Très allongé horizontalement -> Fissure longitudinale
            class_id = 1
            conf = 0.5
        elif aspect_ratio < 0.33:
            # Très allongé verticalement -> Fissure transversale
            class_id = 2
            conf = 0.5
        elif area > 500:
            # Zone de taille moyenne avec forme complexe -> Crocodile
            class_id = 3
            conf = 0.3
        else:
            # Petite fissure par défaut
            class_id = 1
            conf = 0.25

        # Appliquer la détection
        mask[region_mask > 0] = class_id
        confidence[region_mask > 0] = conf

    return mask, confidence


def detect_with_heuristics_and_model(image, model_mask, model_confidence, model_trained):
    """
    Combine la détection heuristique avec celle du modèle

    Args:
        image: Image originale
        model_mask: Masque prédit par le modèle
        model_confidence: Confiance du modèle
        model_trained: Si le modèle est entraîné

    Returns:
        combined_mask, combined_confidence
    """
    if model_trained:
        # Si le modèle est entraîné, lui faire confiance
        return model_mask, model_confidence

    # Sinon, combiner avec la détection heuristique
    heuristic_mask, heuristic_conf = heuristic_crack_detection(image)

    # Stratégie de combinaison:
    # - Si le modèle détecte quelque chose avec confiance > 0.5, utiliser
    # - Sinon, utiliser la détection heuristique
    combined_mask = model_mask.copy()
    combined_conf = model_confidence.copy()

    # Ajouter les détections heuristiques là où le modèle n'a rien détecté
    no_detection = (model_mask == 0)
    combined_mask[no_detection] = heuristic_mask[no_detection]
    combined_conf[no_detection] = heuristic_conf[no_detection]

    return combined_mask, combined_conf


# =============================================================================
# FONCTIONS D'AFFICHAGE DES MÉTRIQUES
# =============================================================================

def load_metrics():
    """Charge les métriques des trois architectures"""
    metrics = {}

    # U-Net
    if UNET_METRICS.exists():
        with open(UNET_METRICS, 'r') as f:
            unet_data = json.load(f)
            metrics['U-Net'] = {
                'epochs': [d['epoch'] for d in unet_data],
                'accuracy': [d['accuracy'] for d in unet_data],
                'dice_coefficient': [d['dice_coefficient'] for d in unet_data],
                'iou': [d['iou'] for d in unet_data],
                'loss': [d['loss'] for d in unet_data],
                'val_accuracy': [d['val_accuracy'] for d in unet_data],
                'val_dice_coefficient': [d['val_dice_coefficient'] for d in unet_data],
                'val_iou': [d['val_iou'] for d in unet_data],
                'val_loss': [d['val_loss'] for d in unet_data]
            }

    # YOLO
    if YOLO_METRICS.exists():
        with open(YOLO_METRICS, 'r') as f:
            yolo_data = json.load(f)
            history = yolo_data['history']
            metrics['YOLO'] = {
                'epochs': list(range(1, len(history['accuracy']) + 1)),
                'accuracy': history['accuracy'],
                'dice_coefficient': history['dice_coefficient'],
                'iou': history['iou'],
                'loss': history['loss'],
                'val_accuracy': history['val_accuracy'],
                'val_dice_coefficient': history['val_dice_coefficient'],
                'val_iou': history['val_iou'],
                'val_loss': history['val_loss']
            }

    # Hybrid
    if HYBRID_METRICS.exists():
        with open(HYBRID_METRICS, 'r') as f:
            hybrid_data = json.load(f)
            metrics['Hybrid'] = {
                'epochs': [d['epoch'] for d in hybrid_data],
                'accuracy': [d['accuracy'] for d in hybrid_data],
                'dice_coefficient': [d['dice_coefficient'] for d in hybrid_data],
                'iou': [d['iou'] for d in hybrid_data],
                'loss': [d['loss'] for d in hybrid_data],
                'val_accuracy': [d['val_accuracy'] for d in hybrid_data],
                'val_dice_coefficient': [d['val_dice_coefficient'] for d in hybrid_data],
                'val_iou': [d['val_iou'] for d in hybrid_data],
                'val_loss': [d['val_loss'] for d in hybrid_data]
            }

    return metrics


def calculate_global_averages(metrics):
    """Calcule les moyennes globales pour chaque métrique et architecture"""
    averages = {}

    metric_names = ['accuracy', 'dice_coefficient', 'iou', 'loss',
                    'val_accuracy', 'val_dice_coefficient', 'val_iou', 'val_loss']

    for arch_name, arch_metrics in metrics.items():
        averages[arch_name] = {}
        for metric in metric_names:
            if metric in arch_metrics:
                averages[arch_name][metric] = np.mean(arch_metrics[metric])

    return averages


# =============================================================================
# SECTIONS DE L'INTERFACE
# =============================================================================

def display_image_analysis():
    """Section d'analyse d'image avec les 3 modèles"""
    st.header("Analyse d'Image - Détection des Dommages Routiers")

    st.markdown("""
    Chargez une photo de route pour analyser les dommages avec nos 3 architectures de deep learning:
    - **U-Net**: Architecture de segmentation sémantique
    - **YOLO**: Détection d'objets avec segmentation
    - **Hybride**: Combinaison U-Net + YOLO avec attention gates
    """)

    # Options avancées
    with st.expander("Options avancées"):
        use_heuristic = st.checkbox(
            "Activer la détection heuristique",
            value=True,
            help="Combine la détection par deep learning avec des méthodes de traitement d'image classiques. Utile quand les modèles ne sont pas entraînés."
        )
        apply_postprocess = st.checkbox(
            "Appliquer le post-traitement",
            value=True,
            help="Applique des opérations morphologiques pour nettoyer les masques."
        )
        overlay_alpha = st.slider(
            "Transparence de l'overlay",
            min_value=0.1, max_value=0.9, value=0.6,
            help="Contrôle la transparence du masque sur l'image originale."
        )

    # Upload de l'image
    uploaded_file = st.file_uploader(
        "Choisir une image de route",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Formats supportés: PNG, JPG, JPEG, BMP"
    )

    if uploaded_file is not None:
        # Charger l'image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Prétraiter l'image
        image_resized, image_normalized = preprocess_image(image_np)

        # Afficher l'image originale
        st.subheader("Image Chargée")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image_resized, caption=f"Image redimensionnée ({IMG_SIZE[0]}x{IMG_SIZE[1]})", use_container_width=True)
        with col2:
            st.info(f"""
            **Informations:**
            - Taille originale: {image_np.shape[1]}x{image_np.shape[0]}
            - Taille traitée: {IMG_SIZE[0]}x{IMG_SIZE[1]}
            - Canaux: {image_np.shape[2] if len(image_np.shape) > 2 else 1}
            """)

        st.divider()

        # Bouton d'analyse
        if st.button("Analyser l'image avec les 3 modèles", type="primary"):
            st.subheader("Résultats de l'Analyse")

            results = {}

            # Créer 3 colonnes pour les résultats
            col1, col2, col3 = st.columns(3)

            # === U-NET ===
            with col1:
                st.markdown("### U-Net")
                with st.spinner("Chargement de U-Net..."):
                    unet_model, unet_trained = load_unet_model()

                if unet_model is not None:
                    status = "Entraîné" if unet_trained else "Non entraîné (poids aléatoires)"
                    st.caption(f"Statut: {status}")

                    if not unet_trained:
                        st.warning("Modèle non entraîné. Détection heuristique activée.")

                    with st.spinner("Analyse en cours..."):
                        mask, confidence, inf_time = predict_unet(unet_model, image_normalized)

                        # Combiner avec la détection heuristique si activée
                        if use_heuristic and not unet_trained:
                            mask, confidence = detect_with_heuristics_and_model(
                                image_resized, mask, confidence, unet_trained
                            )

                        # Post-traitement si activé
                        if apply_postprocess:
                            mask = postprocess_mask(mask, min_area=30)

                    results['U-Net'] = {
                        'mask': mask,
                        'confidence': confidence,
                        'time': inf_time,
                        'trained': unet_trained
                    }

                    # Afficher le masque
                    overlay = overlay_mask_on_image(image_resized, mask, alpha=overlay_alpha)
                    st.image(overlay, caption="Segmentation U-Net", use_container_width=True)

                    # Masque seul
                    colored_mask = mask_to_colored_image(mask)
                    st.image(colored_mask, caption="Masque de segmentation", use_container_width=True)

                    st.metric("Temps d'inférence", f"{inf_time*1000:.1f} ms")
                else:
                    st.error("U-Net non disponible")

            # === YOLO ===
            with col2:
                st.markdown("### YOLO")
                with st.spinner("Chargement de YOLO..."):
                    yolo_result = load_yolo_model()
                    if len(yolo_result) == 3:
                        yolo_model, yolo_trained, yolo_type = yolo_result
                    else:
                        yolo_model, yolo_trained = yolo_result
                        yolo_type = "detection"

                if yolo_model is not None:
                    if yolo_trained:
                        status = "Entraîné sur RDD2022"
                    elif yolo_type == "segmentation":
                        status = "Segmentation (COCO) - Simulation"
                    else:
                        status = "Détection (COCO) - Simulation"
                    st.caption(f"Statut: {status}")

                    if not yolo_trained:
                        st.warning("Modèle non entraîné. Détection heuristique activée.")

                    with st.spinner("Analyse en cours..."):
                        mask, confidence, inf_time = predict_yolo(yolo_model, image_resized, yolo_type)

                        # Combiner avec la détection heuristique si activée
                        if use_heuristic and not yolo_trained:
                            mask, confidence = detect_with_heuristics_and_model(
                                image_resized, mask, confidence, yolo_trained
                            )

                        # Post-traitement si activé
                        if apply_postprocess:
                            mask = postprocess_mask(mask, min_area=30)

                    results['YOLO'] = {
                        'mask': mask,
                        'confidence': confidence,
                        'time': inf_time,
                        'trained': yolo_trained
                    }

                    # Afficher le masque
                    overlay = overlay_mask_on_image(image_resized, mask, alpha=overlay_alpha)
                    st.image(overlay, caption="Détection YOLO", use_container_width=True)

                    # Masque seul
                    colored_mask = mask_to_colored_image(mask)
                    st.image(colored_mask, caption="Masque de détection", use_container_width=True)

                    st.metric("Temps d'inférence", f"{inf_time*1000:.1f} ms")
                else:
                    st.error("YOLO non disponible")

            # === HYBRIDE ===
            with col3:
                st.markdown("### Hybride")
                with st.spinner("Chargement du modèle Hybride..."):
                    hybrid_model, hybrid_trained = load_hybrid_model()

                if hybrid_model is not None:
                    status = "Entraîné" if hybrid_trained else "Non entraîné (poids aléatoires)"
                    st.caption(f"Statut: {status}")

                    if not hybrid_trained:
                        st.warning("Modèle non entraîné. Détection heuristique activée.")

                    with st.spinner("Analyse en cours..."):
                        mask, confidence, inf_time = predict_hybrid(hybrid_model, image_normalized)

                        # Combiner avec la détection heuristique si activée
                        if use_heuristic and not hybrid_trained:
                            mask, confidence = detect_with_heuristics_and_model(
                                image_resized, mask, confidence, hybrid_trained
                            )

                        # Post-traitement si activé
                        if apply_postprocess:
                            mask = postprocess_mask(mask, min_area=30)

                    results['Hybrid'] = {
                        'mask': mask,
                        'confidence': confidence,
                        'time': inf_time,
                        'trained': hybrid_trained
                    }

                    # Afficher le masque
                    overlay = overlay_mask_on_image(image_resized, mask, alpha=overlay_alpha)
                    st.image(overlay, caption="Segmentation Hybride", use_container_width=True)

                    # Masque seul
                    colored_mask = mask_to_colored_image(mask)
                    st.image(colored_mask, caption="Masque de segmentation", use_container_width=True)

                    st.metric("Temps d'inférence", f"{inf_time*1000:.1f} ms")
                else:
                    st.error("Modèle Hybride non disponible")

            # === COMPARAISON DES PERFORMANCES ===
            if results:
                st.divider()
                st.subheader("Comparaison des Performances")

                # Tableau comparatif
                comparison_data = []
                for model_name, res in results.items():
                    # Compter les pixels par classe
                    unique, counts = np.unique(res['mask'], return_counts=True)
                    class_counts = dict(zip(unique, counts))

                    # Calculer le pourcentage de détection
                    total_pixels = res['mask'].size
                    detected_pixels = total_pixels - class_counts.get(0, 0)
                    detection_rate = (detected_pixels / total_pixels) * 100

                    comparison_data.append({
                        'Modèle': model_name,
                        'Temps (ms)': f"{res['time']*1000:.1f}",
                        'Pixels détectés': f"{detected_pixels:,}",
                        'Taux de détection': f"{detection_rate:.2f}%",
                        'Confiance moyenne': f"{np.mean(res['confidence']):.3f}",
                        'Entraîné': "Oui" if res['trained'] else "Non"
                    })

                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)

                # Graphique comparatif des temps d'inférence
                fig, ax = plt.subplots(figsize=(8, 4))
                models = [d['Modèle'] for d in comparison_data]
                times = [float(d['Temps (ms)']) for d in comparison_data]
                colors = ['#2ecc71', '#e74c3c', '#3498db']

                bars = ax.bar(models, times, color=colors[:len(models)])
                ax.set_ylabel('Temps (ms)')
                ax.set_title('Comparaison des Temps d\'Inférence')

                for bar, t in zip(bars, times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{t:.1f}', ha='center', va='bottom')

                st.pyplot(fig)
                plt.close()

                # Légende des couleurs
                st.subheader("Légende des Classes")
                legend_cols = st.columns(5)
                for idx, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
                    color = CLASS_COLORS_OVERLAY[class_id]
                    with legend_cols[idx]:
                        color_box = f'<div style="background-color: rgb{color}; width: 30px; height: 30px; display: inline-block; border: 1px solid black;"></div>'
                        st.markdown(f"{color_box} {class_name}", unsafe_allow_html=True)

                # Détails des détections par modèle
                st.subheader("Détails des Détections")
                for model_name, res in results.items():
                    with st.expander(f"Détails - {model_name}"):
                        summary = create_detection_summary(res['mask'])
                        if summary:
                            for class_id, info in summary.items():
                                st.markdown(f"**{info['class_name']}**")
                                st.write(f"- Pixels détectés: {info['total_pixels']:,}")
                                st.write(f"- Pourcentage de l'image: {info['percentage']:.2f}%")
                                st.write(f"- Nombre de régions: {info['num_regions']}")
                        else:
                            st.info("Aucune détection pour ce modèle.")

                # Avertissement si les modèles ne sont pas entraînés
                untrained_models = [name for name, res in results.items() if not res['trained']]
                if untrained_models:
                    st.divider()
                    st.subheader("Comment améliorer les résultats")
                    st.warning(f"""
                    **Les modèles suivants ne sont pas entraînés:** {', '.join(untrained_models)}

                    Les résultats actuels sont générés avec des poids aléatoires (U-Net, Hybrid)
                    ou un modèle pré-entraîné sur COCO (YOLO) qui ne reconnaît pas les dommages routiers.

                    **Pour obtenir de vrais résultats, vous devez entraîner les modèles:**
                    """)

                    st.code("""
# Pour entraîner U-Net:
python training/train_unet.py

# Pour entraîner le modèle Hybride:
python training/train_hybrid.py

# Pour entraîner YOLO sur RDD2022:
python training/train_yolo.py
                    """, language="bash")

                    st.info("""
                    **Après l'entraînement**, les poids seront sauvegardés dans `results/models/`
                    et seront automatiquement chargés par l'interface.
                    """)


def display_preprocessing_results():
    """Affiche les résultats de prétraitement"""
    st.header("Résultats du Prétraitement")

    st.markdown("""
    Cette section présente les visualisations issues de la phase de prétraitement des données
    du dataset RDD2022 (Road Damage Detection).
    """)

    # Distribution des classes
    st.subheader("Distribution des Classes")

    col1, col2, col3 = st.columns(3)

    train_dist = FIGURES_DIR / "class_distribution_train.png"
    val_dist = FIGURES_DIR / "class_distribution_val.png"
    test_dist = FIGURES_DIR / "class_distribution_test.png"

    with col1:
        if train_dist.exists():
            st.image(str(train_dist), caption="Distribution - Ensemble d'entraînement", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col2:
        if val_dist.exists():
            st.image(str(val_dist), caption="Distribution - Ensemble de validation", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col3:
        if test_dist.exists():
            st.image(str(test_dist), caption="Distribution - Ensemble de test", use_container_width=True)
        else:
            st.warning("Image non disponible")

    # Augmentation et échantillons
    st.subheader("Augmentation des Données et Échantillons")

    col1, col2 = st.columns(2)

    with col1:
        augmentation_img = FIGURES_DIR / "test_augmentation.png"
        if augmentation_img.exists():
            st.image(str(augmentation_img), caption="Exemples d'augmentation de données", use_container_width=True)
        else:
            st.warning("Image d'augmentation non disponible")

    with col2:
        sample_img = FIGURES_DIR / "test_sample_original.png"
        if sample_img.exists():
            st.image(str(sample_img), caption="Échantillon original", use_container_width=True)
        else:
            st.warning("Image d'échantillon non disponible")


def display_metrics_histograms(metrics, averages):
    """Affiche les histogrammes des métriques"""
    st.header("Comparaison des Métriques - Histogrammes")

    architectures = list(averages.keys())
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    # Métriques principales pour l'histogramme de comparaison
    main_metrics = ['accuracy', 'dice_coefficient', 'iou']
    val_metrics = ['val_accuracy', 'val_dice_coefficient', 'val_iou']

    # Tableau récapitulatif des moyennes globales
    st.subheader("Moyennes Globales des Métriques")

    # Créer un DataFrame pour l'affichage
    df_data = []
    for arch in architectures:
        row = {'Architecture': arch}
        for metric in main_metrics + val_metrics + ['loss', 'val_loss']:
            if metric in averages[arch]:
                row[metric.replace('_', ' ').title()] = f"{averages[arch][metric]:.4f}"
        df_data.append(row)

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Histogrammes des métriques d'entraînement
    st.subheader("Métriques d'Entraînement (Moyennes)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(main_metrics):
        ax = axes[idx]
        values = [averages[arch][metric] for arch in architectures]
        bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur')
        ax.set_ylim(0, 1)

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Histogrammes des métriques de validation
    st.subheader("Métriques de Validation (Moyennes)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(val_metrics):
        ax = axes[idx]
        values = [averages[arch][metric] for arch in architectures]
        bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
        ax.set_title(metric.replace('val_', '').replace('_', ' ').title() + ' (Validation)',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur')
        ax.set_ylim(0, 1)

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Comparaison des pertes (Loss)
    st.subheader("Comparaison des Pertes (Loss)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss d'entraînement
    ax = axes[0]
    values = [averages[arch]['loss'] for arch in architectures]
    bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
    ax.set_title('Loss Entraînement (Moyenne)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Loss de validation
    ax = axes[1]
    values = [averages[arch]['val_loss'] for arch in architectures]
    bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
    ax.set_title('Loss Validation (Moyenne)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_training_evolution(metrics):
    """Affiche l'évolution des métriques pendant l'entraînement"""
    st.header("Évolution des Métriques pendant l'Entraînement")

    architectures = list(metrics.keys())
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    # Sélection de la métrique à visualiser
    metric_options = {
        'Accuracy': 'accuracy',
        'Dice Coefficient': 'dice_coefficient',
        'IoU': 'iou',
        'Loss': 'loss',
        'Validation Accuracy': 'val_accuracy',
        'Validation Dice': 'val_dice_coefficient',
        'Validation IoU': 'val_iou',
        'Validation Loss': 'val_loss'
    }

    selected_metric_name = st.selectbox("Sélectionner la métrique", list(metric_options.keys()))
    selected_metric = metric_options[selected_metric_name]

    fig, ax = plt.subplots(figsize=(12, 6))

    for arch in architectures:
        if selected_metric in metrics[arch]:
            epochs = metrics[arch]['epochs']
            values = metrics[arch][selected_metric]
            ax.plot(epochs, values, marker='o', label=arch, color=colors[arch], linewidth=2)

    ax.set_xlabel('Époque', fontsize=12)
    ax.set_ylabel(selected_metric_name, fontsize=12)
    ax.set_title(f'Évolution de {selected_metric_name} par Époque', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_detailed_metrics(metrics):
    """Affiche les métriques détaillées par époque"""
    st.header("Métriques Détaillées par Architecture")

    architecture = st.selectbox("Sélectionner l'architecture", list(metrics.keys()))

    if architecture in metrics:
        arch_metrics = metrics[architecture]

        # Créer un DataFrame avec toutes les métriques
        df_data = {
            'Époque': arch_metrics['epochs'],
            'Accuracy': [f"{v:.4f}" for v in arch_metrics['accuracy']],
            'Dice': [f"{v:.4f}" for v in arch_metrics['dice_coefficient']],
            'IoU': [f"{v:.4f}" for v in arch_metrics['iou']],
            'Loss': [f"{v:.4f}" for v in arch_metrics['loss']],
            'Val Accuracy': [f"{v:.4f}" for v in arch_metrics['val_accuracy']],
            'Val Dice': [f"{v:.4f}" for v in arch_metrics['val_dice_coefficient']],
            'Val IoU': [f"{v:.4f}" for v in arch_metrics['val_iou']],
            'Val Loss': [f"{v:.4f}" for v in arch_metrics['val_loss']]
        }

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def display_final_comparison(averages):
    """Affiche une comparaison finale des architectures"""
    st.header("Synthèse et Comparaison Finale")

    architectures = list(averages.keys())

    # Radar chart pour comparer les architectures
    st.subheader("Comparaison Radar des Performances")

    metrics_to_compare = ['accuracy', 'dice_coefficient', 'iou',
                          'val_accuracy', 'val_dice_coefficient', 'val_iou']
    labels = ['Accuracy', 'Dice', 'IoU', 'Val Acc', 'Val Dice', 'Val IoU']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le polygone

    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    for arch in architectures:
        values = [averages[arch][m] for m in metrics_to_compare]
        values += values[:1]  # Fermer le polygone
        ax.plot(angles, values, 'o-', linewidth=2, label=arch, color=colors[arch])
        ax.fill(angles, values, alpha=0.25, color=colors[arch])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Comparaison des Performances', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Meilleure architecture par métrique
    st.subheader("Meilleure Architecture par Métrique")

    best_results = []
    for metric in metrics_to_compare + ['loss', 'val_loss']:
        best_arch = None
        best_val = None

        for arch in architectures:
            val = averages[arch][metric]
            if best_val is None:
                best_val = val
                best_arch = arch
            elif 'loss' in metric:  # Pour loss, on veut le minimum
                if val < best_val:
                    best_val = val
                    best_arch = arch
            else:  # Pour les autres métriques, on veut le maximum
                if val > best_val:
                    best_val = val
                    best_arch = arch

        best_results.append({
            'Métrique': metric.replace('_', ' ').title(),
            'Meilleure Architecture': best_arch,
            'Valeur': f"{best_val:.4f}"
        })

    df = pd.DataFrame(best_results)
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale de l'application"""

    # Titre principal
    st.title("Projet Deep Learning - Détection des Dommages Routiers")
    st.markdown("""
    **Master 2 HPC - Année 2025-2026**
    Segmentation sémantique des dégradations routières avec U-Net, YOLO et architecture Hybride.
    """)

    st.divider()

    # Menu de navigation
    menu = st.sidebar.radio(
        "Navigation",
        ["Analyse d'Image", "Prétraitement", "Histogrammes des Métriques",
         "Évolution de l'Entraînement", "Métriques Détaillées", "Synthèse Finale"]
    )

    # Charger les métriques
    metrics = load_metrics()

    # Calculer les moyennes globales si des métriques sont disponibles
    averages = calculate_global_averages(metrics) if metrics else {}

    # Afficher la section sélectionnée
    if menu == "Analyse d'Image":
        display_image_analysis()

    elif menu == "Prétraitement":
        display_preprocessing_results()

    elif menu == "Histogrammes des Métriques":
        if metrics:
            display_metrics_histograms(metrics, averages)
        else:
            st.error("Aucune métrique trouvée. Vérifiez que les fichiers JSON sont présents dans results/logs/")

    elif menu == "Évolution de l'Entraînement":
        if metrics:
            display_training_evolution(metrics)
        else:
            st.error("Aucune métrique trouvée.")

    elif menu == "Métriques Détaillées":
        if metrics:
            display_detailed_metrics(metrics)
        else:
            st.error("Aucune métrique trouvée.")

    elif menu == "Synthèse Finale":
        if averages:
            display_final_comparison(averages)
        else:
            st.error("Aucune métrique trouvée.")

    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### Types de dommages
    - **Classe 1**: Fissures longitudinales
    - **Classe 2**: Fissures transversales
    - **Classe 3**: Fissures crocodiles
    - **Classe 4**: Nids-de-poule
    """)

    st.sidebar.markdown("---")
    st.sidebar.info("Projet de Deep Learning - M2 HPC")


if __name__ == "__main__":
    main()