"""
Interface Streamlit pour le projet de Deep Learning
Détection et Segmentation des Dommages Routiers
Master 2 HPC - 2025-2026
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

# Ajouter le répertoire racine au path
sys.path.insert(0, str(BASE_DIR))

# Fichiers de métriques
UNET_METRICS = LOGS_DIR / "unet_100img_20260109_232107_metrics.json"
YOLO_METRICS = LOGS_DIR / "yolo_training_results.json"
HYBRID_METRICS = LOGS_DIR / "hybrid_100img_20260110_231216_metrics.json"

# Configuration
IMG_SIZE = (256, 256)
NUM_CLASSES = 5  # 4 classes + background

# Classes de dommages
CLASS_NAMES = {
    0: "Background",
    1: "Fissure longitudinale",
    2: "Fissure transversale",
    3: "Fissure crocodile",
    4: "Nid-de-poule"
}

# Couleurs pour la visualisation (RGB)
CLASS_COLORS = {
    0: (0, 0, 0),         # Noir pour background
    1: (255, 0, 0),       # Rouge pour longitudinale
    2: (0, 255, 0),       # Vert pour transversale
    3: (0, 0, 255),       # Bleu pour crocodile
    4: (255, 255, 0)      # Jaune pour nid-de-poule
}


# =============================================================================
# FONCTIONS DE CHARGEMENT DES MODÈLES
# =============================================================================

def check_models_status():
    """Vérifie le statut des modèles entraînés"""
    status = {
        'U-Net': {
            'weights_path': MODELS_DIR / "unet_best.h5",
            'trained': (MODELS_DIR / "unet_best.h5").exists(),
            'metrics_available': UNET_METRICS.exists()
        },
        'YOLO': {
            'weights_path': MODELS_DIR / "yolo_best.pt",
            'trained': (MODELS_DIR / "yolo_best.pt").exists(),
            'pretrained_available': (BASE_DIR / "yolov8n.pt").exists(),
            'metrics_available': YOLO_METRICS.exists()
        },
        'Hybrid': {
            'weights_path': MODELS_DIR / "hybrid_best.h5",
            'trained': (MODELS_DIR / "hybrid_best.h5").exists(),
            'metrics_available': HYBRID_METRICS.exists()
        }
    }
    return status


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
        weights_path = MODELS_DIR / "unet_best.h5"
        if weights_path.exists():
            model.load_weights(str(weights_path))
            return model, True
        return model, False
    except Exception as e:
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
        weights_path = MODELS_DIR / "hybrid_best.h5"
        if weights_path.exists():
            model.load_weights(str(weights_path))
            return model, True
        return model, False
    except Exception as e:
        return None, False


@st.cache_resource
def load_yolo_model():
    """Charge le modèle YOLO"""
    try:
        from ultralytics import YOLO
        trained_path = MODELS_DIR / "yolo_best.pt"
        if trained_path.exists():
            model = YOLO(str(trained_path))
            return model, True
        pretrained_path = BASE_DIR / "yolov8n.pt"
        if pretrained_path.exists():
            model = YOLO(str(pretrained_path))
            return model, False
        return None, False
    except Exception as e:
        return None, False


# =============================================================================
# FONCTIONS DE PRÉDICTION
# =============================================================================

def preprocess_image(image, target_size=IMG_SIZE):
    """Prétraite l'image pour les modèles"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0

    return image_resized, image_normalized


def predict_unet(model, image_normalized):
    """Effectue une prédiction avec U-Net"""
    input_tensor = np.expand_dims(image_normalized, axis=0)
    start_time = time.time()
    prediction = model.predict(input_tensor, verbose=0)
    inference_time = time.time() - start_time
    mask = np.argmax(prediction[0], axis=-1)
    confidence = np.max(prediction[0], axis=-1)
    return mask, confidence, inference_time


def predict_hybrid(model, image_normalized):
    """Effectue une prédiction avec le modèle Hybride"""
    input_tensor = np.expand_dims(image_normalized, axis=0)
    start_time = time.time()
    prediction = model.predict(input_tensor, verbose=0)
    inference_time = time.time() - start_time
    mask = np.argmax(prediction[0], axis=-1)
    confidence = np.max(prediction[0], axis=-1)
    return mask, confidence, inference_time


def predict_yolo(model, image):
    """Effectue une prédiction avec YOLO"""
    start_time = time.time()
    results = model.predict(source=image, conf=0.25, verbose=False)
    inference_time = time.time() - start_time

    mask = np.zeros(IMG_SIZE, dtype=np.uint8)
    confidence_map = np.zeros(IMG_SIZE, dtype=np.float32)

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            h, w = IMG_SIZE
            for (x1, y1, x2, y2), cls, conf in zip(xyxyn, classes, confs):
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                mask_value = cls + 1 if cls < 4 else 4
                mask[py1:py2, px1:px2] = mask_value
                confidence_map[py1:py2, px1:px2] = conf

    return mask, confidence_map, inference_time


def mask_to_colored_image(mask):
    """Convertit un masque en image colorée"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def overlay_mask_on_image(image, mask, alpha=0.5):
    """Superpose le masque coloré sur l'image originale"""
    colored_mask = mask_to_colored_image(mask)
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    detection_mask = mask > 0
    result = image.copy()
    result[detection_mask] = overlay[detection_mask]
    return result


# =============================================================================
# FONCTIONS D'AFFICHAGE DES MÉTRIQUES
# =============================================================================

def load_metrics():
    """Charge les métriques des trois architectures"""
    metrics = {}

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


def get_best_metrics(metrics):
    """Récupère les meilleures métriques pour chaque architecture"""
    best = {}
    for arch_name, arch_metrics in metrics.items():
        best[arch_name] = {
            'accuracy': max(arch_metrics['accuracy']),
            'dice_coefficient': max(arch_metrics['dice_coefficient']),
            'iou': max(arch_metrics['iou']),
            'loss': min(arch_metrics['loss']),
            'val_accuracy': max(arch_metrics['val_accuracy']),
            'val_dice_coefficient': max(arch_metrics['val_dice_coefficient']),
            'val_iou': max(arch_metrics['val_iou']),
            'val_loss': min(arch_metrics['val_loss'])
        }
    return best


# =============================================================================
# SECTIONS DE L'INTERFACE
# =============================================================================

def display_image_analysis():
    """Section d'analyse d'image avec les 3 modèles"""
    st.header("Analyse d'Image - Détection des Dommages Routiers")

    # Vérifier le statut des modèles
    status = check_models_status()
    metrics = load_metrics()
    best_metrics = get_best_metrics(metrics) if metrics else {}

    # Afficher le statut des modèles
    st.subheader("Statut des Modèles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**U-Net**")
        if status['U-Net']['trained']:
            st.success("Modèle entraîné disponible")
        else:
            st.warning("Non entraîné (poids aléatoires)")
        if status['U-Net']['metrics_available'] and 'U-Net' in best_metrics:
            st.caption(f"Meilleur IoU: {best_metrics['U-Net']['val_iou']:.4f}")
            st.caption(f"Meilleur Dice: {best_metrics['U-Net']['val_dice_coefficient']:.4f}")

    with col2:
        st.markdown("**YOLO**")
        if status['YOLO']['trained']:
            st.success("Modèle entraîné disponible")
        elif status['YOLO']['pretrained_available']:
            st.info("Pré-entraîné COCO (non spécifique)")
        else:
            st.error("Non disponible")
        if status['YOLO']['metrics_available'] and 'YOLO' in best_metrics:
            st.caption(f"Meilleur IoU: {best_metrics['YOLO']['val_iou']:.4f}")
            st.caption(f"Meilleur Dice: {best_metrics['YOLO']['val_dice_coefficient']:.4f}")

    with col3:
        st.markdown("**Hybride**")
        if status['Hybrid']['trained']:
            st.success("Modèle entraîné disponible")
        else:
            st.warning("Non entraîné (poids aléatoires)")
        if status['Hybrid']['metrics_available'] and 'Hybrid' in best_metrics:
            st.caption(f"Meilleur IoU: {best_metrics['Hybrid']['val_iou']:.4f}")
            st.caption(f"Meilleur Dice: {best_metrics['Hybrid']['val_dice_coefficient']:.4f}")

    st.divider()

    # Afficher comparaison des performances d'entraînement
    if metrics:
        st.subheader("Comparaison des Performances (Entraînement)")

        # Graphique des meilleures métriques
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        architectures = list(best_metrics.keys())
        colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

        metrics_to_show = [('val_accuracy', 'Accuracy (Val)'), ('val_dice_coefficient', 'Dice (Val)'), ('val_iou', 'IoU (Val)')]

        for idx, (metric_key, metric_name) in enumerate(metrics_to_show):
            ax = axes[idx]
            values = [best_metrics[arch][metric_key] for arch in architectures]
            bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
            ax.set_title(f'Meilleur {metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Valeur')
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    st.markdown("""
    ### Charger une image pour l'analyse
    Chargez une photo de route pour analyser les dommages avec les 3 architectures.
    """)

    uploaded_file = st.file_uploader(
        "Choisir une image de route",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Formats supportés: PNG, JPG, JPEG, BMP"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_resized, image_normalized = preprocess_image(image_np)

        st.subheader("Image Chargée")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image_resized, caption=f"Image ({IMG_SIZE[0]}x{IMG_SIZE[1]})", use_container_width=True)
        with col2:
            st.info(f"""
            **Informations:**
            - Taille originale: {image_np.shape[1]}x{image_np.shape[0]}
            - Taille traitée: {IMG_SIZE[0]}x{IMG_SIZE[1]}
            """)

        st.divider()

        if st.button("Analyser l'image avec les 3 modèles", type="primary"):
            st.subheader("Résultats de l'Analyse")

            results = {}
            col1, col2, col3 = st.columns(3)

            # === U-NET ===
            with col1:
                st.markdown("### U-Net")
                with st.spinner("Chargement..."):
                    unet_model, unet_trained = load_unet_model()

                if unet_model is not None:
                    if unet_trained:
                        st.success("Modèle entraîné")
                    else:
                        st.warning("Poids aléatoires")

                    with st.spinner("Analyse..."):
                        mask, confidence, inf_time = predict_unet(unet_model, image_normalized)

                    results['U-Net'] = {'mask': mask, 'confidence': confidence, 'time': inf_time, 'trained': unet_trained}

                    overlay = overlay_mask_on_image(image_resized, mask)
                    st.image(overlay, caption="Segmentation U-Net", use_container_width=True)
                    colored_mask = mask_to_colored_image(mask)
                    st.image(colored_mask, caption="Masque", use_container_width=True)
                    st.metric("Temps", f"{inf_time*1000:.1f} ms")
                else:
                    st.error("Non disponible")

            # === YOLO ===
            with col2:
                st.markdown("### YOLO")
                with st.spinner("Chargement..."):
                    yolo_model, yolo_trained = load_yolo_model()

                if yolo_model is not None:
                    if yolo_trained:
                        st.success("Modèle entraîné RDD2022")
                    else:
                        st.info("Pré-entraîné COCO")

                    with st.spinner("Analyse..."):
                        mask, confidence, inf_time = predict_yolo(yolo_model, image_resized)

                    results['YOLO'] = {'mask': mask, 'confidence': confidence, 'time': inf_time, 'trained': yolo_trained}

                    overlay = overlay_mask_on_image(image_resized, mask)
                    st.image(overlay, caption="Détection YOLO", use_container_width=True)
                    colored_mask = mask_to_colored_image(mask)
                    st.image(colored_mask, caption="Masque", use_container_width=True)
                    st.metric("Temps", f"{inf_time*1000:.1f} ms")
                else:
                    st.error("Non disponible")

            # === HYBRIDE ===
            with col3:
                st.markdown("### Hybride")
                with st.spinner("Chargement..."):
                    hybrid_model, hybrid_trained = load_hybrid_model()

                if hybrid_model is not None:
                    if hybrid_trained:
                        st.success("Modèle entraîné")
                    else:
                        st.warning("Poids aléatoires")

                    with st.spinner("Analyse..."):
                        mask, confidence, inf_time = predict_hybrid(hybrid_model, image_normalized)

                    results['Hybrid'] = {'mask': mask, 'confidence': confidence, 'time': inf_time, 'trained': hybrid_trained}

                    overlay = overlay_mask_on_image(image_resized, mask)
                    st.image(overlay, caption="Segmentation Hybride", use_container_width=True)
                    colored_mask = mask_to_colored_image(mask)
                    st.image(colored_mask, caption="Masque", use_container_width=True)
                    st.metric("Temps", f"{inf_time*1000:.1f} ms")
                else:
                    st.error("Non disponible")

            # === COMPARAISON ===
            if results:
                st.divider()
                st.subheader("Comparaison des Résultats")

                comparison_data = []
                for model_name, res in results.items():
                    unique, counts = np.unique(res['mask'], return_counts=True)
                    class_counts = dict(zip(unique, counts))
                    total_pixels = res['mask'].size
                    detected_pixels = total_pixels - class_counts.get(0, 0)
                    detection_rate = (detected_pixels / total_pixels) * 100

                    # Classes détectées
                    detected_classes = [CLASS_NAMES[c] for c in unique if c > 0]

                    comparison_data.append({
                        'Modèle': model_name,
                        'Temps (ms)': f"{res['time']*1000:.1f}",
                        'Pixels détectés': f"{detected_pixels:,}",
                        'Taux détection': f"{detection_rate:.2f}%",
                        'Classes détectées': ', '.join(detected_classes) if detected_classes else 'Aucune',
                        'Entraîné': "Oui" if res['trained'] else "Non"
                    })

                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Légende
                st.subheader("Légende des Classes")
                legend_cols = st.columns(5)
                for idx, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
                    color = CLASS_COLORS[class_id]
                    with legend_cols[idx]:
                        color_box = f'<div style="background-color: rgb{color}; width: 30px; height: 30px; display: inline-block; border: 1px solid black; vertical-align: middle;"></div>'
                        st.markdown(f"{color_box} {class_name}", unsafe_allow_html=True)

                # Note importante
                st.info("""
                **Note:** Pour de meilleurs résultats, les modèles doivent être entraînés sur le dataset RDD2022.
                Les fichiers de poids doivent être placés dans `results/models/`:
                - `unet_best.h5` pour U-Net
                - `yolo_best.pt` pour YOLO
                - `hybrid_best.h5` pour Hybride
                """)


def display_preprocessing_results():
    """Affiche les résultats de prétraitement"""
    st.header("Résultats du Prétraitement")

    st.markdown("""
    Cette section présente les visualisations issues de la phase de prétraitement des données
    du dataset RDD2022 (Road Damage Detection).
    """)

    st.subheader("Distribution des Classes")
    col1, col2, col3 = st.columns(3)

    train_dist = FIGURES_DIR / "class_distribution_train.png"
    val_dist = FIGURES_DIR / "class_distribution_val.png"
    test_dist = FIGURES_DIR / "class_distribution_test.png"

    with col1:
        if train_dist.exists():
            st.image(str(train_dist), caption="Ensemble d'entraînement", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col2:
        if val_dist.exists():
            st.image(str(val_dist), caption="Ensemble de validation", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col3:
        if test_dist.exists():
            st.image(str(test_dist), caption="Ensemble de test", use_container_width=True)
        else:
            st.warning("Image non disponible")

    st.subheader("Augmentation des Données")
    col1, col2 = st.columns(2)

    with col1:
        augmentation_img = FIGURES_DIR / "test_augmentation.png"
        if augmentation_img.exists():
            st.image(str(augmentation_img), caption="Exemples d'augmentation", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col2:
        sample_img = FIGURES_DIR / "test_sample_original.png"
        if sample_img.exists():
            st.image(str(sample_img), caption="Échantillon original", use_container_width=True)
        else:
            st.warning("Image non disponible")


def display_metrics_histograms(metrics, averages):
    """Affiche les histogrammes des métriques"""
    st.header("Comparaison des Métriques - Histogrammes")

    architectures = list(averages.keys())
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    main_metrics = ['accuracy', 'dice_coefficient', 'iou']
    val_metrics = ['val_accuracy', 'val_dice_coefficient', 'val_iou']

    st.subheader("Moyennes Globales des Métriques (sur toutes les époques)")

    df_data = []
    for arch in architectures:
        row = {'Architecture': arch}
        for metric in main_metrics + val_metrics + ['loss', 'val_loss']:
            if metric in averages[arch]:
                row[metric.replace('_', ' ').title()] = f"{averages[arch][metric]:.4f}"
        df_data.append(row)

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Métriques d'Entraînement (Moyennes)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(main_metrics):
        ax = axes[idx]
        values = [averages[arch][metric] for arch in architectures]
        bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Comparaison des Pertes (Loss)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    values = [averages[arch]['loss'] for arch in architectures]
    bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
    ax.set_title('Loss Entraînement', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax = axes[1]
    values = [averages[arch]['val_loss'] for arch in architectures]
    bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
    ax.set_title('Loss Validation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_training_evolution(metrics):
    """Affiche l'évolution des métriques pendant l'entraînement"""
    st.header("Évolution des Métriques pendant l'Entraînement")

    architectures = list(metrics.keys())
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

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
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    st.subheader("Comparaison Radar des Performances")

    metrics_to_compare = ['accuracy', 'dice_coefficient', 'iou',
                          'val_accuracy', 'val_dice_coefficient', 'val_iou']
    labels = ['Accuracy', 'Dice', 'IoU', 'Val Acc', 'Val Dice', 'Val IoU']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for arch in architectures:
        values = [averages[arch][m] for m in metrics_to_compare]
        values += values[:1]
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
            elif 'loss' in metric:
                if val < best_val:
                    best_val = val
                    best_arch = arch
            else:
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

    st.title("Projet Deep Learning - Détection des Dommages Routiers")
    st.markdown("""
    **Master 2 HPC - Année 2025-2026**
    Segmentation sémantique des dégradations routières avec U-Net, YOLO et architecture Hybride.
    """)

    st.divider()

    menu = st.sidebar.radio(
        "Navigation",
        ["Analyse d'Image", "Prétraitement", "Histogrammes des Métriques",
         "Évolution de l'Entraînement", "Métriques Détaillées", "Synthèse Finale"]
    )

    metrics = load_metrics()
    averages = calculate_global_averages(metrics) if metrics else {}

    if menu == "Analyse d'Image":
        display_image_analysis()

    elif menu == "Prétraitement":
        display_preprocessing_results()

    elif menu == "Histogrammes des Métriques":
        if metrics:
            display_metrics_histograms(metrics, averages)
        else:
            st.error("Aucune métrique trouvée.")

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
