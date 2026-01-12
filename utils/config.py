"""
Configuration globale pour le projet de détection de dommages routiers
RDD2022 Dataset - Deep Learning Project
"""

import os

# ============================================================================
# CHEMINS DU PROJET
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')

# ============================================================================
# CONFIGURATION DU DATASET RDD2022
# ============================================================================
# Classes de dommages routiers
CLASS_NAMES = {
    0: "Fissure longitudinale",
    1: "Fissure transversale", 
    2: "Fissure crocodile",
    4: "Nid-de-poule"
}

NUM_CLASSES = len(CLASS_NAMES)
CLASS_IDS = list(CLASS_NAMES.keys())

# Couleurs pour la visualisation (BGR pour OpenCV)
CLASS_COLORS = {
    0: (255, 0, 0),      # Bleu pour longitudinale
    1: (0, 255, 0),      # Vert pour transversale
    2: (0, 0, 255),      # Rouge pour crocodile
    4: (255, 255, 0)     # Cyan pour nid-de-poule
}

# ============================================================================
# CONFIGURATION DES IMAGES
# ============================================================================
# Taille des images pour l'entraînement
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Taille originale des images (si connue, sinon sera détectée)
ORIGINAL_HEIGHT = 720
ORIGINAL_WIDTH = 1280

# ============================================================================
# CONFIGURATION DE L'ENTRAÎNEMENT
# ============================================================================
# Batch size
BATCH_SIZE = 16
BATCH_SIZE_INFERENCE = 1

# Nombre d'epochs
EPOCHS = 100
EPOCHS_YOLO = 50

# Learning rate
LEARNING_RATE = 1e-4
LEARNING_RATE_YOLO = 1e-3

# Validation split (si besoin de re-split)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================================
# CONFIGURATION DE L'AUGMENTATION
# ============================================================================
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 15,          # degrés
    'zoom_range': 0.1,              # 10%
    'brightness_range': (0.8, 1.2),
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
}

# ============================================================================
# CONFIGURATION U-NET
# ============================================================================
UNET_CONFIG = {
    'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
    'filters_base': 64,              # Nombre de filtres dans la première couche
    'num_classes': NUM_CLASSES + 1,  # +1 pour le background
    'dropout': 0.3,
    'batch_norm': True,
}

# ============================================================================
# CONFIGURATION YOLO
# ============================================================================
YOLO_CONFIG = {
    'model_name': 'yolov8n-seg.pt',  # YOLOv8 nano pour la segmentation
    'img_size': 640,
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
}

# ============================================================================
# CONFIGURATION MODÈLE HYBRIDE
# ============================================================================
HYBRID_CONFIG = {
    'backbone': 'efficientnet',      # ou 'resnet', 'mobilenet'
    'use_attention': True,
    'use_skip_connections': True,
    'decoder_filters': [256, 128, 64, 32],
    'dropout': 0.3,
}

# ============================================================================
# CONFIGURATION DES CALLBACKS
# ============================================================================
CALLBACKS_CONFIG = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True,
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-7,
    },
    'model_checkpoint': {
        'monitor': 'val_loss',
        'save_best_only': True,
    }
}

# ============================================================================
# CONFIGURATION DES MÉTRIQUES
# ============================================================================
METRICS_CONFIG = {
    'iou_threshold': 0.5,
    'dice_smooth': 1e-6,
}

# ============================================================================
# CHEMINS DE SAUVEGARDE
# ============================================================================
MODELS_DIR = os.path.join(RESULTS_ROOT, 'models')
LOGS_DIR = os.path.join(RESULTS_ROOT, 'logs')
FIGURES_DIR = os.path.join(RESULTS_ROOT, 'figures')

# ============================================================================
# SEED POUR LA REPRODUCTIBILITÉ
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# FONCTION UTILITAIRE
# ============================================================================
def create_directories():
    """Crée tous les dossiers nécessaires pour le projet"""
    directories = [
        DATA_ROOT,
        RESULTS_ROOT,
        MODELS_DIR,
        LOGS_DIR,
        FIGURES_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Tous les dossiers nécessaires ont été créés")

if __name__ == "__main__":
    create_directories()
    print("\n📋 Configuration du projet:")
    print(f"  - Nombre de classes: {NUM_CLASSES}")
    print(f"  - Taille des images: {IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
