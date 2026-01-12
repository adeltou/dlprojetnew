"""
Script d'Entraînement ULTRA-RAPIDE avec 100 Images
Temps estimé : 5-8 minutes pour les 3 modèles
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor
from models.unet_scratch import create_unet_model
from models.hybrid_model import create_hybrid_model
from models.model_utils import DiceCoefficient, IoUMetric
from training.callbacks import create_callbacks
from utils.config import *
from utils.helpers import set_seeds

import tensorflow as tf


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
NUM_TRAIN_IMAGES = 500   # ← CHANGE ICI pour modifier le nombre d'images train
NUM_VAL_IMAGES = 100      # ← CHANGE ICI pour modifier le nombre d'images val
NUM_EPOCHS = 10          # ← CHANGE ICI pour modifier le nombre d'epochs


def train_unet_100(data_path, epochs=NUM_EPOCHS):
    """
    Entraîne U-Net sur 100 images
    Temps estimé : 2-3 minutes
    """
    print("\n" + "=" * 100)
    print(f"⚡ ENTRAÎNEMENT U-NET - {NUM_TRAIN_IMAGES} IMAGES, {epochs} EPOCHS")
    print("=" * 100)
    print(f"⏱️  Temps estimé : 2-3 minutes")
    print("=" * 100)
    
    # Configuration
    set_seeds(RANDOM_SEED)
    
    # 1. Charger les données
    print(f"\n📦 Chargement de {NUM_TRAIN_IMAGES} images...")
    
    train_loader = RDD2022DataLoader(data_path, split='train')
    val_loader = RDD2022DataLoader(data_path, split='val')
    
    # Charger les images
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    train_images = []
    train_masks = []
    for i in range(NUM_TRAIN_IMAGES):
        img, mask, _ = train_loader[i]
        train_images.append(img)
        train_masks.append(mask)
    
    val_images = []
    val_masks = []
    for i in range(NUM_VAL_IMAGES):
        img, mask, _ = val_loader[i]
        val_images.append(img)
        val_masks.append(mask)
    
    # Prétraiter
    print("🔧 Prétraitement...")
    X_train, y_train = preprocessor.preprocess_batch(train_images, train_masks)
    X_val, y_val = preprocessor.preprocess_batch(val_images, val_masks)
    
    # Convertir en categorical
    y_train_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_train])
    y_val_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_val])
    
    print(f"✅ Train: {X_train.shape[0]} images")
    print(f"✅ Val: {X_val.shape[0]} images")
    
    # 2. Créer le modèle
    print("\n🏗️  Création du modèle U-Net...")
    
    model = create_unet_model(
        input_shape=IMG_SIZE + (IMG_CHANNELS,),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        compile_model=True
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', DiceCoefficient(), IoUMetric()]
    )
    
    print("✅ Modèle créé")
    
    # 3. Callbacks
    callbacks = create_callbacks(
        model_name='unet_100img',
        models_dir=MODELS_DIR,
        log_dir=LOGS_DIR,
        monitor='val_dice_coefficient',
        patience_early_stop=5,  # Réduit pour 100 images
        patience_reduce_lr=3
    )
    
    # 4. Entraînement
    print(f"\n🚀 Entraînement ({epochs} epochs)...\n")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=8,  # Réduit pour aller plus vite avec peu d'images
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Résultats
    print("\n" + "=" * 100)
    print(f"✅ U-NET TERMINÉ ({NUM_TRAIN_IMAGES} images, {epochs} epochs)")
    print("=" * 100)
    
    final_metrics = {
        'train_dice': history.history['dice_coefficient'][-1],
        'train_iou': history.history['iou'][-1],
        'val_dice': history.history['val_dice_coefficient'][-1],
        'val_iou': history.history['val_iou'][-1]
    }
    
    print(f"\n📊 Résultats finaux:")
    for key, value in final_metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    return model, history


def train_hybrid_100(data_path, epochs=NUM_EPOCHS):
    """
    Entraîne le modèle Hybride sur 100 images
    Temps estimé : 2-3 minutes
    """
    print("\n" + "=" * 100)
    print(f"⚡ ENTRAÎNEMENT HYBRIDE - {NUM_TRAIN_IMAGES} IMAGES, {epochs} EPOCHS")
    print("=" * 100)
    print(f"⏱️  Temps estimé : 2-3 minutes")
    print("=" * 100)
    
    # Configuration
    set_seeds(RANDOM_SEED)
    
    # 1. Charger les données
    print(f"\n📦 Chargement de {NUM_TRAIN_IMAGES} images...")
    
    train_loader = RDD2022DataLoader(data_path, split='train')
    val_loader = RDD2022DataLoader(data_path, split='val')
    
    # Charger et prétraiter
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    train_images = []
    train_masks = []
    for i in range(NUM_TRAIN_IMAGES):
        img, mask, _ = train_loader[i]
        train_images.append(img)
        train_masks.append(mask)
    
    val_images = []
    val_masks = []
    for i in range(NUM_VAL_IMAGES):
        img, mask, _ = val_loader[i]
        val_images.append(img)
        val_masks.append(mask)
    
    print("🔧 Prétraitement...")
    X_train, y_train = preprocessor.preprocess_batch(train_images, train_masks)
    X_val, y_val = preprocessor.preprocess_batch(val_images, val_masks)
    
    y_train_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_train])
    y_val_cat = np.array([preprocessor.mask_to_categorical(m) for m in y_val])
    
    print(f"✅ Train: {X_train.shape[0]} images")
    print(f"✅ Val: {X_val.shape[0]} images")
    
    # 2. Créer le modèle
    print("\n🏗️  Création du modèle Hybride...")
    
    model = create_hybrid_model(
        input_shape=IMG_SIZE + (IMG_CHANNELS,),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        compile_model=True
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', DiceCoefficient(), IoUMetric()]
    )
    
    print("✅ Modèle créé")
    
    # 3. Callbacks
    callbacks = create_callbacks(
        model_name='hybrid_100img',
        models_dir=MODELS_DIR,
        log_dir=LOGS_DIR,
        monitor='val_dice_coefficient',
        patience_early_stop=5,
        patience_reduce_lr=3
    )
    
    # 4. Entraînement
    print(f"\n🚀 Entraînement ({epochs} epochs)...\n")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Résultats
    print("\n" + "=" * 100)
    print(f"✅ HYBRIDE TERMINÉ ({NUM_TRAIN_IMAGES} images, {epochs} epochs)")
    print("=" * 100)
    
    final_metrics = {
        'train_dice': history.history['dice_coefficient'][-1],
        'train_iou': history.history['iou'][-1],
        'val_dice': history.history['val_dice_coefficient'][-1],
        'val_iou': history.history['val_iou'][-1]
    }
    
    print(f"\n📊 Résultats finaux:")
    for key, value in final_metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    return model, history


def train_yolo_100(data_path, epochs=NUM_EPOCHS):
    """
    Entraîne YOLO sur 100 images
    Temps estimé : 2-3 minutes
    """
    print("\n" + "=" * 100)
    print(f"⚡ ENTRAÎNEMENT YOLO - {NUM_TRAIN_IMAGES} IMAGES, {epochs} EPOCHS")
    print("=" * 100)
    print(f"⏱️  Temps estimé : 2-3 minutes")
    print("=" * 100)
    
    try:
        from ultralytics import YOLO
        from models.yolo_pretrained import YOLOSegmentation, create_yolo_data_yaml
        import shutil
        from pathlib import Path
        
        # Créer un dataset temporaire avec 100 images
        print(f"\n📦 Création dataset temporaire ({NUM_TRAIN_IMAGES} images)...")
        
        temp_dir = Path(data_path).parent / 'RDD_SPLIT_100'
        
        # Supprimer l'ancien si existe
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Créer la structure
        (temp_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copier 100 images train
        source_train_img = Path(data_path) / 'train' / 'images'
        source_train_lbl = Path(data_path) / 'train' / 'labels'
        
        train_files = list(source_train_img.glob('*.jpg'))[:NUM_TRAIN_IMAGES]
        
        for img_file in train_files:
            shutil.copy(img_file, temp_dir / 'train' / 'images')
            lbl_file = source_train_lbl / (img_file.stem + '.txt')
            if lbl_file.exists():
                shutil.copy(lbl_file, temp_dir / 'train' / 'labels')
        
        # Copier 20 images val
        source_val_img = Path(data_path) / 'val' / 'images'
        source_val_lbl = Path(data_path) / 'val' / 'labels'
        
        val_files = list(source_val_img.glob('*.jpg'))[:NUM_VAL_IMAGES]
        
        for img_file in val_files:
            shutil.copy(img_file, temp_dir / 'val' / 'images')
            lbl_file = source_val_lbl / (img_file.stem + '.txt')
            if lbl_file.exists():
                shutil.copy(lbl_file, temp_dir / 'val' / 'labels')
        
        print(f"✅ Dataset créé: {temp_dir}")
        
        # Créer le fichier YAML
        yaml_path = temp_dir / 'data.yaml'
        create_yolo_data_yaml(
            train_path=str(temp_dir / 'train'),
            val_path=str(temp_dir / 'val'),
            output_path=str(yaml_path)
        )
        
        # Créer le modèle YOLO
        print("\n🏗️  Chargement YOLO...")
        yolo = YOLOSegmentation(
            model_name='yolov8n-seg.pt',
            num_classes=NUM_CLASSES,
            img_size=640
        )
        yolo.load_pretrained()
        
        # Entraîner
        print(f"\n🚀 Entraînement YOLO ({epochs} epochs)...\n")
        
        results = yolo.model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=4,  # Réduit pour aller plus vite
            imgsz=640,
            project=str(Path(data_path).parent / 'yolo_temp_results'),
            name='yolo_100img',
            patience=5,
            save=False,  # Ne pas sauvegarder pour gagner du temps
            plots=False,
            verbose=True
        )
        
        print("\n" + "=" * 100)
        print(f"✅ YOLO TERMINÉ ({NUM_TRAIN_IMAGES} images, {epochs} epochs)")
        print("=" * 100)
        
        # Nettoyage
        print("\n🗑️  Nettoyage...")
        shutil.rmtree(temp_dir)
        
        return yolo, results
        
    except ImportError:
        print("❌ Ultralytics non disponible!")
        return None, None


if __name__ == "__main__":
    # Chemin vers le dataset
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    
    # Vérifier que le chemin existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERREUR: Le dataset n'existe pas à {DATA_PATH}")
        sys.exit(1)
    
    # Menu
    print("\n" + "=" * 100)
    print(f"⚡ ENTRAÎNEMENT ULTRA-RAPIDE - {NUM_TRAIN_IMAGES} IMAGES, {NUM_EPOCHS} EPOCHS")
    print("=" * 100)
    print("\n💡 Pour changer le nombre d'images ou d'epochs :")
    print(f"   - Ouvre ce fichier et modifie les lignes 19-21")
    print(f"   - NUM_TRAIN_IMAGES = {NUM_TRAIN_IMAGES}  ← Nombre d'images train")
    print(f"   - NUM_VAL_IMAGES = {NUM_VAL_IMAGES}      ← Nombre d'images val")
    print(f"   - NUM_EPOCHS = {NUM_EPOCHS}            ← Nombre d'epochs")
    print()
    print("Quel modèle voulez-vous entraîner ?")
    print("  1. U-Net (~2 min)")
    print("  2. Hybride (~3 min)")
    print("  3. YOLO (~2 min)")
    print("  4. TOUS (~8 min)")
    print()
    
    choice = input("Votre choix (1-4): ").strip()
    
    if choice == '1':
        model, history = train_unet_100(DATA_PATH)
        
    elif choice == '2':
        model, history = train_hybrid_100(DATA_PATH)
        
    elif choice == '3':
        model, results = train_yolo_100(DATA_PATH)
        
    elif choice == '4':
        print("\n🚀 Entraînement des 3 modèles...")
        print(f"⏱️  Temps total estimé : ~8 minutes")
        print()
        
        # U-Net
        print("\n" + "🔹" * 50)
        print("1/3 - U-NET")
        print("🔹" * 50)
        unet_model, unet_history = train_unet_100(DATA_PATH)
        
        # Hybride
        print("\n" + "🔹" * 50)
        print("2/3 - HYBRIDE")
        print("🔹" * 50)
        hybrid_model, hybrid_history = train_hybrid_100(DATA_PATH)
        
        # YOLO
        print("\n" + "🔹" * 50)
        print("3/3 - YOLO")
        print("🔹" * 50)
        yolo_model, yolo_results = train_yolo_100(DATA_PATH)
        
        print("\n" + "=" * 100)
        print("🎉 TOUS LES ENTRAÎNEMENTS TERMINÉS!")
        print("=" * 100)
        print(f"\n📊 Résumé:")
        print(f"  - {NUM_TRAIN_IMAGES} images d'entraînement")
        print(f"  - {NUM_EPOCHS} epochs par modèle")
        print(f"  - 3 modèles entraînés avec succès")
        
    else:
        print("❌ Choix invalide!")
