"""
Test ULTRA-RAPIDE pour U-Net
Utilise seulement 200 images pour tester que tout fonctionne
Durée estimée : 2-3 minutes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

from models.unet_scratch import create_unet_model
from models.model_utils import DiceCoefficient, IoUMetric
from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor
from utils.config import *
from utils.helpers import *


class FastUNetDataGenerator(keras.utils.Sequence):
    """
    Data Generator RAPIDE - utilise seulement un subset des données
    """
    
    def __init__(self,
                 data_loader: RDD2022DataLoader,
                 preprocessor: ImagePreprocessor,
                 max_samples: int = 200,  # SEULEMENT 200 images !
                 batch_size: int = 16,
                 shuffle: bool = True):
        """
        Args:
            data_loader: Instance de RDD2022DataLoader
            preprocessor: Instance de ImagePreprocessor
            max_samples: Nombre maximum d'images à utiliser
            batch_size: Taille des batches
            shuffle: Mélanger les données
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Utiliser seulement un subset des données
        total_samples = len(self.data_loader)
        self.max_samples = min(max_samples, total_samples)
        
        # Indices aléatoires
        self.indices = np.random.choice(total_samples, self.max_samples, replace=False)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        print(f"⚡ Mode RAPIDE: Utilise {self.max_samples} images sur {total_samples}")
    
    def __len__(self):
        """Nombre de batches par epoch"""
        return int(np.ceil(self.max_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """Génère un batch"""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.max_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        
        images_list = []
        masks_list = []
        
        for i in batch_indices:
            # Charger
            image, mask, _ = self.data_loader[i]
            
            # Prétraiter
            image = self.preprocessor.preprocess_image(image)
            mask = self.preprocessor.preprocess_mask(mask)
            
            images_list.append(image)
            masks_list.append(mask)
        
        # Convertir
        images_batch = np.array(images_list, dtype=np.float32)
        masks_batch = np.array(masks_list, dtype=np.uint8)
        
        # Masks en categorical
        masks_categorical = np.array([
            self.preprocessor.mask_to_categorical(mask)
            for mask in masks_batch
        ], dtype=np.float32)
        
        return images_batch, masks_categorical
    
    def on_epoch_end(self):
        """Appelé à la fin de chaque epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def ultra_fast_test(data_path: str):
    """
    Test ULTRA-RAPIDE avec seulement 200 images et 2 epochs
    Durée estimée : 2-3 minutes
    """
    print("\n" + "=" * 100)
    print("⚡ TEST ULTRA-RAPIDE U-NET (200 IMAGES, 2 EPOCHS)")
    print("=" * 100)
    print("\n🎯 Objectif : Vérifier que l'entraînement fonctionne")
    print("⏱️  Durée estimée : 2-3 minutes")
    print("=" * 100)
    
    # Seeds
    set_seeds(RANDOM_SEED)
    
    # Créer les dossiers
    create_directories()
    
    # ========================================================================
    # 1. CHARGEMENT DES DONNÉES (MODE RAPIDE)
    # ========================================================================
    print("\n📦 Chargement des données (mode rapide)...")
    
    train_loader = RDD2022DataLoader(data_path, split='train')
    val_loader = RDD2022DataLoader(data_path, split='val')
    
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    # Créer les generators RAPIDES
    train_generator = FastUNetDataGenerator(
        data_loader=train_loader,
        preprocessor=preprocessor,
        max_samples=200,  # Seulement 200 images !
        batch_size=16,
        shuffle=True
    )
    
    val_generator = FastUNetDataGenerator(
        data_loader=val_loader,
        preprocessor=preprocessor,
        max_samples=50,  # Seulement 50 images pour validation !
        batch_size=16,
        shuffle=False
    )
    
    print(f"✅ Données chargées:")
    print(f"  - Train: {200} images (au lieu de {len(train_loader)})")
    print(f"  - Val: {50} images (au lieu de {len(val_loader)})")
    print(f"  - Train batches: {len(train_generator)}")
    print(f"  - Val batches: {len(val_generator)}")
    
    # ========================================================================
    # 2. CRÉATION DU MODÈLE (PETIT)
    # ========================================================================
    print("\n🏗️  Création du modèle U-Net (version petite)...")
    
    # Utiliser moins de filtres pour aller plus vite
    model = create_unet_model(
        input_shape=IMG_SIZE + (IMG_CHANNELS,),
        num_classes=NUM_CLASSES + 1,
        filters_base=32,  # 32 au lieu de 64 = 4x plus rapide !
        compile_model=False
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # LR plus élevé pour test
        loss='categorical_crossentropy',
        metrics=['accuracy', DiceCoefficient(), IoUMetric()]
    )
    
    print("✅ Modèle créé")
    print(f"  - Paramètres: {model.count_params():,}")
    
    # ========================================================================
    # 3. CALLBACKS SIMPLES
    # ========================================================================
    print("\n📊 Configuration des callbacks...")
    
    callbacks = [
        keras.callbacks.ProgbarLogger(),
        keras.callbacks.History()
    ]
    
    print("✅ Callbacks configurés (mode simple)")
    
    # ========================================================================
    # 4. ENTRAÎNEMENT
    # ========================================================================
    print("\n🚀 Début de l'entraînement...")
    print("=" * 100)
    
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=2,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 100)
    print("✅ TEST TERMINÉ!")
    print("=" * 100)
    print(f"⏱️  Temps total: {format_time(training_time)}")
    
    # ========================================================================
    # 5. RÉSULTATS
    # ========================================================================
    print("\n📊 Résultats finaux:")
    print("-" * 100)
    
    final_metrics = {
        'loss': history.history['loss'][-1],
        'accuracy': history.history['accuracy'][-1],
        'dice': history.history['dice_coefficient'][-1],
        'iou': history.history['iou'][-1],
        'val_loss': history.history['val_loss'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'val_dice': history.history['val_dice_coefficient'][-1],
        'val_iou': history.history['val_iou'][-1]
    }
    
    for name, value in final_metrics.items():
        print(f"  {name:20s}: {value:.4f}")
    
    print("-" * 100)
    
    # ========================================================================
    # 6. CONCLUSION
    # ========================================================================
    print("\n" + "=" * 100)
    print("🎉 CONCLUSION DU TEST")
    print("=" * 100)
    
    if final_metrics['val_dice'] > 0.3:
        print("✅ Le modèle apprend correctement!")
        print("✅ Dice coefficient > 0.3 : C'est bon signe")
        print("\n💡 Tu peux maintenant lancer l'entraînement complet:")
        print("   python training/train_unet.py")
    else:
        print("⚠️  Le modèle apprend lentement (normal pour 2 epochs)")
        print("⚠️  Mais le code fonctionne correctement!")
        print("\n💡 Pour de meilleurs résultats, lance l'entraînement complet:")
        print("   python training/train_unet.py")
    
    print("=" * 100)
    
    return model, history


if __name__ == "__main__":
    # Chemin du dataset
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    
    # Vérifier que le chemin existe
    if not os.path.exists(DATA_PATH):
        print("\n" + "=" * 100)
        print("❌ ERREUR: Le chemin du dataset n'existe pas!")
        print("=" * 100)
        print(f"\nChemin spécifié: {DATA_PATH}")
        print("\nVeuillez modifier la variable DATA_PATH dans ce script.")
        print("=" * 100)
    else:
        # Lancer le test ultra-rapide
        model, history = ultra_fast_test(DATA_PATH)
