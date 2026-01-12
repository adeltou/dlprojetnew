"""
Script d'Entraînement pour U-Net
Entraîne le modèle U-Net from scratch sur le dataset RDD2022
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from datetime import datetime

from models.unet_scratch import create_unet_model
from models.model_utils import DiceCoefficient, IoUMetric
from preprocessing.data_loader import RDD2022DataLoader
from preprocessing.preprocessing import ImagePreprocessor, DataAugmentorSimple
from training.callbacks import create_callbacks, TrainingProgressCallback
from utils.config import *
from utils.helpers import *


class UNetDataGenerator(keras.utils.Sequence):
    """
    Data Generator pour U-Net
    Génère des batches d'images et masques pour l'entraînement
    """
    
    def __init__(self,
                 data_loader: RDD2022DataLoader,
                 preprocessor: ImagePreprocessor,
                 batch_size: int = BATCH_SIZE,
                 shuffle: bool = True,
                 augment: bool = False):
        """
        Args:
            data_loader: Instance de RDD2022DataLoader
            preprocessor: Instance de ImagePreprocessor
            batch_size: Taille des batches
            shuffle: Mélanger les données à chaque epoch
            augment: Appliquer l'augmentation de données
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Créer l'augmenteur si nécessaire
        if self.augment:
            self.augmentor = DataAugmentorSimple()
        
        # Indices
        self.indices = np.arange(len(self.data_loader))
        
        # Mélanger au début
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Nombre de batches par epoch"""
        return int(np.ceil(len(self.data_loader) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Génère un batch de données
        
        Args:
            idx: Index du batch
            
        Returns:
            (images_batch, masks_batch)
        """
        # Indices pour ce batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Listes pour stocker les données
        images_list = []
        masks_list = []
        
        for i in batch_indices:
            # Charger l'image et le masque
            image, mask, _ = self.data_loader[i]
            
            # Prétraiter
            image = self.preprocessor.preprocess_image(image)
            mask = self.preprocessor.preprocess_mask(mask)
            
            # Augmenter si demandé
            if self.augment:
                image, mask = self.augmentor.augment(image, mask)
            
            images_list.append(image)
            masks_list.append(mask)
        
        # Convertir en arrays numpy
        images_batch = np.array(images_list, dtype=np.float32)
        masks_batch = np.array(masks_list, dtype=np.uint8)
        
        # Convertir les masques en categorical
        masks_categorical = np.array([
            self.preprocessor.mask_to_categorical(mask)
            for mask in masks_batch
        ], dtype=np.float32)
        
        return images_batch, masks_categorical
    
    def on_epoch_end(self):
        """Appelé à la fin de chaque epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_unet(data_path: str,
               epochs: int = EPOCHS,
               batch_size: int = BATCH_SIZE,
               learning_rate: float = LEARNING_RATE,
               use_augmentation: bool = True,
               save_results: bool = True):
    """
    Fonction principale pour entraîner U-Net
    
    Args:
        data_path: Chemin vers le dataset RDD_SPLIT
        epochs: Nombre d'epochs
        batch_size: Taille du batch
        learning_rate: Learning rate
        use_augmentation: Utiliser l'augmentation de données
        save_results: Sauvegarder les résultats
        
    Returns:
        (model, history)
    """
    print("\n" + "=" * 100)
    print("ENTRAÎNEMENT U-NET - ROAD DAMAGE DETECTION")
    print("=" * 100)
    
    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================
    print("\n🔧 PHASE 1: Configuration")
    print("-" * 100)
    
    # Seeds pour reproductibilité
    set_seeds(RANDOM_SEED)
    
    # Créer les dossiers
    create_directories()
    
    print(f"✅ Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Image size: {IMG_SIZE}")
    print(f"  - Augmentation: {use_augmentation}")
    
    # ========================================================================
    # 2. CHARGEMENT DES DONNÉES
    # ========================================================================
    print("\n📦 PHASE 2: Chargement des données")
    print("-" * 100)
    
    # Charger les datasets
    train_loader = RDD2022DataLoader(data_path, split='train')
    val_loader = RDD2022DataLoader(data_path, split='val')
    
    print(f"✅ Données chargées:")
    print(f"  - Train: {len(train_loader)} images")
    print(f"  - Val: {len(val_loader)} images")
    
    # Créer le préprocesseur
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    # Créer les data generators
    train_generator = UNetDataGenerator(
        data_loader=train_loader,
        preprocessor=preprocessor,
        batch_size=batch_size,
        shuffle=True,
        augment=use_augmentation
    )
    
    val_generator = UNetDataGenerator(
        data_loader=val_loader,
        preprocessor=preprocessor,
        batch_size=batch_size,
        shuffle=False,
        augment=False  # Pas d'augmentation pour la validation
    )
    
    print(f"✅ Data generators créés:")
    print(f"  - Train batches: {len(train_generator)}")
    print(f"  - Val batches: {len(val_generator)}")
    
    # ========================================================================
    # 3. CRÉATION DU MODÈLE
    # ========================================================================
    print("\n🏗️  PHASE 3: Création du modèle U-Net")
    print("-" * 100)
    
    # Créer le modèle
    model = create_unet_model(
        input_shape=IMG_SIZE + (IMG_CHANNELS,),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        compile_model=False
    )
    
    # Compiler avec des métriques personnalisées
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            DiceCoefficient(),
            IoUMetric()
        ]
    )
    
    print("✅ Modèle U-Net créé et compilé")
    
    # Afficher le résumé
    model.summary()
    
    # ========================================================================
    # 4. CRÉATION DES CALLBACKS
    # ========================================================================
    print("\n📊 PHASE 4: Configuration des callbacks")
    print("-" * 100)
    
    callbacks = create_callbacks(
        model_name='unet_rdd2022',
        monitor='val_dice_coefficient',
        patience_early_stop=CALLBACKS_CONFIG['early_stopping']['patience'],
        patience_reduce_lr=CALLBACKS_CONFIG['reduce_lr']['patience'],
        save_best_only=True
    )
    
    # Ajouter le callback de progression
    callbacks.append(TrainingProgressCallback(epochs=epochs))
    
    # ========================================================================
    # 5. ENTRAÎNEMENT
    # ========================================================================
    print("\n🚀 PHASE 5: Entraînement du modèle")
    print("-" * 100)
    print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    
    start_time = time.time()
    
    # Entraîner le modèle
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 100)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 100)
    print(f"Temps total: {format_time(training_time)}")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # 6. ÉVALUATION FINALE
    # ========================================================================
    print("\n📈 PHASE 6: Évaluation finale")
    print("-" * 100)
    
    # Évaluer sur le set de validation
    val_results = model.evaluate(val_generator, verbose=0)
    
    print("\n📊 Résultats sur le set de validation:")
    print("-" * 100)
    for metric_name, value in zip(model.metrics_names, val_results):
        print(f"  {metric_name:25s}: {value:.4f}")
    print("-" * 100)
    
    # ========================================================================
    # 7. SAUVEGARDE DES RÉSULTATS
    # ========================================================================
    if save_results:
        print("\n💾 PHASE 7: Sauvegarde des résultats")
        print("-" * 100)
        
        # Créer un dictionnaire avec les résultats
        results = {
            'model': 'U-Net',
            'architecture': 'from_scratch',
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': training_time,
            'final_metrics': {
                metric_name: float(value)
                for metric_name, value in zip(model.metrics_names, val_results)
            },
            'history': {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'image_size': IMG_SIZE,
                'num_classes': NUM_CLASSES + 1,
                'augmentation': use_augmentation
            }
        }
        
        # Sauvegarder en JSON
        results_path = os.path.join(RESULTS_ROOT, 'unet_training_results.json')
        save_results_json(results, results_path)
        
        # Visualiser l'historique d'entraînement
        fig_path = os.path.join(FIGURES_DIR, 'unet_training_history.png')
        plot_training_history(history.history, save_path=fig_path)
    
    print("\n" + "=" * 100)
    print("✅ TOUS LES TRAITEMENTS TERMINÉS!")
    print("=" * 100)
    
    return model, history


def quick_test_training(data_path: str):
    """
    Test rapide de l'entraînement avec 2 epochs
    
    Args:
        data_path: Chemin vers le dataset
    """
    print("\n" + "=" * 100)
    print("TEST RAPIDE DE L'ENTRAÎNEMENT U-NET (2 EPOCHS)")
    print("=" * 100)
    
    model, history = train_unet(
        data_path=data_path,
        epochs=2,
        batch_size=8,
        learning_rate=1e-4,
        use_augmentation=False,
        save_results=False
    )
    
    print("\n✅ Test rapide réussi!")
    
    return model, history


if __name__ == "__main__":
    # IMPORTANT: Modifier ce chemin selon votre configuration
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
        # Choix de l'entraînement
        print("\n" + "=" * 100)
        print("ENTRAÎNEMENT U-NET")
        print("=" * 100)
        print("\n1. Test rapide (2 epochs)")
        print("2. Entraînement complet (100 epochs)")
        
        choice = input("\nVotre choix (1 ou 2): ")
        
        if choice == "1":
            # Test rapide
            model, history = quick_test_training(DATA_PATH)
        elif choice == "2":
            # Entraînement complet
            model, history = train_unet(
                data_path=DATA_PATH,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                use_augmentation=True,
                save_results=True
            )
        else:
            print("\n❌ Choix invalide!")
