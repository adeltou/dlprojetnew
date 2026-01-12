"""
Script de Test du Module Training
Vérifie que tous les composants du training fonctionnent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf

from training.callbacks import create_callbacks, test_callbacks
from models.unet_scratch import create_unet_model
from models.hybrid_model import create_hybrid_model
from models.model_utils import DiceCoefficient, IoUMetric
from utils.config import *
from utils.helpers import *


def test_callbacks_creation():
    """
    Test de création des callbacks
    """
    print("\n" + "=" * 100)
    print("TEST 1: Création des Callbacks")
    print("=" * 100)
    
    try:
        callbacks = create_callbacks(
            model_name='test_model',
            monitor='val_loss',
            patience_early_stop=5,
            patience_reduce_lr=3,
            save_best_only=True
        )
        
        print(f"\n✅ {len(callbacks)} callbacks créés:")
        for i, cb in enumerate(callbacks, 1):
            print(f"  {i}. {cb.__class__.__name__}")
        
        print("\n✅ Test des callbacks réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generator():
    """
    Test du data generator
    """
    print("\n" + "=" * 100)
    print("TEST 2: Data Generator")
    print("=" * 100)
    
    try:
        # Créer des données factices
        print("\n📦 Création de données de test...")
        
        # Simuler un dataset
        class FakeDataLoader:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Retourner image, mask, annotations factices
                image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                mask = np.random.randint(0, NUM_CLASSES + 1, (720, 1280), dtype=np.uint8)
                annotations = []
                return image, mask, annotations
        
        from preprocessing.preprocessing import ImagePreprocessor
        from training.train_unet import UNetDataGenerator
        
        fake_loader = FakeDataLoader(size=32)
        preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
        
        # Créer le generator
        generator = UNetDataGenerator(
            data_loader=fake_loader,
            preprocessor=preprocessor,
            batch_size=8,
            shuffle=True,
            augment=False
        )
        
        print(f"✅ Generator créé: {len(generator)} batches")
        
        # Tester un batch
        print("\n🧪 Test d'un batch...")
        images, masks = generator[0]
        
        print(f"✅ Batch généré:")
        print(f"  - Images shape: {images.shape}")
        print(f"  - Masks shape: {masks.shape}")
        print(f"  - Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  - Masks classes: {masks.shape[-1]}")
        
        print("\n✅ Test du data generator réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compilation():
    """
    Test de compilation des modèles avec métriques
    """
    print("\n" + "=" * 100)
    print("TEST 3: Compilation des Modèles")
    print("=" * 100)
    
    try:
        print("\n🏗️  Test U-Net...")
        unet = create_unet_model(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=32,  # Réduit pour le test
            compile_model=False
        )
        
        unet.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', DiceCoefficient(), IoUMetric()]
        )
        
        print("✅ U-Net compilé avec métriques personnalisées")
        
        print("\n🏗️  Test Hybride...")
        hybrid = create_hybrid_model(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=32,  # Réduit pour le test
            compile_model=False
        )
        
        hybrid.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', DiceCoefficient(), IoUMetric()]
        )
        
        print("✅ Hybride compilé avec métriques personnalisées")
        
        print("\n✅ Test de compilation réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_one_batch():
    """
    Test d'entraînement sur un seul batch
    """
    print("\n" + "=" * 100)
    print("TEST 4: Entraînement sur 1 Batch")
    print("=" * 100)
    
    try:
        print("\n🏗️  Création du modèle...")
        model = create_unet_model(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=32,  # Petit modèle pour le test
            compile_model=False
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', DiceCoefficient(), IoUMetric()]
        )
        
        print("✅ Modèle créé et compilé")
        
        # Créer des données factices
        print("\n📦 Création de données de test...")
        batch_size = 4
        X = np.random.rand(batch_size, *IMG_SIZE, IMG_CHANNELS).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES + 1, (batch_size, *IMG_SIZE))
        
        # Convertir en categorical
        y_cat = np.array([
            tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES + 1)
            for mask in y
        ], dtype=np.float32)
        
        print(f"✅ Données créées:")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y_cat.shape}")
        
        # Entraîner sur 1 batch
        print("\n🚀 Entraînement sur 1 batch...")
        history = model.fit(
            X, y_cat,
            batch_size=batch_size,
            epochs=2,
            verbose=1
        )
        
        print("\n✅ Entraînement réussi!")
        print(f"  - Loss initiale: {history.history['loss'][0]:.4f}")
        print(f"  - Loss finale: {history.history['loss'][-1]:.4f}")
        
        # Tester la prédiction
        print("\n🧪 Test de prédiction...")
        predictions = model.predict(X, verbose=0)
        
        print(f"✅ Prédiction réussie:")
        print(f"  - Predictions shape: {predictions.shape}")
        
        print("\n✅ Test d'entraînement réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all():
    """
    Lance tous les tests
    """
    print("\n" + "=" * 100)
    print("TEST COMPLET DU MODULE TRAINING")
    print("=" * 100)
    
    # Créer les dossiers
    create_directories()
    
    # Seeds
    set_seeds(RANDOM_SEED)
    
    # Résultats
    results = {}
    
    # Test 1: Callbacks
    results['callbacks'] = test_callbacks_creation()
    
    # Test 2: Data Generator
    results['data_generator'] = test_data_generator()
    
    # Test 3: Compilation
    results['compilation'] = test_model_compilation()
    
    # Test 4: Training
    results['training'] = test_training_one_batch()
    
    # Résumé
    print("\n" + "=" * 100)
    print("RÉSUMÉ DES TESTS")
    print("=" * 100)
    
    for test_name, success in results.items():
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 100)
    if all_passed:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("=" * 100)
        print("\n✅ Le module training est prêt à être utilisé!")
        print("\n🎯 Prochaines étapes:")
        print("  1. Lancer train_unet.py pour entraîner U-Net")
        print("  2. Lancer train_yolo.py pour entraîner YOLO")
        print("  3. Lancer train_hybrid.py pour entraîner le modèle hybride")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("=" * 100)
        print("\nVeuillez corriger les erreurs avant de continuer.")
    
    print("=" * 100)
    
    return all_passed


if __name__ == "__main__":
    success = test_all()
    
    if not success:
        sys.exit(1)
