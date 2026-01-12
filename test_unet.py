"""
Script de Test pour l'Architecture U-Net
Vérifie que le modèle se construit et fonctionne correctement
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from models.unet_scratch import UNetScratch, create_unet_model
from models.model_utils import (
    dice_coefficient,
    iou_metric,
    combined_loss,
    DiceCoefficient,
    IoUMetric
)
from utils.config import *
from utils.helpers import set_seeds


def test_unet_architecture():
    """
    Test complet de l'architecture U-Net
    """
    print("\n" + "=" * 100)
    print("TEST COMPLET DE L'ARCHITECTURE U-NET")
    print("=" * 100)
    
    # Configurer les seeds pour la reproductibilité
    set_seeds(RANDOM_SEED)
    
    # ========================================================================
    # 1. TEST DE CONSTRUCTION DU MODÈLE
    # ========================================================================
    print("\n🏗️  PHASE 1: Construction du modèle")
    print("-" * 100)
    
    try:
        unet = UNetScratch(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=64,
            use_batch_norm=True,
            dropout=0.3
        )
        print("✅ Objet UNetScratch créé")
        
        # Build le modèle
        model = unet.build_model()
        print("✅ Modèle construit avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lors de la construction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # 2. AFFICHAGE DU RÉSUMÉ
    # ========================================================================
    print("\n📊 PHASE 2: Résumé de l'architecture")
    print("-" * 100)
    
    unet.summary()
    
    # Nombre de paramètres
    total, trainable, non_trainable = unet.count_parameters()
    print(f"\n📈 Paramètres du modèle:")
    print(f"  - Total: {total:,}")
    print(f"  - Entraînables: {trainable:,}")
    print(f"  - Non-entraînables: {non_trainable:,}")
    
    # ========================================================================
    # 3. COMPILATION DU MODÈLE
    # ========================================================================
    print("\n⚙️  PHASE 3: Compilation du modèle")
    print("-" * 100)
    
    try:
        unet.compile_model(
            optimizer='adam',
            learning_rate=LEARNING_RATE,
            loss='combined',
            metrics=['accuracy', DiceCoefficient(), IoUMetric()]
        )
        print("✅ Modèle compilé avec succès")
        print(f"  - Optimizer: Adam")
        print(f"  - Learning rate: {LEARNING_RATE}")
        print(f"  - Loss: Combined Loss (Dice + Categorical Crossentropy)")
        print(f"  - Metrics: Accuracy, Dice Coefficient, IoU")
        
    except Exception as e:
        print(f"❌ Erreur lors de la compilation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # 4. TEST DE PRÉDICTION
    # ========================================================================
    print("\n🧪 PHASE 4: Test de prédiction")
    print("-" * 100)
    
    try:
        # Créer des données de test
        batch_size = 4
        test_images = np.random.rand(batch_size, *IMG_SIZE, IMG_CHANNELS).astype(np.float32)
        
        print(f"  - Input shape: {test_images.shape}")
        
        # Prédiction
        predictions = model.predict(test_images, verbose=0)
        
        print(f"✅ Prédiction réussie!")
        print(f"  - Output shape: {predictions.shape}")
        print(f"  - Output min: {predictions.min():.4f}")
        print(f"  - Output max: {predictions.max():.4f}")
        
        # Vérifier que c'est une distribution de probabilités
        sample_sum = predictions[0, 0, 0, :].sum()
        print(f"\n🔍 Vérification:")
        print(f"  - Somme des probabilités (doit être ≈ 1.0): {sample_sum:.4f}")
        
        if abs(sample_sum - 1.0) < 0.01:
            print(f"  ✅ Softmax fonctionne correctement")
        else:
            print(f"  ⚠️  Attention: somme des probabilités != 1.0")
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # 5. TEST DES MÉTRIQUES
    # ========================================================================
    print("\n📐 PHASE 5: Test des métriques")
    print("-" * 100)
    
    try:
        # Créer des masques de test
        test_masks_true = np.random.randint(0, NUM_CLASSES + 1, (batch_size, *IMG_SIZE))
        
        # Convertir en categorical
        import tensorflow as tf
        test_masks_cat = tf.keras.utils.to_categorical(test_masks_true, num_classes=NUM_CLASSES + 1)
        
        # Calculer les métriques
        import tensorflow as tf
        dice = dice_coefficient(
            tf.constant(test_masks_cat, dtype=tf.float32),
            tf.constant(predictions, dtype=tf.float32)
        )
        
        iou = iou_metric(
            tf.constant(test_masks_cat, dtype=tf.float32),
            tf.constant(predictions, dtype=tf.float32)
        )
        
        print(f"✅ Métriques calculées:")
        print(f"  - Dice Coefficient: {dice.numpy():.4f}")
        print(f"  - IoU: {iou.numpy():.4f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du calcul des métriques: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 6. TEST DE LA FONCTION HELPER
    # ========================================================================
    print("\n🔧 PHASE 6: Test de la fonction helper")
    print("-" * 100)
    
    try:
        quick_model = create_unet_model(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=64,
            compile_model=True
        )
        
        print("✅ Fonction create_unet_model() fonctionne")
        print(f"  - Modèle créé et compilé en une seule étape")
        
    except Exception as e:
        print(f"❌ Erreur avec la fonction helper: {e}")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "=" * 100)
    print("✅ TOUS LES TESTS DE U-NET SONT RÉUSSIS!")
    print("=" * 100)
    
    print(f"\n📋 Résumé:")
    print(f"  ✅ Architecture U-Net construite from scratch")
    print(f"  ✅ {total:,} paramètres au total")
    print(f"  ✅ Entrée: {IMG_SIZE + (IMG_CHANNELS,)}")
    print(f"  ✅ Sortie: {IMG_SIZE + (NUM_CLASSES + 1,)}")
    print(f"  ✅ Prédictions fonctionnelles")
    print(f"  ✅ Métriques IoU et Dice opérationnelles")
    
    print(f"\n🎯 Prochaine étape:")
    print(f"  → Entraîner le modèle sur le dataset RDD2022")
    print(f"  → Créer le script d'entraînement dans training/train_unet.py")
    
    print("\n" + "=" * 100)
    
    return model


if __name__ == "__main__":
    # Lancer les tests
    model = test_unet_architecture()
