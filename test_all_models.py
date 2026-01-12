"""
Script de Test Complet pour les 3 Architectures
Teste U-Net, YOLO et Hybride
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from models.unet_scratch import UNetScratch, create_unet_model
from models.yolo_pretrained import YOLOSegmentation, create_yolo_model
from models.hybrid_model import HybridModel, create_hybrid_model
from models.model_utils import (
    dice_coefficient,
    iou_metric,
    DiceCoefficient,
    IoUMetric
)
from utils.config import *
from utils.helpers import set_seeds


def test_all_architectures():
    """
    Test complet des 3 architectures
    """
    print("\n" + "=" * 100)
    print("TEST COMPLET DES 3 ARCHITECTURES")
    print("=" * 100)
    
    # Configurer les seeds
    set_seeds(RANDOM_SEED)
    
    # Résumé des résultats
    results = {}
    
    # ========================================================================
    # 1. TEST U-NET
    # ========================================================================
    print("\n" + "🏗️ " * 30)
    print("ARCHITECTURE 1 : U-NET FROM SCRATCH")
    print("🏗️ " * 30)
    
    try:
        print("\n📦 Création du modèle U-Net...")
        unet = UNetScratch(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=64,
            use_batch_norm=True,
            dropout=0.3
        )
        
        model_unet = unet.build_model()
        print("✅ U-Net construit")
        
        # Compiler
        unet.compile_model(
            optimizer='adam',
            learning_rate=LEARNING_RATE,
            loss='combined',
            metrics=['accuracy', DiceCoefficient(), IoUMetric()]
        )
        print("✅ U-Net compilé")
        
        # Compter les paramètres
        total, trainable, _ = unet.count_parameters()
        results['U-Net'] = {
            'total_params': total,
            'trainable_params': trainable,
            'status': 'SUCCESS'
        }
        print(f"✅ U-Net: {total:,} paramètres")
        
        # Test de prédiction
        test_input = np.random.rand(2, *IMG_SIZE, IMG_CHANNELS).astype(np.float32)
        pred_unet = model_unet.predict(test_input, verbose=0)
        print(f"✅ Prédiction U-Net: {pred_unet.shape}")
        
    except Exception as e:
        print(f"❌ Erreur U-Net: {e}")
        results['U-Net'] = {'status': 'FAILED', 'error': str(e)}
    
    # ========================================================================
    # 2. TEST YOLO
    # ========================================================================
    print("\n" + "🎯 " * 30)
    print("ARCHITECTURE 2 : YOLO SEGMENTATION (ULTRALYTICS)")
    print("🎯 " * 30)
    
    try:
        print("\n📦 Création du modèle YOLO...")
        
        # Vérifier si Ultralytics est disponible
        try:
            from ultralytics import YOLO
            
            yolo = create_yolo_model(model_size='n', pretrained=True)
            print("✅ YOLO chargé (Ultralytics)")
            
            # Informations
            yolo.get_model_info()
            
            results['YOLO'] = {
                'model_size': 'nano',
                'pretrained': True,
                'status': 'SUCCESS'
            }
            
            # Test de prédiction
            test_img_yolo = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            pred_yolo = yolo.predict(source=test_img_yolo, conf=0.25, save=False)
            print(f"✅ Prédiction YOLO réussie")
            
            # Convertir en masque
            mask_yolo = yolo.convert_masks_to_segmentation(pred_yolo, target_size=IMG_SIZE)
            print(f"✅ Masque YOLO converti: {mask_yolo.shape}")
            
        except ImportError:
            print("⚠️  Ultralytics non disponible")
            print("   Installez avec: pip install ultralytics")
            results['YOLO'] = {
                'status': 'SKIPPED',
                'reason': 'Ultralytics not installed'
            }
        
    except Exception as e:
        print(f"❌ Erreur YOLO: {e}")
        results['YOLO'] = {'status': 'FAILED', 'error': str(e)}
    
    # ========================================================================
    # 3. TEST HYBRIDE
    # ========================================================================
    print("\n" + "🚀 " * 30)
    print("ARCHITECTURE 3 : MODÈLE HYBRIDE (U-NET + YOLO INSPIRÉ)")
    print("🚀 " * 30)
    
    try:
        print("\n📦 Création du modèle Hybride...")
        hybrid = HybridModel(
            input_shape=IMG_SIZE + (IMG_CHANNELS,),
            num_classes=NUM_CLASSES + 1,
            filters_base=64,
            use_attention=True,
            use_residual=True,
            use_csp=True,
            use_aspp=True,
            dropout=0.3
        )
        
        model_hybrid = hybrid.build_model()
        print("✅ Hybride construit")
        
        # Compiler
        hybrid.compile_model(
            optimizer='adam',
            learning_rate=LEARNING_RATE,
            loss='combined',
            metrics=['accuracy', DiceCoefficient(), IoUMetric()]
        )
        print("✅ Hybride compilé")
        
        # Compter les paramètres
        total_h, trainable_h, _ = hybrid.count_parameters()
        results['Hybrid'] = {
            'total_params': total_h,
            'trainable_params': trainable_h,
            'features': ['Attention Gates', 'CSP Blocks', 'ASPP', 'Residual Connections'],
            'status': 'SUCCESS'
        }
        print(f"✅ Hybride: {total_h:,} paramètres")
        
        # Test de prédiction
        pred_hybrid = model_hybrid.predict(test_input, verbose=0)
        print(f"✅ Prédiction Hybride: {pred_hybrid.shape}")
        
    except Exception as e:
        print(f"❌ Erreur Hybride: {e}")
        results['Hybrid'] = {'status': 'FAILED', 'error': str(e)}
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "=" * 100)
    print("RÉSUMÉ DES TESTS")
    print("=" * 100)
    
    for model_name, info in results.items():
        print(f"\n🏆 {model_name}:")
        for key, value in info.items():
            if isinstance(value, list):
                print(f"   - {key}:")
                for item in value:
                    print(f"     • {item}")
            else:
                print(f"   - {key}: {value}")
    
    # Tableau comparatif
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF DES ARCHITECTURES")
    print("=" * 100)
    
    print(f"\n{'Architecture':<20} {'Status':<15} {'Paramètres':<20} {'Notes':<40}")
    print("-" * 100)
    
    if 'U-Net' in results and results['U-Net']['status'] == 'SUCCESS':
        params = f"{results['U-Net']['total_params']:,}"
        print(f"{'U-Net from scratch':<20} {'✅ SUCCESS':<15} {params:<20} {'Architecture originale 2015':<40}")
    
    if 'YOLO' in results:
        if results['YOLO']['status'] == 'SUCCESS':
            print(f"{'YOLO-seg':<20} {'✅ SUCCESS':<15} {'Pretrained':<20} {'Ultralytics YOLOv8-seg':<40}")
        else:
            reason = results['YOLO'].get('reason', 'Error')
            print(f"{'YOLO-seg':<20} {'⚠️  SKIPPED':<15} {'-':<20} {reason:<40}")
    
    if 'Hybrid' in results and results['Hybrid']['status'] == 'SUCCESS':
        params_h = f"{results['Hybrid']['total_params']:,}"
        features = ', '.join(results['Hybrid']['features'][:2])
        print(f"{'Hybrid Model':<20} {'✅ SUCCESS':<15} {params_h:<20} {features + '...':<40}")
    
    print("-" * 100)
    
    # Statistiques finales
    total_success = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    total_tested = len(results)
    
    print(f"\n📊 Statistiques:")
    print(f"   - Architectures testées: {total_tested}")
    print(f"   - Succès: {total_success}")
    print(f"   - Échecs: {total_tested - total_success}")
    
    if total_success == total_tested:
        print("\n" + "🎉 " * 30)
        print("TOUTES LES ARCHITECTURES FONCTIONNENT PARFAITEMENT!")
        print("🎉 " * 30)
    
    print("\n" + "=" * 100)
    
    # Prochaines étapes
    print("\n🎯 PROCHAINES ÉTAPES:")
    print("-" * 100)
    print("1. ✅ Preprocessing - COMPLET")
    print("2. ✅ Models (3 architectures) - COMPLET")
    print("3. ⏭️  Training - À créer")
    print("   - callbacks.py")
    print("   - train_unet.py")
    print("   - train_yolo.py")
    print("   - train_hybrid.py")
    print("4. ⏭️  Evaluation - À créer")
    print("5. ⏭️  Interface - À créer")
    print("=" * 100 + "\n")
    
    return results


if __name__ == "__main__":
    # Lancer les tests
    results = test_all_architectures()
