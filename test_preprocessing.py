"""
Script de Test Complet du Module Preprocessing
Vérifie que tous les composants fonctionnent correctement
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.data_loader import RDD2022DataLoader

from preprocessing.preprocessing import ImagePreprocessor, DataAugmentorSimple

from preprocessing.augmentation import AdvancedDataAugmentor

from utils.config import *

from utils.helpers import *

def test_complete_preprocessing_pipeline(data_path: str):
    """
    Test complet du pipeline de preprocessing
    
    Args:
        data_path: Chemin vers le dossier RDD_SPLIT
    """
    print("\n" + "=" * 100)
    print("TEST COMPLET DU PIPELINE DE PREPROCESSING")
    print("=" * 100)
    
    # ============================================================================
    # 1. TEST DU DATA LOADER
    # ============================================================================
    print("\n📦 PHASE 1: Test du Data Loader")
    print("-" * 100)
    
    try:
        # Charger les données de train
        train_loader = RDD2022DataLoader(data_path, split='train')
        print(f"✅ Train loader créé: {len(train_loader)} images")
        
        # Charger les données de validation
        val_loader = RDD2022DataLoader(data_path, split='val')
        print(f"✅ Val loader créé: {len(val_loader)} images")
        
        # Charger les données de test
        test_loader = RDD2022DataLoader(data_path, split='test')
        print(f"✅ Test loader créé: {len(test_loader)} images")
        
        # Charger un échantillon
        if len(train_loader) > 0:
            image, mask, annotations = train_loader[0]
            print(f"\n📊 Échantillon chargé:")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Mask shape: {mask.shape}")
            print(f"  - Nombre d'annotations: {len(annotations)}")
            print(f"  - Classes présentes dans le masque: {np.unique(mask)}")
            
            # Visualiser l'échantillon
            save_path = os.path.join(FIGURES_DIR, 'test_sample_original.png')
            train_loader.visualize_sample(0, save_path=save_path)
            
        # Distribution des classes
        print(f"\n📊 Distribution des classes:")
        for split, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            if len(loader) > 0:
                class_dist = loader.get_class_distribution()
                print(f"\n  {split.upper()}:")
                for class_id, count in class_dist.items():
                    print(f"    - {CLASS_NAMES[class_id]}: {count}")
                
                # Visualiser la distribution
                save_path = os.path.join(FIGURES_DIR, f'class_distribution_{split}.png')
                plot_class_distribution(class_dist, save_path=save_path, 
                                      title=f"Distribution - {split.upper()}")
        
        print("\n✅ Test du Data Loader réussi!")
        
    except Exception as e:
        print(f"\n❌ Erreur dans le Data Loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # 2. TEST DU PREPROCESSING
    # ============================================================================
    print("\n🔧 PHASE 2: Test du Preprocessing")
    print("-" * 100)
    
    try:
        # Créer le préprocesseur
        preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
        print(f"✅ Preprocessor créé (target_size={IMG_SIZE})")
        
        # Prétraiter l'échantillon
        image, mask, _ = train_loader[0]
        
        print(f"\n📊 Avant preprocessing:")
        print(f"  - Image: {image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
        print(f"  - Mask: {mask.shape}, dtype={mask.dtype}, unique={np.unique(mask)}")
        
        proc_image = preprocessor.preprocess_image(image)
        proc_mask = preprocessor.preprocess_mask(mask)
        
        print(f"\n📊 Après preprocessing:")
        print(f"  - Image: {proc_image.shape}, dtype={proc_image.dtype}, range=[{proc_image.min():.3f}, {proc_image.max():.3f}]")
        print(f"  - Mask: {proc_mask.shape}, dtype={proc_mask.dtype}, unique={np.unique(proc_mask)}")
        
        # Test de conversion categorical
        cat_mask = preprocessor.mask_to_categorical(proc_mask)
        print(f"\n✅ Masque categorical: {cat_mask.shape}")
        
        # Reconversion
        reconstructed_mask = preprocessor.categorical_to_mask(cat_mask)
        print(f"✅ Reconversion: {reconstructed_mask.shape}")
        print(f"   Masques identiques: {np.array_equal(proc_mask, reconstructed_mask)}")
        
        # Test du batch preprocessing
        images_batch = [train_loader[i][0] for i in range(min(4, len(train_loader)))]
        masks_batch = [train_loader[i][1] for i in range(min(4, len(train_loader)))]
        
        proc_images, proc_masks = preprocessor.preprocess_batch(images_batch, masks_batch)
        print(f"\n✅ Batch preprocessing:")
        print(f"  - Images batch: {proc_images.shape}")
        print(f"  - Masks batch: {proc_masks.shape}")
        
        print("\n✅ Test du Preprocessing réussi!")
        
    except Exception as e:
        print(f"\n❌ Erreur dans le Preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # 3. TEST DE L'AUGMENTATION
    # ============================================================================
    print("\n🎨 PHASE 3: Test de l'Augmentation")
    print("-" * 100)
    
    try:
        # Test de l'augmentation simple
        print("\n🔄 Test de l'augmentation simple...")
        simple_augmentor = DataAugmentorSimple()
        
        aug_image, aug_mask = simple_augmentor.augment(proc_image, proc_mask)
        print(f"✅ Augmentation simple appliquée")
        print(f"  - Image augmentée: {aug_image.shape}")
        print(f"  - Mask augmenté: {aug_mask.shape}")
        
        # Visualiser l'augmentation
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(preprocessor.denormalize_image(proc_image))
        axes[0, 0].set_title("Image Originale")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(proc_mask, cmap='tab10')
        axes[0, 1].set_title("Masque Original")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(preprocessor.denormalize_image(aug_image))
        axes[1, 0].set_title("Image Augmentée")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(aug_mask, cmap='tab10')
        axes[1, 1].set_title("Masque Augmenté")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, 'test_augmentation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualisation sauvegardée: {save_path}")
        plt.close()
        
        # Test de l'augmentation avancée (si Albumentations disponible)
        print("\n🔄 Test de l'augmentation avancée...")
        try:
            advanced_augmentor = AdvancedDataAugmentor(use_albumentation=True)
            aug_adv_image, aug_adv_mask = advanced_augmentor.augment_with_albumentation(
                preprocessor.denormalize_image(proc_image), proc_mask
            )
            print(f"✅ Augmentation avancée (Albumentations) appliquée")
        except Exception as e:
            print(f"⚠️  Albumentations non disponible: {e}")
            print("   Utilisation de l'augmentation simple uniquement")
        
        print("\n✅ Test de l'Augmentation réussi!")
        
    except Exception as e:
        print(f"\n❌ Erreur dans l'Augmentation: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # RÉSUMÉ FINAL
    # ============================================================================
    print("\n" + "=" * 100)
    print("✅ TOUS LES TESTS DU PREPROCESSING SONT RÉUSSIS!")
    print("=" * 100)
    
    print(f"\n📊 Résumé du dataset:")
    print(f"  - Train: {len(train_loader)} images")
    print(f"  - Val: {len(val_loader)} images")
    print(f"  - Test: {len(test_loader)} images")
    print(f"  - Total: {len(train_loader) + len(val_loader) + len(test_loader)} images")
    
    print(f"\n🎯 Configuration:")
    print(f"  - Taille cible des images: {IMG_SIZE}")
    print(f"  - Nombre de classes: {NUM_CLASSES}")
    print(f"  - Batch size: {BATCH_SIZE}")
    
    print(f"\n📁 Fichiers générés dans: {FIGURES_DIR}")
    print("  - test_sample_original.png")
    print("  - class_distribution_train.png")
    print("  - class_distribution_val.png")
    print("  - class_distribution_test.png")
    print("  - test_augmentation.png")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    # IMPORTANT: Modifier ce chemin selon votre configuration
    # Chemin vers le dossier contenant train/val/test
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"  # À MODIFIER !
    
    # Vérifier que le chemin existe
    if not os.path.exists(DATA_PATH):
        print("\n" + "=" * 100)
        print("❌ ERREUR: Le chemin du dataset n'existe pas!")
        print("=" * 100)
        print(f"\nChemin spécifié: {DATA_PATH}")
        print("\nVeuillez modifier la variable DATA_PATH dans ce script avec le bon chemin.")
        print("\nLe chemin doit pointer vers le dossier contenant:")
        print("  - train/")
        print("  - val/")
        print("  - test/")
        print("\n" + "=" * 100)
    else:
        # Créer les dossiers de résultats
        create_directories()
        
        # Configurer les seeds
        set_seeds(RANDOM_SEED)
        
        # Lancer les tests
        test_complete_preprocessing_pipeline(DATA_PATH)
