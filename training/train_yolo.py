"""
Script d'Entraînement pour YOLO Segmentation
Entraîne YOLOv8-seg sur le dataset RDD2022
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from datetime import datetime
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("❌ Ultralytics non disponible. Installez avec: pip install ultralytics")

from models.yolo_pretrained import create_yolo_data_yaml
from utils.config import *
from utils.helpers import *


def create_yolo_yaml_for_rdd2022(data_path: str, output_path: str = None):
    """
    Crée le fichier YAML de configuration pour YOLO
    
    Args:
        data_path: Chemin vers le dossier RDD_SPLIT
        output_path: Chemin de sortie du YAML (optionnel)
        
    Returns:
        Chemin du fichier YAML créé
    """
    import yaml
    
    if output_path is None:
        output_path = os.path.join(data_path, 'rdd2022.yaml')
    
    # Configuration pour RDD2022
    # Note: YOLO utilise des class IDs séquentiels (0, 1, 2, 3)
    # On mappe nos classes: 0->0, 1->1, 2->2, 4->3
    data_config = {
        'path': str(Path(data_path).absolute()),  # Chemin absolu
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,  # 4 classes (pas de background pour YOLO)
        'names': {
            0: 'Longitudinal',
            1: 'Transverse',
            2: 'Crocodile',
            3: 'Pothole'  # Classe 4 devient 3 pour YOLO
        }
    }
    
    # Sauvegarder le YAML
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Fichier YAML créé: {output_path}")
    
    return output_path


def train_yolo(data_path: str,
               model_size: str = 'n',
               epochs: int = EPOCHS_YOLO,
               batch_size: int = BATCH_SIZE,
               img_size: int = 640,
               learning_rate: float = LEARNING_RATE_YOLO,
               patience: int = 15,
               save_results: bool = True):
    """
    Fonction principale pour entraîner YOLO segmentation
    
    Args:
        data_path: Chemin vers le dataset RDD_SPLIT
        model_size: Taille du modèle ('n', 's', 'm', 'l', 'x')
        epochs: Nombre d'epochs
        batch_size: Taille du batch
        img_size: Taille des images pour YOLO
        learning_rate: Learning rate initial
        patience: Patience pour early stopping
        save_results: Sauvegarder les résultats
        
    Returns:
        (model, results)
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics n'est pas installé. Installez avec: pip install ultralytics")
    
    print("\n" + "=" * 100)
    print("ENTRAÎNEMENT YOLO SEGMENTATION - ROAD DAMAGE DETECTION")
    print("=" * 100)
    
    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================
    print("\n🔧 PHASE 1: Configuration")
    print("-" * 100)
    
    # Seeds
    set_seeds(RANDOM_SEED)
    
    # Créer les dossiers
    create_directories()
    
    model_name = f'yolov8{model_size}-seg'
    
    print(f"✅ Configuration:")
    print(f"  - Modèle: {model_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {img_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Patience: {patience}")
    
    # ========================================================================
    # 2. PRÉPARATION DES DONNÉES
    # ========================================================================
    print("\n📦 PHASE 2: Préparation des données")
    print("-" * 100)
    
    # Créer le fichier YAML de configuration
    yaml_path = create_yolo_yaml_for_rdd2022(data_path)
    
    print(f"✅ Configuration YAML créée: {yaml_path}")
    
    # ========================================================================
    # 3. CHARGEMENT DU MODÈLE
    # ========================================================================
    print("\n🏗️  PHASE 3: Chargement du modèle YOLO")
    print("-" * 100)
    
    # Charger le modèle pré-entraîné
    model = YOLO(f'{model_name}.pt')
    
    print(f"✅ Modèle {model_name} chargé (pré-entraîné sur COCO)")
    
    # Afficher les infos du modèle
    model.info(verbose=False)
    
    # ========================================================================
    # 4. ENTRAÎNEMENT
    # ========================================================================
    print("\n🚀 PHASE 4: Entraînement du modèle")
    print("-" * 100)
    print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    
    start_time = time.time()
    
    # Créer le dossier de sauvegarde
    project_dir = os.path.join(MODELS_DIR, 'yolo_training')
    run_name = f"yolo_{model_size}_rdd2022_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Entraîner le modèle
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        patience=patience,
        project=project_dir,
        name=run_name,
        save=True,
        save_period=10,  # Sauvegarder tous les 10 epochs
        plots=True,
        verbose=True,
        device='0' if torch.cuda.is_available() else 'cpu',
        # Augmentation
        mosaic=1.0,  # Augmentation par mosaïque
        mixup=0.1,   # Augmentation par mixup
        degrees=15.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.5,     # Scale
        fliplr=0.5,    # Flip horizontal
        flipud=0.0,    # Pas de flip vertical (routes)
        # Autres paramètres
        close_mosaic=10,  # Désactiver mosaic les 10 derniers epochs
        amp=True,  # Automatic Mixed Precision
        val=True,  # Validation pendant l'entraînement
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 100)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 100)
    print(f"Temps total: {format_time(training_time)}")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # 5. VALIDATION FINALE
    # ========================================================================
    print("\n📈 PHASE 5: Validation finale")
    print("-" * 100)
    
    # Charger le meilleur modèle
    best_model_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)
    
    print(f"✅ Meilleur modèle chargé: {best_model_path}")
    
    # Valider sur le set de test
    val_results = best_model.val(
        data=yaml_path,
        split='test',
        imgsz=img_size,
        batch=batch_size,
        verbose=True
    )
    
    print("\n📊 Résultats sur le set de test:")
    print("-" * 100)
    print(f"  Box mAP50: {val_results.box.map50:.4f}")
    print(f"  Box mAP50-95: {val_results.box.map:.4f}")
    print(f"  Mask mAP50: {val_results.seg.map50:.4f}")
    print(f"  Mask mAP50-95: {val_results.seg.map:.4f}")
    print("-" * 100)
    
    # ========================================================================
    # 6. SAUVEGARDE DES RÉSULTATS
    # ========================================================================
    if save_results:
        print("\n💾 PHASE 6: Sauvegarde des résultats")
        print("-" * 100)
        
        # Créer un dictionnaire avec les résultats
        results_dict = {
            'model': 'YOLO',
            'model_size': model_size,
            'model_name': model_name,
            'architecture': 'ultralytics_pretrained',
            'epochs_trained': epochs,
            'training_time_seconds': training_time,
            'best_model_path': best_model_path,
            'final_metrics': {
                'box_map50': float(val_results.box.map50),
                'box_map50_95': float(val_results.box.map),
                'mask_map50': float(val_results.seg.map50),
                'mask_map50_95': float(val_results.seg.map),
            },
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'image_size': img_size,
                'patience': patience
            },
            'training_directory': os.path.join(project_dir, run_name)
        }
        
        # Sauvegarder en JSON
        results_path = os.path.join(RESULTS_ROOT, 'yolo_training_results.json')
        save_results_json(results_dict, results_path)
        
        print(f"✅ Résultats sauvegardés: {results_path}")
    
    print("\n" + "=" * 100)
    print("✅ TOUS LES TRAITEMENTS TERMINÉS!")
    print("=" * 100)
    print(f"\n📁 Dossier d'entraînement: {os.path.join(project_dir, run_name)}")
    print(f"📊 Visualiser les résultats:")
    print(f"   - Graphiques dans: {os.path.join(project_dir, run_name)}")
    print(f"   - Meilleur modèle: {best_model_path}")
    print("=" * 100)
    
    return best_model, results


def quick_test_training(data_path: str):
    """
    Test rapide de l'entraînement avec 2 epochs
    
    Args:
        data_path: Chemin vers le dataset
    """
    print("\n" + "=" * 100)
    print("TEST RAPIDE DE L'ENTRAÎNEMENT YOLO (2 EPOCHS)")
    print("=" * 100)
    
    model, results = train_yolo(
        data_path=data_path,
        model_size='n',
        epochs=2,
        batch_size=8,
        img_size=640,
        patience=5,
        save_results=False
    )
    
    print("\n✅ Test rapide réussi!")
    
    return model, results


if __name__ == "__main__":
    # IMPORTANT: Modifier ce chemin selon votre configuration
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT_YOLO_SEG"
    
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
        print("ENTRAÎNEMENT YOLO SEGMENTATION")
        print("=" * 100)
        print("\n1. Test rapide (2 epochs, nano)")
        print("2. Entraînement complet (50 epochs, nano)")
        print("3. Entraînement complet (50 epochs, small)")
        
        choice = input("\nVotre choix (1, 2 ou 3): ")
        
        if choice == "1":
            # Test rapide
            model, results = quick_test_training(DATA_PATH)
        elif choice == "2":
            # Entraînement nano
            model, results = train_yolo(
                data_path=DATA_PATH,
                model_size='n',
                epochs=EPOCHS_YOLO,
                batch_size=BATCH_SIZE,
                save_results=True
            )
        elif choice == "3":
            # Entraînement small
            model, results = train_yolo(
                data_path=DATA_PATH,
                model_size='s',
                epochs=EPOCHS_YOLO,
                batch_size=BATCH_SIZE // 2,  # Batch plus petit pour 's'
                save_results=True
            )
        else:
            print("\n❌ Choix invalide!")
