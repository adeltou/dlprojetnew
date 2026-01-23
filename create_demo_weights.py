"""
Script pour créer des poids de démonstration pour les modèles
Ce script crée des modèles avec des poids initialisés et les sauvegarde
pour permettre de tester l'interface.

NOTE: Ces poids ne sont PAS entraînés sur RDD2022!
      Pour de vrais résultats, utilisez les scripts d'entraînement.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

def create_demo_weights():
    """Crée des poids de démonstration pour U-Net et Hybrid"""
    print("=" * 80)
    print("CRÉATION DE POIDS DE DÉMONSTRATION")
    print("=" * 80)

    try:
        import tensorflow as tf
        from models.unet_scratch import create_unet_model
        from models.hybrid_model import create_hybrid_model

        # Configuration
        INPUT_SHAPE = (256, 256, 3)
        NUM_CLASSES = 5
        MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")

        # Créer le dossier si nécessaire
        os.makedirs(MODELS_DIR, exist_ok=True)

        print("\n1. Création du modèle U-Net...")
        unet = create_unet_model(
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            compile_model=False
        )
        unet_path = os.path.join(MODELS_DIR, "unet_best.h5")
        unet.save_weights(unet_path)
        print(f"   Sauvegardé: {unet_path}")

        print("\n2. Création du modèle Hybrid...")
        hybrid = create_hybrid_model(
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            compile_model=False
        )
        hybrid_path = os.path.join(MODELS_DIR, "hybrid_best.h5")
        hybrid.save_weights(hybrid_path)
        print(f"   Sauvegardé: {hybrid_path}")

        print("\n" + "=" * 80)
        print("POIDS DE DÉMONSTRATION CRÉÉS!")
        print("=" * 80)
        print("\nATTENTION: Ces poids sont ALÉATOIRES et ne produiront pas")
        print("           de résultats de segmentation valides.")
        print("\nPour de vrais résultats, entraînez les modèles avec:")
        print("   python training/train_unet.py")
        print("   python training/train_hybrid.py")
        print("=" * 80)

        return True

    except ImportError as e:
        print(f"\nErreur: {e}")
        print("\nVeuillez installer TensorFlow:")
        print("   pip install tensorflow")
        return False
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset():
    """Vérifie si le dataset RDD2022 est présent"""
    print("\n" + "=" * 80)
    print("VÉRIFICATION DU DATASET")
    print("=" * 80)

    # Chemins possibles du dataset
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data", "RDD_SPLIT"),
        os.path.join(os.path.dirname(__file__), "RDD_SPLIT"),
        os.path.join(os.path.dirname(__file__), "data", "RDD2022"),
        "C:/Users/DELL/Desktop/dataset/RDD_SPLIT",  # Chemin Windows dans le code
    ]

    dataset_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\nDataset trouvé: {path}")

            # Vérifier les sous-dossiers
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(path, split)
                if os.path.exists(split_path):
                    images_path = os.path.join(split_path, 'images')
                    labels_path = os.path.join(split_path, 'labels')

                    n_images = len(os.listdir(images_path)) if os.path.exists(images_path) else 0
                    n_labels = len(os.listdir(labels_path)) if os.path.exists(labels_path) else 0

                    print(f"  - {split}: {n_images} images, {n_labels} labels")

            dataset_found = True
            break

    if not dataset_found:
        print("\nDataset RDD2022 non trouvé!")
        print("\nPour utiliser ce projet, vous devez:")
        print("1. Télécharger le dataset RDD2022")
        print("2. Extraire dans ./data/RDD_SPLIT/ avec la structure:")
        print("   RDD_SPLIT/")
        print("   ├── train/")
        print("   │   ├── images/")
        print("   │   └── labels/")
        print("   ├── val/")
        print("   │   ├── images/")
        print("   │   └── labels/")
        print("   └── test/")
        print("       ├── images/")
        print("       └── labels/")

    return dataset_found


def check_dependencies():
    """Vérifie les dépendances requises"""
    print("\n" + "=" * 80)
    print("VÉRIFICATION DES DÉPENDANCES")
    print("=" * 80)

    dependencies = {
        'tensorflow': 'TensorFlow (modèles de deep learning)',
        'ultralytics': 'Ultralytics (YOLO)',
        'streamlit': 'Streamlit (interface web)',
        'opencv-python': 'OpenCV (traitement d\'images)',
        'numpy': 'NumPy (calculs numériques)',
        'scipy': 'SciPy (traitement scientifique)',
    }

    missing = []

    for module, description in dependencies.items():
        try:
            __import__(module.replace('-', '_').split('-')[0])
            status = "OK"
        except ImportError:
            status = "MANQUANT"
            missing.append(module)
        print(f"  {module:20s} - {status:10s} ({description})")

    if missing:
        print(f"\nInstaller les dépendances manquantes avec:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print("\nToutes les dépendances sont installées!")

    return len(missing) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script utilitaire pour le projet RDD2022")
    parser.add_argument('--create-weights', action='store_true',
                       help='Créer des poids de démonstration')
    parser.add_argument('--check', action='store_true',
                       help='Vérifier le dataset et les dépendances')
    parser.add_argument('--all', action='store_true',
                       help='Exécuter toutes les vérifications et créer les poids')

    args = parser.parse_args()

    if args.all or args.check:
        check_dependencies()
        check_dataset()

    if args.all or args.create_weights:
        create_demo_weights()

    if not (args.create_weights or args.check or args.all):
        print("Usage: python create_demo_weights.py [--create-weights] [--check] [--all]")
        print("\nOptions:")
        print("  --create-weights : Créer des poids de démonstration")
        print("  --check         : Vérifier le dataset et les dépendances")
        print("  --all           : Tout faire")
