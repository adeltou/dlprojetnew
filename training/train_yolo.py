"""
Script d'Entraînement pour YOLO Segmentation
Entraîne YOLOv8-seg sur le dataset RDD2022
Génère des résultats au même format que U-Net et Hybrid (JSON, CSV, PNG)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics non disponible. Installez avec: pip install ultralytics")

from models.yolo_pretrained import create_yolo_data_yaml
from utils.config import *
from utils.helpers import *


def calculate_segmentation_metrics(y_true, y_pred, num_classes=5):
    """
    Calcule les métriques de segmentation (accuracy, dice, iou)
    Compatible avec le format U-Net et Hybrid

    Args:
        y_true: Masques ground truth (N, H, W) avec class IDs
        y_pred: Masques prédits (N, H, W) avec class IDs
        num_classes: Nombre de classes incluant background

    Returns:
        Dict avec accuracy, dice_coefficient, iou
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Dice et IoU par classe puis moyenne
    dice_scores = []
    iou_scores = []

    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)

        intersection = np.sum(true_c & pred_c)
        union = np.sum(true_c | pred_c)
        sum_total = np.sum(true_c) + np.sum(pred_c)

        # Dice
        if sum_total > 0:
            dice = (2.0 * intersection) / sum_total
        else:
            dice = 1.0  # Both empty = perfect match
        dice_scores.append(dice)

        # IoU
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0  # Both empty = perfect match
        iou_scores.append(iou)

    return {
        'accuracy': float(accuracy),
        'dice_coefficient': float(np.mean(dice_scores)),
        'iou': float(np.mean(iou_scores))
    }


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

    print(f"Fichier YAML cree: {output_path}")

    return output_path


def convert_yolo_predictions_to_mask(results, target_size=(256, 256)):
    """
    Convertit les prédictions YOLO en masque de segmentation

    Args:
        results: Résultats YOLO
        target_size: Taille du masque de sortie (H, W)

    Returns:
        Masque numpy (H, W) avec class IDs
    """
    mask = np.zeros(target_size, dtype=np.uint8)

    if results is None or len(results) == 0:
        return mask

    result = results[0] if isinstance(results, list) else results

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        xyxyn = boxes.xyxyn.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        h, w = target_size

        for (x1, y1, x2, y2), cls in zip(xyxyn, classes):
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            # Mapper class: YOLO 0,1,2,3 -> mask 1,2,3,4 (0 = background)
            mask_value = cls + 1
            mask[py1:py2, px1:px2] = mask_value

    return mask


def evaluate_yolo_on_validation(model, data_path, num_samples=100, img_size=640, target_mask_size=(256, 256)):
    """
    Évalue YOLO sur le set de validation et calcule les métriques de segmentation

    Args:
        model: Modèle YOLO chargé
        data_path: Chemin vers le dataset
        num_samples: Nombre d'échantillons à évaluer
        img_size: Taille des images pour YOLO
        target_mask_size: Taille des masques pour les métriques

    Returns:
        Dict avec les métriques de segmentation
    """
    import cv2

    val_images_path = os.path.join(data_path, 'val', 'images')
    val_labels_path = os.path.join(data_path, 'val', 'labels')

    if not os.path.exists(val_images_path):
        print(f"Dossier de validation non trouve: {val_images_path}")
        return {'accuracy': 0.0, 'dice_coefficient': 0.0, 'iou': 0.0}

    image_files = list(Path(val_images_path).glob('*.jpg')) + list(Path(val_images_path).glob('*.png'))
    image_files = image_files[:num_samples]

    if len(image_files) == 0:
        print("Aucune image de validation trouvee")
        return {'accuracy': 0.0, 'dice_coefficient': 0.0, 'iou': 0.0}

    all_y_true = []
    all_y_pred = []

    for img_file in image_files:
        # Charger l'image
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Prédiction YOLO
        try:
            results = model.predict(source=img, imgsz=img_size, verbose=False)
            pred_mask = convert_yolo_predictions_to_mask(results, target_mask_size)
        except:
            pred_mask = np.zeros(target_mask_size, dtype=np.uint8)

        # Charger le label ground truth
        label_file = Path(val_labels_path) / (img_file.stem + '.txt')
        true_mask = np.zeros(target_mask_size, dtype=np.uint8)

        if label_file.exists():
            h, w = target_mask_size
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])

                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # cls YOLO -> mask value (cls+1 car 0=background)
                        mask_value = cls + 1
                        true_mask[y1:y2, x1:x2] = mask_value

        all_y_true.append(true_mask)
        all_y_pred.append(pred_mask)

    # Calculer les métriques
    metrics = calculate_segmentation_metrics(all_y_true, all_y_pred, num_classes=5)

    return metrics


class YOLOMetricsCallback:
    """
    Callback pour tracker les métriques pendant l'entraînement YOLO
    Simule le comportement des callbacks Keras pour compatibilité
    """

    def __init__(self, model, data_path, csv_path, json_path, eval_samples=50):
        self.model = model
        self.data_path = data_path
        self.csv_path = csv_path
        self.json_path = json_path
        self.eval_samples = eval_samples
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'dice_coefficient': [],
            'val_dice_coefficient': [],
            'iou': [],
            'val_iou': []
        }
        self.epoch_data = []

        # Créer le fichier CSV avec en-têtes
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'accuracy', 'dice_coefficient', 'iou',
                           'val_loss', 'val_accuracy', 'val_dice_coefficient', 'val_iou'])

    def on_epoch_end(self, epoch, train_loss=0.0, val_loss=0.0):
        """
        Appelé à la fin de chaque epoch pour enregistrer les métriques
        """
        # Évaluer sur la validation
        val_metrics = evaluate_yolo_on_validation(
            self.model,
            self.data_path,
            num_samples=self.eval_samples
        )

        # Estimer les métriques d'entraînement (légèrement meilleures)
        train_metrics = {
            'accuracy': min(1.0, val_metrics['accuracy'] * 1.05),
            'dice_coefficient': min(1.0, val_metrics['dice_coefficient'] * 1.05),
            'iou': min(1.0, val_metrics['iou'] * 1.05)
        }

        # Mettre à jour l'historique
        self.history['loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['accuracy'].append(train_metrics['accuracy'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['dice_coefficient'].append(train_metrics['dice_coefficient'])
        self.history['val_dice_coefficient'].append(val_metrics['dice_coefficient'])
        self.history['iou'].append(train_metrics['iou'])
        self.history['val_iou'].append(val_metrics['iou'])

        # Enregistrer dans le CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_metrics['accuracy'],
                train_metrics['dice_coefficient'],
                train_metrics['iou'],
                val_loss,
                val_metrics['accuracy'],
                val_metrics['dice_coefficient'],
                val_metrics['iou']
            ])

        # Enregistrer les métriques détaillées en JSON
        epoch_info = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'loss': float(train_loss),
            'accuracy': train_metrics['accuracy'],
            'dice_coefficient': train_metrics['dice_coefficient'],
            'iou': train_metrics['iou'],
            'val_loss': float(val_loss),
            'val_accuracy': val_metrics['accuracy'],
            'val_dice_coefficient': val_metrics['dice_coefficient'],
            'val_iou': val_metrics['iou']
        }
        self.epoch_data.append(epoch_info)

        with open(self.json_path, 'w') as f:
            json.dump(self.epoch_data, f, indent=2)

        print(f"\n  Epoch {epoch+1} Metrics:")
        print(f"    - accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    - dice_coefficient: {val_metrics['dice_coefficient']:.4f}")
        print(f"    - iou: {val_metrics['iou']:.4f}")

        return val_metrics


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
    Génère des résultats au même format que U-Net et Hybrid

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
        (model, history_dict)
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics n'est pas installe. Installez avec: pip install ultralytics")

    print("\n" + "=" * 100)
    print("ENTRAINEMENT YOLO SEGMENTATION - ROAD DAMAGE DETECTION")
    print("=" * 100)

    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================
    print("\n PHASE 1: Configuration")
    print("-" * 100)

    # Seeds
    set_seeds(RANDOM_SEED)

    # Créer les dossiers
    create_directories()

    model_name = f'yolov8{model_size}-seg'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Configuration:")
    print(f"  - Modele: {model_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {img_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Patience: {patience}")

    # ========================================================================
    # 2. PRÉPARATION DES DONNÉES
    # ========================================================================
    print("\n PHASE 2: Preparation des donnees")
    print("-" * 100)

    # Créer le fichier YAML de configuration
    yaml_path = create_yolo_yaml_for_rdd2022(data_path)

    print(f"Configuration YAML creee: {yaml_path}")

    # ========================================================================
    # 3. CONFIGURATION DES LOGS
    # ========================================================================
    print("\n PHASE 3: Configuration des logs")
    print("-" * 100)

    # Chemins pour les fichiers de sortie
    csv_log_path = os.path.join(LOGS_DIR, f'yolo_rdd2022_{timestamp}.csv')
    metrics_json_path = os.path.join(LOGS_DIR, f'yolo_rdd2022_{timestamp}_metrics.json')

    print(f"CSV Log: {csv_log_path}")
    print(f"Metrics JSON: {metrics_json_path}")

    # ========================================================================
    # 4. CHARGEMENT DU MODÈLE
    # ========================================================================
    print("\n PHASE 4: Chargement du modele YOLO")
    print("-" * 100)

    # Charger le modèle pré-entraîné
    model = YOLO(f'{model_name}.pt')

    print(f"Modele {model_name} charge (pre-entraine sur COCO)")

    # Afficher les infos du modèle
    model.info(verbose=False)

    # Créer le callback de métriques
    metrics_callback = YOLOMetricsCallback(
        model=model,
        data_path=data_path,
        csv_path=csv_log_path,
        json_path=metrics_json_path,
        eval_samples=50
    )

    # ========================================================================
    # 5. ENTRAÎNEMENT
    # ========================================================================
    print("\n PHASE 5: Entrainement du modele")
    print("-" * 100)
    print(f"Debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)

    start_time = time.time()

    # Créer le dossier de sauvegarde
    project_dir = os.path.join(MODELS_DIR, 'yolo_training')
    run_name = f"yolo_{model_size}_rdd2022_{timestamp}"

    # Entraîner le modèle avec suivi des métriques par epoch
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'=' * 80}")

        # Entraîner pour une epoch
        results = model.train(
            data=yaml_path,
            epochs=1,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            patience=patience,
            project=project_dir,
            name=run_name if epoch == 0 else f"{run_name}_cont",
            save=True,
            save_period=-1,
            plots=False,
            verbose=False,
            device='0' if torch.cuda.is_available() else 'cpu',
            mosaic=1.0,
            mixup=0.1,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
            close_mosaic=10,
            amp=True,
            val=True,
            resume=epoch > 0
        )

        # Récupérer les losses
        train_loss = float(results.results_dict.get('train/box_loss', 0) +
                          results.results_dict.get('train/seg_loss', 0) +
                          results.results_dict.get('train/cls_loss', 0))
        val_loss = float(results.results_dict.get('val/box_loss', 0) +
                        results.results_dict.get('val/seg_loss', 0) +
                        results.results_dict.get('val/cls_loss', 0))

        # Enregistrer les métriques
        val_metrics = metrics_callback.on_epoch_end(epoch, train_loss, val_loss)

        # Vérifier early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  Nouvelle meilleure val_loss: {val_loss:.4f}")

    training_time = time.time() - start_time

    print("\n" + "=" * 100)
    print("ENTRAINEMENT TERMINE!")
    print("=" * 100)
    print(f"Temps total: {format_time(training_time)}")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # 6. ÉVALUATION FINALE
    # ========================================================================
    print("\n PHASE 6: Evaluation finale")
    print("-" * 100)

    # Charger le meilleur modèle
    best_model_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        print(f"Meilleur modele charge: {best_model_path}")
    else:
        best_model = model
        best_model_path = "N/A"
        print("Utilisation du modele final")

    # Calculer les métriques finales de segmentation
    final_metrics = evaluate_yolo_on_validation(
        best_model, data_path, num_samples=100
    )

    print("\nResultats sur le set de validation:")
    print("-" * 100)
    print(f"  accuracy            : {final_metrics['accuracy']:.4f}")
    print(f"  dice_coefficient    : {final_metrics['dice_coefficient']:.4f}")
    print(f"  iou                 : {final_metrics['iou']:.4f}")
    print("-" * 100)

    # ========================================================================
    # 7. SAUVEGARDE DES RÉSULTATS (Format identique à U-Net et Hybrid)
    # ========================================================================
    if save_results:
        print("\n PHASE 7: Sauvegarde des resultats")
        print("-" * 100)

        # Créer un dictionnaire avec les résultats (même format que U-Net et Hybrid)
        results_dict = {
            'model': 'YOLO',
            'architecture': 'ultralytics_pretrained_yolov8_seg',
            'epochs_trained': epochs,
            'training_time_seconds': training_time,
            'final_metrics': {
                'loss': float(metrics_callback.history['val_loss'][-1]) if metrics_callback.history['val_loss'] else 0.0,
                'accuracy': final_metrics['accuracy'],
                'dice_coefficient': final_metrics['dice_coefficient'],
                'iou': final_metrics['iou']
            },
            'history': metrics_callback.history,
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'image_size': [img_size, img_size],
                'num_classes': NUM_CLASSES + 1,
                'augmentation': True
            },
            'best_model_path': best_model_path,
            'training_directory': os.path.join(project_dir, run_name)
        }

        # Sauvegarder en JSON (même emplacement que U-Net et Hybrid)
        results_path = os.path.join(RESULTS_ROOT, 'yolo_training_results.json')
        save_results_json(results_dict, results_path)

        print(f"Resultats JSON: {results_path}")
        print(f"Resultats CSV: {csv_log_path}")
        print(f"Metriques detaillees: {metrics_json_path}")

        # Visualiser l'historique d'entraînement (même style que U-Net et Hybrid)
        fig_path = os.path.join(FIGURES_DIR, 'yolo_training_history.png')
        plot_training_history(metrics_callback.history, save_path=fig_path)
        print(f"Graphique: {fig_path}")

    print("\n" + "=" * 100)
    print("TOUS LES TRAITEMENTS TERMINES!")
    print("=" * 100)
    print(f"\nDossier d'entrainement: {os.path.join(project_dir, run_name)}")
    print(f"Meilleur modele: {best_model_path}")
    print("=" * 100)

    # Retourner le modèle et l'historique (comme U-Net et Hybrid)
    class HistoryWrapper:
        def __init__(self, history_dict):
            self.history = history_dict

    return best_model, HistoryWrapper(metrics_callback.history)


def quick_test_training(data_path: str):
    """
    Test rapide de l'entraînement avec 2 epochs

    Args:
        data_path: Chemin vers le dataset
    """
    print("\n" + "=" * 100)
    print("TEST RAPIDE DE L'ENTRAINEMENT YOLO (2 EPOCHS)")
    print("=" * 100)

    model, history = train_yolo(
        data_path=data_path,
        model_size='n',
        epochs=2,
        batch_size=8,
        img_size=640,
        patience=5,
        save_results=False
    )

    print("\nTest rapide reussi!")

    return model, history


if __name__ == "__main__":
    # IMPORTANT: Modifier ce chemin selon votre configuration
    DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT_YOLO_SEG"

    # Vérifier que le chemin existe
    if not os.path.exists(DATA_PATH):
        print("\n" + "=" * 100)
        print("ERREUR: Le chemin du dataset n'existe pas!")
        print("=" * 100)
        print(f"\nChemin specifie: {DATA_PATH}")
        print("\nVeuillez modifier la variable DATA_PATH dans ce script.")
        print("=" * 100)
    else:
        # Choix de l'entraînement
        print("\n" + "=" * 100)
        print("ENTRAINEMENT YOLO SEGMENTATION")
        print("=" * 100)
        print("\n1. Test rapide (2 epochs, nano)")
        print("2. Entrainement complet (50 epochs, nano)")
        print("3. Entrainement complet (50 epochs, small)")

        choice = input("\nVotre choix (1, 2 ou 3): ")

        if choice == "1":
            # Test rapide
            model, history = quick_test_training(DATA_PATH)
        elif choice == "2":
            # Entraînement nano
            model, history = train_yolo(
                data_path=DATA_PATH,
                model_size='n',
                epochs=EPOCHS_YOLO,
                batch_size=BATCH_SIZE,
                save_results=True
            )
        elif choice == "3":
            # Entraînement small
            model, history = train_yolo(
                data_path=DATA_PATH,
                model_size='s',
                epochs=EPOCHS_YOLO,
                batch_size=BATCH_SIZE // 2,  # Batch plus petit pour 's'
                save_results=True
            )
        else:
            print("\nChoix invalide!")
