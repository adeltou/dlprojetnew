"""
Modèle YOLO pour la Segmentation - Utilisant Ultralytics
Méthode 2 : Utilisation de fonctions prédéfinies (YOLOv8-seg)
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ Ultralytics non disponible. Installez avec: pip install ultralytics")

# Import de la configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *


class YOLOSegmentation:
    """
    Classe wrapper pour YOLOv8 Segmentation
    Utilise les fonctions prédéfinies d'Ultralytics
    """
    
    def __init__(self, 
                 model_name: str = 'yolov8n-seg.pt',
                 num_classes: int = NUM_CLASSES,
                 img_size: int = 640):
        """
        Initialise le modèle YOLO pour la segmentation
        
        Args:
            model_name: Nom du modèle YOLO ('yolov8n-seg.pt', 'yolov8s-seg.pt', etc.)
            num_classes: Nombre de classes (sans background)
            img_size: Taille des images pour YOLO
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics n'est pas installé. Installez avec: pip install ultralytics")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
        print(f"✅ YOLOSegmentation initialisé")
        print(f"   - Modèle: {model_name}")
        print(f"   - Classes: {num_classes}")
        print(f"   - Image size: {img_size}")
    
    def load_pretrained(self):
        """
        Charge le modèle YOLO pré-entraîné
        
        Returns:
            Modèle YOLO chargé
        """
        try:
            self.model = YOLO(self.model_name)
            print(f"✅ Modèle pré-entraîné chargé: {self.model_name}")
            return self.model
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def prepare_for_training(self, data_yaml_path: str):
        """
        Prépare le modèle pour le fine-tuning sur RDD2022
        
        Args:
            data_yaml_path: Chemin vers le fichier de configuration YAML du dataset
            
        Returns:
            Modèle prêt pour l'entraînement
        """
        if self.model is None:
            self.load_pretrained()
        
        print(f"✅ Modèle prêt pour le fine-tuning")
        print(f"   - Dataset config: {data_yaml_path}")
        
        return self.model
    
    def train(self,
              data_yaml: str,
              epochs: int = EPOCHS_YOLO,
              batch: int = BATCH_SIZE,
              imgsz: int = None,
              project: str = MODELS_DIR,
              name: str = 'yolo_rdd2022',
              **kwargs):
        """
        Entraîne le modèle YOLO sur le dataset RDD2022
        
        Args:
            data_yaml: Chemin vers le fichier YAML du dataset
            epochs: Nombre d'epochs
            batch: Taille du batch
            imgsz: Taille des images (défaut: self.img_size)
            project: Dossier de sauvegarde
            name: Nom de l'expérience
            **kwargs: Arguments supplémentaires pour YOLO.train()
            
        Returns:
            Résultats de l'entraînement
        """
        if self.model is None:
            self.load_pretrained()
        
        if imgsz is None:
            imgsz = self.img_size
        
        print("\n" + "=" * 80)
        print("ENTRAÎNEMENT YOLO SEGMENTATION")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch}")
        print(f"  - Image size: {imgsz}")
        print(f"  - Project: {project}")
        print(f"  - Name: {name}")
        print("=" * 80 + "\n")
        
        # Entraîner le modèle
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=name,
            patience=15,  # Early stopping
            save=True,
            plots=True,
            **kwargs
        )
        
        print("\n✅ Entraînement terminé!")
        
        return results
    
    def predict(self,
                source: Union[str, np.ndarray, List],
                conf: float = YOLO_CONFIG['conf_threshold'],
                iou: float = YOLO_CONFIG['iou_threshold'],
                imgsz: int = None,
                save: bool = False,
                **kwargs):
        """
        Effectue des prédictions avec le modèle YOLO
        
        Args:
            source: Image(s) à prédire (chemin, array numpy, ou liste)
            conf: Seuil de confiance
            iou: Seuil IoU pour NMS
            imgsz: Taille des images
            save: Sauvegarder les résultats
            **kwargs: Arguments supplémentaires
            
        Returns:
            Résultats de prédiction
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez load_pretrained() d'abord.")
        
        if imgsz is None:
            imgsz = self.img_size
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            **kwargs
        )
        
        return results
    
    def validate(self, data_yaml: str = None, **kwargs):
        """
        Valide le modèle sur le dataset de validation
        
        Args:
            data_yaml: Chemin vers le fichier YAML du dataset
            **kwargs: Arguments supplémentaires
            
        Returns:
            Métriques de validation
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez load_pretrained() d'abord.")
        
        print("\n🔍 Validation du modèle...")
        
        metrics = self.model.val(data=data_yaml, **kwargs)
        
        print("✅ Validation terminée")
        
        return metrics
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        Exporte le modèle dans différents formats
        
        Args:
            format: Format d'export ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Arguments supplémentaires
            
        Returns:
            Chemin du modèle exporté
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez load_pretrained() d'abord.")
        
        print(f"\n📦 Export du modèle en format {format}...")
        
        path = self.model.export(format=format, **kwargs)
        
        print(f"✅ Modèle exporté: {path}")
        
        return path
    
    def get_model_info(self):
        """
        Affiche les informations du modèle
        
        Returns:
            Dictionnaire avec les infos du modèle
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez load_pretrained() d'abord.")
        
        info = self.model.info(verbose=True)
        
        return info
    
    def convert_masks_to_segmentation(self, results, target_size: Tuple[int, int] = IMG_SIZE):
        """
        Convertit les masques YOLO en format de segmentation compatible avec U-Net
        
        Args:
            results: Résultats de prédiction YOLO
            target_size: Taille cible (height, width)
            
        Returns:
            Masque de segmentation (H, W) avec class IDs
        """
        masks_list = []
        
        for result in results:
            if result.masks is None:
                # Pas de masques détectés
                mask = np.zeros(target_size, dtype=np.uint8)
            else:
                # Créer un masque vide
                mask = np.zeros(target_size, dtype=np.uint8)
                
                # Récupérer les masques et les classes
                masks = result.masks.data.cpu().numpy()  # (N, H, W)
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                # Redimensionner et fusionner les masques
                for i, (m, cls) in enumerate(zip(masks, classes)):
                    # Redimensionner le masque
                    m_resized = cv2.resize(m, (target_size[1], target_size[0]))
                    m_binary = (m_resized > 0.5).astype(np.uint8)
                    
                    # Mapper class_id: 0->1, 1->2, 2->3, 4->4
                    if cls == 4:
                        mask_value = 4
                    else:
                        mask_value = cls + 1
                    
                    # Appliquer au masque final
                    mask[m_binary > 0] = mask_value
            
            masks_list.append(mask)
        
        if len(masks_list) == 1:
            return masks_list[0]
        
        return np.array(masks_list)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def create_yolo_data_yaml(train_path: str,
                         val_path: str,
                         test_path: str = None,
                         output_path: str = "rdd2022.yaml"):
    """
    Crée le fichier YAML de configuration pour YOLO
    
    Args:
        train_path: Chemin vers les données d'entraînement
        val_path: Chemin vers les données de validation
        test_path: Chemin vers les données de test (optionnel)
        output_path: Chemin de sauvegarde du YAML
        
    Returns:
        Chemin du fichier YAML créé
    """
    import yaml
    
    # Configuration pour RDD2022
    data_config = {
        'path': os.path.dirname(train_path),  # Chemin racine
        'train': os.path.basename(train_path),
        'val': os.path.basename(val_path),
        'nc': NUM_CLASSES,  # Nombre de classes (sans background)
        'names': {
            0: 'Longitudinal',
            1: 'Transverse',
            2: 'Crocodile',
            3: 'Pothole'  # Note: class 4 devient 3 pour YOLO
        }
    }
    
    if test_path:
        data_config['test'] = os.path.basename(test_path)
    
    # Sauvegarder le YAML
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Fichier YAML créé: {output_path}")
    
    return output_path


def create_yolo_model(model_size: str = 'n', pretrained: bool = True):
    """
    Fonction helper pour créer rapidement un modèle YOLO
    
    Args:
        model_size: Taille du modèle ('n', 's', 'm', 'l', 'x')
        pretrained: Si True, charge les poids pré-entraînés
        
    Returns:
        Instance de YOLOSegmentation
    """
    model_name = f'yolov8{model_size}-seg.pt' if pretrained else f'yolov8{model_size}-seg.yaml'
    
    yolo = YOLOSegmentation(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        img_size=YOLO_CONFIG['img_size']
    )
    
    if pretrained:
        yolo.load_pretrained()
    
    return yolo


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_yolo():
    """
    Fonction de test du modèle YOLO
    """
    print("\n" + "=" * 100)
    print("TEST DU MODÈLE YOLO SEGMENTATION")
    print("=" * 100)
    
    if not ULTRALYTICS_AVAILABLE:
        print("❌ Ultralytics non disponible. Installez avec: pip install ultralytics")
        return
    
    # Créer le modèle
    print("\n🏗️  Création du modèle YOLO...")
    yolo = create_yolo_model(model_size='n', pretrained=True)
    
    # Informations sur le modèle
    print("\n📊 Informations du modèle:")
    print("-" * 100)
    yolo.get_model_info()
    
    # Test de prédiction avec une image aléatoire
    print("\n🧪 Test de prédiction...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    try:
        results = yolo.predict(source=test_image, conf=0.25, save=False)
        print("✅ Prédiction réussie!")
        print(f"  - Nombre de résultats: {len(results)}")
        
        # Convertir en masque de segmentation
        mask = yolo.convert_masks_to_segmentation(results, target_size=(256, 256))
        print(f"  - Masque converti: {mask.shape}")
        print(f"  - Classes présentes: {np.unique(mask)}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("✅ TEST DE YOLO TERMINÉ")
    print("=" * 100)
    
    print("\n💡 Pour entraîner YOLO sur RDD2022:")
    print("   1. Créer le fichier YAML avec create_yolo_data_yaml()")
    print("   2. Appeler yolo.train(data_yaml='rdd2022.yaml')")
    print("\n" + "=" * 100)
    
    return yolo


if __name__ == "__main__":
    # Tester le modèle YOLO
    model = test_yolo()
