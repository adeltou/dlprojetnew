"""
Data Loader pour le dataset RDD2022
Charge les images et les annotations au format YOLO
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# Import de la configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *


class RDD2022DataLoader:
    """
    Classe pour charger et gérer le dataset RDD2022
    Format: YOLO (classe x_center y_center width height)
    """
    
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Initialise le data loader
        
        Args:
            data_path: Chemin vers le dossier contenant train/val/test
            split: 'train', 'val', ou 'test'
        """
        self.data_path = Path(data_path)
        self.split = split
        
        # Chemins vers images et labels
        self.images_path = self.data_path / split / 'images'
        self.labels_path = self.data_path / split / 'labels'
        
        # Vérifier que les dossiers existent
        if not self.images_path.exists():
            raise FileNotFoundError(f"Le dossier d'images n'existe pas: {self.images_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Le dossier de labels n'existe pas: {self.labels_path}")
        
        # Lister tous les fichiers
        self.image_files = sorted(list(self.images_path.glob('*.jpg')) + 
                                 list(self.images_path.glob('*.png')))
        
        print(f"✅ {split.upper()}: {len(self.image_files)} images trouvées")
        
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Charge une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image en RGB
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        # Convertir BGR (OpenCV) vers RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_yolo_labels(self, label_path: Path, img_height: int, img_width: int) -> List[Dict]:
        """
        Charge les annotations au format YOLO
        
        Format YOLO: class_id x_center y_center width height (valeurs normalisées 0-1)
        
        Args:
            label_path: Chemin vers le fichier de labels
            img_height: Hauteur de l'image
            img_width: Largeur de l'image
            
        Returns:
            Liste de dictionnaires contenant les annotations
        """
        annotations = []
        
        if not label_path.exists():
            return annotations  # Pas d'annotations pour cette image
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convertir de coordonnées normalisées vers pixels
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height
            
            # Calculer les coordonnées du coin supérieur gauche
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            annotation = {
                'class_id': class_id,
                'class_name': CLASS_NAMES.get(class_id, 'Unknown'),
                'bbox': [x1, y1, x2, y2],  # Format: [x1, y1, x2, y2]
                'bbox_normalized': [x_center, y_center, width, height],  # Format YOLO
            }
            annotations.append(annotation)
        
        return annotations
    
    def create_segmentation_mask(self, annotations: List[Dict], 
                                 img_height: int, img_width: int) -> np.ndarray:
        """
        Crée un masque de segmentation à partir des bounding boxes
        
        Args:
            annotations: Liste des annotations
            img_height: Hauteur de l'image
            img_width: Largeur de l'image
            
        Returns:
            Masque de segmentation (img_height, img_width) avec class_id + 1
            (0 = background, 1-4 = classes de dommages)
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class_id']
            
            # Remplir la région avec class_id + 1 (car 0 = background)
            # On mappe: 0->1, 1->2, 2->3, 4->4
            if class_id == 4:
                mask_value = 4
            else:
                mask_value = class_id + 1
                
            mask[y1:y2, x1:x2] = mask_value
        
        return mask
    
    def get_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Récupère une image, son masque et ses annotations
        
        Args:
            idx: Index de l'image
            
        Returns:
            (image, mask, annotations)
        """
        # Charger l'image
        image_path = self.image_files[idx]
        image = self.load_image(image_path)
        
        # Charger les labels
        label_path = self.labels_path / (image_path.stem + '.txt')
        annotations = self.load_yolo_labels(label_path, image.shape[0], image.shape[1])
        
        # Créer le masque de segmentation
        mask = self.create_segmentation_mask(annotations, image.shape[0], image.shape[1])
        
        return image, mask, annotations
    
    def __len__(self) -> int:
        """Retourne le nombre d'images"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Permet d'utiliser loader[idx]"""
        return self.get_item(idx)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Calcule la distribution des classes dans le dataset
        
        Returns:
            Dictionnaire {class_id: count}
        """
        class_counts = {class_id: 0 for class_id in CLASS_IDS}
        
        for idx in range(len(self)):
            _, _, annotations = self.get_item(idx)
            for ann in annotations:
                class_id = ann['class_id']
                if class_id in class_counts:
                    class_counts[class_id] += 1
        
        return class_counts
    
    def visualize_sample(self, idx: int, save_path: str = None):
        """
        Visualise un échantillon avec ses annotations
        
        Args:
            idx: Index de l'image
            save_path: Chemin pour sauvegarder la figure (optionnel)
        """
        image, mask, annotations = self.get_item(idx)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Image originale
        axes[0].imshow(image)
        axes[0].set_title(f"Image Originale - {self.split}")
        axes[0].axis('off')
        
        # Image avec bounding boxes
        image_with_boxes = image.copy()
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class_id']
            color = CLASS_COLORS[class_id]
            # Convertir RGB vers BGR pour OpenCV
            color_rgb = (color[2], color[1], color[0])
            
            # Dessiner le rectangle
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Ajouter le label
            label = ann['class_name']
            cv2.putText(image_with_boxes, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
        
        axes[1].imshow(image_with_boxes)
        axes[1].set_title(f"Détections ({len(annotations)} dommages)")
        axes[1].axis('off')
        
        # Masque de segmentation
        axes[2].imshow(mask, cmap='tab10', vmin=0, vmax=4)
        axes[2].set_title("Masque de Segmentation")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Figure sauvegardée: {save_path}")
        
        plt.show()


def test_data_loader(data_path: str):
    """
    Fonction de test du data loader
    
    Args:
        data_path: Chemin vers le dossier du dataset
    """
    print("=" * 80)
    print("TEST DU DATA LOADER RDD2022")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        print(f"\n📂 Chargement du split: {split}")
        try:
            loader = RDD2022DataLoader(data_path, split=split)
            
            # Afficher des statistiques
            print(f"  - Nombre d'images: {len(loader)}")
            
            # Charger un échantillon
            if len(loader) > 0:
                image, mask, annotations = loader[0]
                print(f"  - Shape image: {image.shape}")
                print(f"  - Shape mask: {mask.shape}")
                print(f"  - Nombre d'annotations: {len(annotations)}")
                
                # Distribution des classes
                print(f"\n  📊 Distribution des classes dans {split}:")
                class_dist = loader.get_class_distribution()
                for class_id, count in class_dist.items():
                    print(f"    - Classe {class_id} ({CLASS_NAMES[class_id]}): {count} instances")
                
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test avec un chemin d'exemple
    # MODIFIER CE CHEMIN SELON VOTRE CONFIGURATION
    data_path = "/chemin/vers/RDD_SPLIT"
    
    # Tester le data loader
    test_data_loader(data_path)
