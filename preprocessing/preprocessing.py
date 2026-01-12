"""
Preprocessing des images et masques pour l'entraînement
Normalisation, redimensionnement, et préparation des batches
"""

import numpy as np
import cv2
from typing import Tuple, List
import tensorflow as tf

# Import de la configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *


class ImagePreprocessor:
    """
    Classe pour le prétraitement des images et masques
    """
    
    def __init__(self, target_size: Tuple[int, int] = IMG_SIZE, normalize: bool = True):
        """
        Initialise le préprocesseur
        
        Args:
            target_size: Taille cible (height, width)
            normalize: Si True, normalise les images entre 0 et 1
        """
        self.target_size = target_size
        self.normalize = normalize
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionne une image
        
        Args:
            image: Image à redimensionner
            
        Returns:
            Image redimensionnée
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Redimensionne un masque (utilise INTER_NEAREST pour préserver les labels)
        
        Args:
            mask: Masque à redimensionner
            
        Returns:
            Masque redimensionné
        """
        return cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise une image entre 0 et 1
        
        Args:
            image: Image à normaliser (0-255)
            
        Returns:
            Image normalisée (0-1)
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Dénormalise une image (0-1) vers (0-255)
        
        Args:
            image: Image normalisée (0-1)
            
        Returns:
            Image (0-255) en uint8
        """
        return (image * 255.0).astype(np.uint8)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline complet de prétraitement d'une image
        
        Args:
            image: Image originale
            
        Returns:
            Image prétraitée
        """
        # Redimensionner
        image = self.resize_image(image)
        
        # Normaliser si demandé
        if self.normalize:
            image = self.normalize_image(image)
        
        return image
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Pipeline complet de prétraitement d'un masque
        
        Args:
            mask: Masque original
            
        Returns:
            Masque prétraité
        """
        # Redimensionner
        mask = self.resize_mask(mask)
        
        return mask
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prétraite un batch d'images et de masques
        
        Args:
            images: Liste d'images
            masks: Liste de masques
            
        Returns:
            (images_batch, masks_batch) - Arrays numpy
        """
        processed_images = []
        processed_masks = []
        
        for image, mask in zip(images, masks):
            # Prétraiter l'image et le masque
            proc_image = self.preprocess_image(image)
            proc_mask = self.preprocess_mask(mask)
            
            processed_images.append(proc_image)
            processed_masks.append(proc_mask)
        
        # Convertir en arrays numpy
        images_batch = np.array(processed_images, dtype=np.float32)
        masks_batch = np.array(processed_masks, dtype=np.uint8)
        
        return images_batch, masks_batch
    
    def mask_to_categorical(self, mask: np.ndarray, num_classes: int = NUM_CLASSES + 1) -> np.ndarray:
        """
        Convertit un masque en format categorical (one-hot encoding)
        
        Args:
            mask: Masque avec class IDs (H, W)
            num_classes: Nombre total de classes (incluant le background)
            
        Returns:
            Masque categorical (H, W, num_classes)
        """
        # S'assurer que le masque est de la bonne forme
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # One-hot encoding
        categorical_mask = tf.keras.utils.to_categorical(mask, num_classes=num_classes)
        
        return categorical_mask
    
    def categorical_to_mask(self, categorical_mask: np.ndarray) -> np.ndarray:
        """
        Convertit un masque categorical en masque avec class IDs
        
        Args:
            categorical_mask: Masque categorical (H, W, num_classes)
            
        Returns:
            Masque avec class IDs (H, W)
        """
        return np.argmax(categorical_mask, axis=-1).astype(np.uint8)


class DataAugmentorSimple:
    """
    Augmentation simple de données sans rotation complexe
    (pour ne pas déformer les annotations)
    """
    
    def __init__(self, horizontal_flip: bool = True, 
                 brightness_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initialise l'augmenteur
        
        Args:
            horizontal_flip: Activer le flip horizontal
            brightness_range: Plage de modification de luminosité
        """
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
    
    def flip_horizontal(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flip horizontal d'une image et son masque
        
        Args:
            image: Image
            mask: Masque
            
        Returns:
            (image_flipped, mask_flipped)
        """
        image_flipped = cv2.flip(image, 1)  # 1 = horizontal flip
        mask_flipped = cv2.flip(mask, 1)
        return image_flipped, mask_flipped
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Ajuste la luminosité d'une image
        
        Args:
            image: Image (0-1 si normalisée, 0-255 sinon)
            factor: Facteur de luminosité (1.0 = pas de changement)
            
        Returns:
            Image avec luminosité ajustée
        """
        # Vérifier si l'image est normalisée
        is_normalized = image.max() <= 1.0
        
        if is_normalized:
            # Image déjà normalisée
            adjusted = image * factor
            adjusted = np.clip(adjusted, 0, 1)
        else:
            # Image en 0-255
            adjusted = image * factor
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def augment(self, image: np.ndarray, mask: np.ndarray, 
                apply_flip: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique l'augmentation à une image et son masque
        
        Args:
            image: Image
            mask: Masque
            apply_flip: Si None, décide aléatoirement
            
        Returns:
            (image_augmented, mask_augmented)
        """
        aug_image = image.copy()
        aug_mask = mask.copy()
        
        # Flip horizontal (50% de chance)
        if apply_flip is None:
            apply_flip = np.random.rand() < 0.5
        
        if self.horizontal_flip and apply_flip:
            aug_image, aug_mask = self.flip_horizontal(aug_image, aug_mask)
        
        # Ajustement de luminosité
        brightness_factor = np.random.uniform(*self.brightness_range)
        aug_image = self.adjust_brightness(aug_image, brightness_factor)
        
        return aug_image, aug_mask


def create_tf_dataset(images: np.ndarray, masks: np.ndarray, 
                     batch_size: int = BATCH_SIZE,
                     shuffle: bool = True,
                     augment: bool = False) -> tf.data.Dataset:
    """
    Crée un tf.data.Dataset pour l'entraînement
    
    Args:
        images: Array d'images (N, H, W, C)
        masks: Array de masques (N, H, W)
        batch_size: Taille du batch
        shuffle: Si True, mélange les données
        augment: Si True, applique l'augmentation
        
    Returns:
        tf.data.Dataset
    """
    # Créer le dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    # Mélanger si demandé
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    # Batcher
    dataset = dataset.batch(batch_size)
    
    # Prefetch pour améliorer les performances
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def test_preprocessing():
    """
    Fonction de test du preprocessing
    """
    print("=" * 80)
    print("TEST DU PREPROCESSING")
    print("=" * 80)
    
    # Créer une image et un masque de test
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 5, (720, 1280), dtype=np.uint8)
    
    print(f"\n📊 Image originale: {test_image.shape}")
    print(f"📊 Masque original: {test_mask.shape}")
    
    # Test du préprocesseur
    preprocessor = ImagePreprocessor(target_size=IMG_SIZE, normalize=True)
    
    # Prétraiter
    proc_image = preprocessor.preprocess_image(test_image)
    proc_mask = preprocessor.preprocess_mask(test_mask)
    
    print(f"\n✅ Image prétraitée: {proc_image.shape}")
    print(f"   - Min: {proc_image.min():.3f}, Max: {proc_image.max():.3f}")
    print(f"✅ Masque prétraité: {proc_mask.shape}")
    print(f"   - Classes uniques: {np.unique(proc_mask)}")
    
    # Test de la conversion categorical
    cat_mask = preprocessor.mask_to_categorical(proc_mask)
    print(f"\n✅ Masque categorical: {cat_mask.shape}")
    
    # Test de l'augmentation
    augmentor = DataAugmentorSimple()
    aug_image, aug_mask = augmentor.augment(proc_image, proc_mask)
    print(f"\n✅ Augmentation appliquée")
    print(f"   - Image augmentée: {aug_image.shape}")
    print(f"   - Masque augmenté: {aug_mask.shape}")
    
    print("\n" + "=" * 80)
    print("✅ Tous les tests de preprocessing sont passés!")
    print("=" * 80)


if __name__ == "__main__":
    test_preprocessing()
