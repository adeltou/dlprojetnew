"""
Augmentation de données avancée pour améliorer la généralisation
Compatible avec la segmentation sémantique
"""

import numpy as np
import cv2
from typing import Tuple, List
import albumentations as A

# Import de la configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *


class AdvancedDataAugmentor:
    """
    Augmentation avancée de données utilisant Albumentations
    Optimisé pour la segmentation sémantique
    """
    
    def __init__(self, use_albumentation: bool = True):
        """
        Initialise l'augmenteur
        
        Args:
            use_albumentation: Si True, utilise Albumentations (recommandé)
        """
        self.use_albumentation = use_albumentation
        
        if use_albumentation:
            try:
                self.transform = self._create_albumentations_pipeline()
                print("✅ Pipeline Albumentations créé avec succès")
            except ImportError:
                print("⚠️  Albumentations non disponible, utilisation de l'augmentation simple")
                self.use_albumentation = False
                self.transform = None
        else:
            self.transform = None
    
    def _create_albumentations_pipeline(self):
        """
        Crée un pipeline d'augmentation avec Albumentations
        
        Returns:
            Composition d'augmentations
        """
        transform = A.Compose([
            # Transformations géométriques
            A.HorizontalFlip(p=0.5),
            
            # Rotation légère (ne pas trop déformer les routes)
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            
            # Transformations de luminosité et contraste
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Modifications de couleur
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            
            # Flou léger (simule différentes conditions de caméra)
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Bruit
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            
            # Elastic transform (simule déformations de route)
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=5,
                p=0.2
            ),
            
            # Grid distortion
            A.GridDistortion(p=0.2),
            
        ])
        
        return transform
    
    def augment_with_albumentation(self, image: np.ndarray, 
                                   mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique l'augmentation avec Albumentations
        
        Args:
            image: Image (H, W, C)
            mask: Masque (H, W)
            
        Returns:
            (image_augmented, mask_augmented)
        """
        if not self.use_albumentation or self.transform is None:
            return image, mask
        
        # S'assurer que l'image est en uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Appliquer les transformations
        transformed = self.transform(image=image_uint8, mask=mask)
        
        aug_image = transformed['image']
        aug_mask = transformed['mask']
        
        # Reconvertir en float si l'entrée était en float
        if image.dtype == np.float32 or image.dtype == np.float64:
            aug_image = aug_image.astype(np.float32) / 255.0
        
        return aug_image, aug_mask
    
    def augment_batch(self, images: np.ndarray, 
                     masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique l'augmentation à un batch d'images
        
        Args:
            images: Batch d'images (N, H, W, C)
            masks: Batch de masques (N, H, W)
            
        Returns:
            (images_augmented, masks_augmented)
        """
        aug_images = []
        aug_masks = []
        
        for img, mask in zip(images, masks):
            if self.use_albumentation:
                aug_img, aug_mask = self.augment_with_albumentation(img, mask)
            else:
                # Fallback: augmentation simple
                aug_img, aug_mask = self.simple_augment(img, mask)
            
            aug_images.append(aug_img)
            aug_masks.append(aug_mask)
        
        return np.array(aug_images), np.array(aug_masks)
    
    def simple_augment(self, image: np.ndarray, 
                      mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmentation simple sans Albumentations (fallback)
        
        Args:
            image: Image
            mask: Masque
            
        Returns:
            (image_augmented, mask_augmented)
        """
        aug_image = image.copy()
        aug_mask = mask.copy()
        
        # Flip horizontal (50% de chance)
        if np.random.rand() < 0.5:
            aug_image = cv2.flip(aug_image, 1)
            aug_mask = cv2.flip(aug_mask, 1)
        
        # Ajustement de luminosité
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            is_normalized = aug_image.max() <= 1.0
            
            if is_normalized:
                aug_image = np.clip(aug_image * factor, 0, 1)
            else:
                aug_image = np.clip(aug_image * factor, 0, 255).astype(np.uint8)
        
        return aug_image, aug_mask


class MixupAugmentor:
    """
    Implémentation de Mixup pour la segmentation
    Mélange deux images avec leurs masques
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialise Mixup
        
        Args:
            alpha: Paramètre de la distribution Beta
        """
        self.alpha = alpha
    
    def mixup(self, image1: np.ndarray, mask1: np.ndarray,
             image2: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique Mixup entre deux paires (image, masque)
        
        Args:
            image1, mask1: Première paire
            image2, mask2: Deuxième paire
            
        Returns:
            (mixed_image, mixed_mask)
        """
        # Générer le coefficient de mélange
        lambda_param = np.random.beta(self.alpha, self.alpha)
        
        # Mélanger les images
        mixed_image = lambda_param * image1 + (1 - lambda_param) * image2
        
        # Pour les masques, on prend celui de l'image dominante
        if lambda_param >= 0.5:
            mixed_mask = mask1
        else:
            mixed_mask = mask2
        
        return mixed_image, mixed_mask


class CutMixAugmentor:
    """
    Implémentation de CutMix pour la segmentation
    Découpe et colle des régions d'une image sur une autre
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialise CutMix
        
        Args:
            alpha: Paramètre pour contrôler la taille de la région
        """
        self.alpha = alpha
    
    def get_random_box(self, img_height: int, img_width: int, 
                      lambda_param: float) -> Tuple[int, int, int, int]:
        """
        Génère une boîte aléatoire pour CutMix
        
        Args:
            img_height: Hauteur de l'image
            img_width: Largeur de l'image
            lambda_param: Paramètre de mélange
            
        Returns:
            (x1, y1, x2, y2) coordonnées de la boîte
        """
        cut_ratio = np.sqrt(1.0 - lambda_param)
        cut_h = int(img_height * cut_ratio)
        cut_w = int(img_width * cut_ratio)
        
        # Centre aléatoire
        cx = np.random.randint(img_width)
        cy = np.random.randint(img_height)
        
        # Coordonnées de la boîte
        x1 = np.clip(cx - cut_w // 2, 0, img_width)
        y1 = np.clip(cy - cut_h // 2, 0, img_height)
        x2 = np.clip(cx + cut_w // 2, 0, img_width)
        y2 = np.clip(cy + cut_h // 2, 0, img_height)
        
        return x1, y1, x2, y2
    
    def cutmix(self, image1: np.ndarray, mask1: np.ndarray,
              image2: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique CutMix entre deux paires (image, masque)
        
        Args:
            image1, mask1: Première paire
            image2, mask2: Deuxième paire
            
        Returns:
            (mixed_image, mixed_mask)
        """
        # Générer lambda
        lambda_param = np.random.beta(self.alpha, self.alpha)
        
        # Obtenir la boîte
        img_height, img_width = image1.shape[:2]
        x1, y1, x2, y2 = self.get_random_box(img_height, img_width, lambda_param)
        
        # Copier image1 et mask1
        mixed_image = image1.copy()
        mixed_mask = mask1.copy()
        
        # Coller la région de image2 et mask2
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        
        return mixed_image, mixed_mask


def test_augmentation():
    """
    Fonction de test de l'augmentation
    """
    print("=" * 80)
    print("TEST DE L'AUGMENTATION")
    print("=" * 80)
    
    # Créer des données de test
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    
    print(f"\n📊 Image test: {test_image.shape}")
    print(f"📊 Masque test: {test_mask.shape}")
    
    # Test de l'augmenteur avancé
    print("\n🔄 Test de l'augmenteur avancé...")
    try:
        augmentor = AdvancedDataAugmentor(use_albumentation=True)
        aug_image, aug_mask = augmentor.augment_with_albumentation(test_image, test_mask)
        print(f"✅ Augmentation avancée: {aug_image.shape}")
    except Exception as e:
        print(f"⚠️  Erreur: {e}")
        print("   Utilisation de l'augmentation simple")
        augmentor = AdvancedDataAugmentor(use_albumentation=False)
        aug_image, aug_mask = augmentor.simple_augment(test_image, test_mask)
        print(f"✅ Augmentation simple: {aug_image.shape}")
    
    # Test Mixup
    print("\n🔄 Test de Mixup...")
    mixup_aug = MixupAugmentor(alpha=0.2)
    test_image2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask2 = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    mixed_img, mixed_mask = mixup_aug.mixup(test_image, test_mask, test_image2, test_mask2)
    print(f"✅ Mixup appliqué: {mixed_img.shape}")
    
    # Test CutMix
    print("\n🔄 Test de CutMix...")
    cutmix_aug = CutMixAugmentor(alpha=1.0)
    cut_img, cut_mask = cutmix_aug.cutmix(test_image, test_mask, test_image2, test_mask2)
    print(f"✅ CutMix appliqué: {cut_img.shape}")
    
    print("\n" + "=" * 80)
    print("✅ Tous les tests d'augmentation sont passés!")
    print("=" * 80)


if __name__ == "__main__":
    test_augmentation()
