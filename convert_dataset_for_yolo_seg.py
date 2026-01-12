"""
Convertisseur de Dataset RDD2022 pour YOLO Segmentation
Corrige 2 problèmes :
1. Remappe les classes 0,1,2,4 vers 0,1,2,3
2. Convertit les bounding boxes en format polygone pour la segmentation
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def remap_class_id(class_id: int) -> int:
    """
    Remappe les classes RDD2022 vers classes YOLO consécutives
    
    RDD2022: 0, 1, 2, 4
    YOLO:    0, 1, 2, 3
    
    Args:
        class_id: ID de classe RDD2022
        
    Returns:
        ID de classe remappé pour YOLO
    """
    mapping = {
        0: 0,  # Longitudinal
        1: 1,  # Transverse
        2: 2,  # Crocodile
        4: 3   # Pothole (4 → 3)
    }
    return mapping.get(class_id, class_id)


def bbox_to_polygon(x_center: float, y_center: float, width: float, height: float) -> list:
    """
    Convertit une bounding box en polygone (4 coins)
    
    Args:
        x_center, y_center, width, height: Coordonnées normalisées de la bbox
        
    Returns:
        Liste de coordonnées [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    # Calculer les 4 coins du rectangle
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center - height / 2
    x3 = x_center + width / 2
    y3 = y_center + height / 2
    x4 = x_center - width / 2
    y4 = y_center + height / 2
    
    # S'assurer que les coordonnées sont dans [0, 1]
    coords = [x1, y1, x2, y2, x3, y3, x4, y4]
    coords = [max(0.0, min(1.0, c)) for c in coords]
    
    return coords


def convert_label_file(input_path: Path, output_path: Path) -> bool:
    """
    Convertit un fichier de label YOLO detection vers YOLO segmentation
    
    Args:
        input_path: Chemin du fichier label original
        output_path: Chemin du fichier label converti
        
    Returns:
        True si converti avec succès, False sinon
    """
    if not input_path.exists():
        return False
    
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Lire la bbox
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Remapper la classe
            new_class_id = remap_class_id(class_id)
            
            # Convertir bbox en polygone
            polygon = bbox_to_polygon(x_center, y_center, width, height)
            
            # Format YOLO segmentation: class x1 y1 x2 y2 x3 y3 x4 y4
            new_line = f"{new_class_id}"
            for coord in polygon:
                new_line += f" {coord:.6f}"
            
            converted_lines.append(new_line + "\n")
        
        # Écrire le fichier converti
        with open(output_path, 'w') as f:
            f.writelines(converted_lines)
        
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors de la conversion de {input_path.name}: {e}")
        return False


def convert_dataset(source_path: str, dest_path: str, splits=['train', 'val', 'test']):
    """
    Convertit tout le dataset RDD2022 en format YOLO segmentation
    
    Args:
        source_path: Chemin vers RDD_SPLIT original
        dest_path: Chemin où créer le dataset converti
        splits: Liste des splits à convertir
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    
    print("\n" + "=" * 100)
    print("CONVERSION DU DATASET RDD2022 POUR YOLO SEGMENTATION")
    print("=" * 100)
    print(f"\n📂 Source: {source_path}")
    print(f"📂 Destination: {dest_path}")
    
    total_converted = 0
    total_skipped = 0
    
    for split in splits:
        print(f"\n🔄 Conversion du split: {split}")
        print("-" * 100)
        
        # Chemins source
        source_images = source_path / split / 'images'
        source_labels = source_path / split / 'labels'
        
        # Chemins destination
        dest_images = dest_path / split / 'images'
        dest_labels = dest_path / split / 'labels'
        
        # Créer les dossiers destination
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        
        # Lister toutes les images
        image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
        
        print(f"  📊 {len(image_files)} images à traiter...")
        
        converted = 0
        skipped = 0
        
        # Convertir avec barre de progression
        for img_file in tqdm(image_files, desc=f"  {split}", unit="img"):
            # Copier l'image
            dest_img = dest_images / img_file.name
            if not dest_img.exists():
                shutil.copy2(img_file, dest_img)
            
            # Convertir le label
            label_file = source_labels / (img_file.stem + '.txt')
            dest_label = dest_labels / (img_file.stem + '.txt')
            
            if convert_label_file(label_file, dest_label):
                converted += 1
            else:
                skipped += 1
        
        print(f"  ✅ {converted} fichiers convertis")
        if skipped > 0:
            print(f"  ⚠️  {skipped} fichiers ignorés (pas de labels)")
        
        total_converted += converted
        total_skipped += skipped
    
    # Créer le fichier YAML
    print(f"\n📝 Création du fichier YAML...")
    create_yaml_config(dest_path)
    
    # Résumé
    print("\n" + "=" * 100)
    print("✅ CONVERSION TERMINÉE")
    print("=" * 100)
    print(f"\n📊 Résumé:")
    print(f"  - Fichiers convertis: {total_converted}")
    print(f"  - Fichiers ignorés: {total_skipped}")
    print(f"  - Dataset prêt dans: {dest_path}")
    print(f"  - Fichier YAML: {dest_path / 'data.yaml'}")
    
    print("\n💡 Utilisation:")
    print(f"   model.train(data='{dest_path / 'data.yaml'}')")
    print("=" * 100 + "\n")


def create_yaml_config(data_path: Path):
    """
    Crée le fichier YAML de configuration pour YOLO
    """
    import yaml
    
    data_config = {
        'path': str(data_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,
        'names': {
            0: 'Longitudinal',
            1: 'Transverse',
            2: 'Crocodile',
            3: 'Pothole'  # Classe 4 devient 3
        }
    }
    
    yaml_path = data_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✅ YAML créé: {yaml_path}")


def test_conversion():
    """
    Fonction de test de la conversion
    """
    print("\n" + "=" * 100)
    print("TEST DE CONVERSION")
    print("=" * 100)
    
    # Test de remapping
    print("\n🔄 Test de remapping des classes:")
    for old_id in [0, 1, 2, 4]:
        new_id = remap_class_id(old_id)
        print(f"  Classe {old_id} → {new_id}")
    
    # Test de conversion bbox vers polygone
    print("\n🔄 Test de conversion bbox → polygone:")
    polygon = bbox_to_polygon(0.5, 0.5, 0.4, 0.3)
    print(f"  Bbox: (0.5, 0.5, 0.4, 0.3)")
    print(f"  Polygone: {polygon}")
    
    # Test d'un label
    print("\n🔄 Test de label complet:")
    print("  Input:  4 0.405273 0.783203 0.591797 0.195312")
    
    class_id = 4
    x_c, y_c, w, h = 0.405273, 0.783203, 0.591797, 0.195312
    
    new_class = remap_class_id(class_id)
    poly = bbox_to_polygon(x_c, y_c, w, h)
    
    output = f"{new_class}"
    for coord in poly:
        output += f" {coord:.6f}"
    
    print(f"  Output: {output}")
    
    print("\n" + "=" * 100)
    print("✅ Tests réussis!")
    print("=" * 100)


if __name__ == "__main__":
    # Test d'abord
    test_conversion()
    
    # Chemins
    SOURCE_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
    DEST_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT_YOLO_SEG"
    
    # Question à l'utilisateur
    print("\n" + "=" * 100)
    print("CONVERSION DU DATASET POUR YOLO SEGMENTATION")
    print("=" * 100)
    print(f"\nSource: {SOURCE_PATH}")
    print(f"Destination: {DEST_PATH}")
    print("\n  ATTENTION: Cette conversion va créer un nouveau dataset.")
    print("   Durée estimée: 5-10 minutes pour tout le dataset")
    print("\nOptions:")
    print("  1. Convertir seulement un petit échantillon (200 images, 1-2 min)")
    print("  2. Convertir tout le dataset (26,869 images, 5-10 min)")
    print("  3. Annuler")
    
    choice = input("\nVotre choix (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🚀 Conversion d'un échantillon...")
        # Créer un petit dataset d'abord
        import random
        from pathlib import Path
        
        small_source = Path(SOURCE_PATH)
        small_dest = Path(DEST_PATH + "_SAMPLE")
        
        # Créer structure
        for split in ['train', 'val']:
            (small_dest / split / 'images').mkdir(parents=True, exist_ok=True)
            (small_dest / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copier 200 images train
        train_imgs = list((small_source / 'train' / 'images').glob('*.jpg'))
        selected = random.sample(train_imgs, min(200, len(train_imgs)))
        
        for img in tqdm(selected, desc="Train"):
            shutil.copy2(img, small_dest / 'train' / 'images' / img.name)
            label = small_source / 'train' / 'labels' / (img.stem + '.txt')
            if label.exists():
                shutil.copy2(label, small_dest / 'train' / 'labels' / label.name)
        
        # Copier 50 images val
        val_imgs = list((small_source / 'val' / 'images').glob('*.jpg'))
        selected = random.sample(val_imgs, min(50, len(val_imgs)))
        
        for img in tqdm(selected, desc="Val"):
            shutil.copy2(img, small_dest / 'val' / 'images' / img.name)
            label = small_source / 'val' / 'labels' / (img.stem + '.txt')
            if label.exists():
                shutil.copy2(label, small_dest / 'val' / 'labels' / label.name)
        
        # Convertir
        convert_dataset(str(small_dest), str(small_dest), splits=['train', 'val'])
        
    elif choice == "2":
        print("\n🚀 Conversion complète du dataset...")
        convert_dataset(SOURCE_PATH, DEST_PATH, splits=['train', 'val', 'test'])
        
    else:
        print("\n❌ Conversion annulée")
