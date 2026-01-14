"""
SCRIPT PRINCIPAL D'ÉVALUATION - VERSION CORRIGÉE
Évalue les 3 modèles entraînés et génère les comparaisons

UTILISATION:
    python run_evaluation.py
"""

import numpy as np
import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration de base
IMG_SIZE = (256, 256)
NUM_CLASSES = 4  # Sans compter le background
CLASS_NAMES = {
    0: "Fissure Longitudinale",
    1: "Fissure Transversale", 
    2: "Fissure Crocodile",
    4: "Nid-de-poule"
}

print("\n" + "="*100)
print("🎯 ÉVALUATION COMPLÈTE DES MODÈLES DE SEGMENTATION")
print("="*100)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n📦 PHASE 1: Chargement des données de test")
print("-"*100)

# Chemins
DATA_PATH = "C:/Users/DELL/Desktop/dataset/RDD_SPLIT"
TEST_IMAGES_PATH = os.path.join(DATA_PATH, "test/images")
TEST_LABELS_PATH = os.path.join(DATA_PATH, "test/labels")

def load_image(image_path):
    """Charge et normalise une image"""
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def load_yolo_mask(label_path, image_shape=(256, 256)):
    """
    Charge un label YOLO et le convertit en masque de segmentation
    
    YOLO format: class_id center_x center_y width height
    YOLO classes: 0=longitudinal, 1=transversal, 2=crocodile, 4=pothole
    Mask classes: 0=background, 1=longitudinal, 2=transversal, 3=crocodile, 4=pothole
    """
    import cv2
    
    # Mapping YOLO → Mask
    YOLO_TO_MASK = {0: 1, 1: 2, 2: 3, 4: 4}
    
    # Créer masque vide (background)
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Lire les annotations YOLO
    if not os.path.exists(label_path):
        return mask
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        yolo_class = int(float(parts[0]))
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convertir en coordonnées pixel
        x1 = int((center_x - width/2) * image_shape[1])
        y1 = int((center_y - height/2) * image_shape[0])
        x2 = int((center_x + width/2) * image_shape[1])
        y2 = int((center_y + height/2) * image_shape[0])
        
        # Limiter aux frontières de l'image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_shape[1], x2)
        y2 = min(image_shape[0], y2)
        
        # Mapper la classe YOLO vers la classe du masque
        mask_class = YOLO_TO_MASK.get(yolo_class, 0)
        
        # Remplir la région (approximation rectangulaire)
        mask[y1:y2, x1:x2] = mask_class
    
    return mask

def load_test_dataset(num_samples=500):
    """Charge le dataset de test"""
    images = []
    masks = []
    filenames = []
    
    # Lister les images de test
    image_files = sorted([f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith(('.jpg', '.png'))])
    
    # Limiter le nombre d'échantillons
    image_files = image_files[:num_samples]
    
    print(f"📂 Chargement de {len(image_files)} images de test...")
    
    for i, img_file in enumerate(image_files):
        # Charger l'image
        img_path = os.path.join(TEST_IMAGES_PATH, img_file)
        img = load_image(img_path)
        images.append(img)
        
        # Charger le masque
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(TEST_LABELS_PATH, label_file)
        mask = load_yolo_mask(label_path, IMG_SIZE)
        masks.append(mask)
        
        filenames.append(img_file)
        
        if (i + 1) % 100 == 0:
            print(f"  Chargé {i + 1}/{len(image_files)} images...")
    
    print(f"✅ Dataset chargé: {len(images)} images")
    
    return np.array(images), np.array(masks), filenames

# Charger le dataset
test_images, test_masks, test_filenames = load_test_dataset(num_samples=500)

print(f"\n📊 Statistiques du dataset:")
print(f"  - Nombre d'images: {len(test_images)}")
print(f"  - Shape images: {test_images.shape}")
print(f"  - Shape masques: {test_masks.shape}")
print(f"  - Classes uniques: {np.unique(test_masks)}")

# Distribution des classes
for cls_id in range(5):
    pixels = np.sum(test_masks == cls_id)
    percentage = 100 * pixels / test_masks.size
    class_name = CLASS_NAMES.get(cls_id, f"Classe {cls_id}") if cls_id > 0 else "Background"
    print(f"  - {class_name}: {pixels} pixels ({percentage:.2f}%)")

# ============================================================================
# 2. CHARGEMENT DES MODÈLES
# ============================================================================

print("\n📦 PHASE 2: Chargement des modèles entraînés")
print("-"*100)

# Chemins des modèles entraînés (d'après vos fichiers de sortie)
MODEL_PATHS = {
    'U-Net': 'C:/Users/DELL/Desktop/results/models/unet_100img_20260109_232107.keras',
    'YOLO': 'C:/Users/DELL/Desktop/dataset/yolo_temp_results/yolo_100img7/weights/last.pt',
    'Hybrid': 'C:/Users/DELL/Desktop/results/models/hybrid_100img_20260110_231216.keras'
}

def load_models(model_paths):
    """Charge les 3 modèles"""
    import tensorflow as tf
    from tensorflow import keras
    
    models = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n📦 Chargement de {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"   ⚠️  Fichier non trouvé: {model_path}")
            print(f"   ⚠️  Ce modèle sera ignoré")
            continue
        
        try:
            if model_name == 'YOLO':
                # Charger YOLO
                from ultralytics import YOLO
                model = YOLO(model_path)
                print(f"   ✅ YOLO chargé depuis: {model_path}")
            else:
                # Charger modèles Keras avec custom objects
                custom_objects = {
                    'dice_coefficient': lambda y_true, y_pred: tf.reduce_mean(y_pred),
                    'iou_metric': lambda y_true, y_pred: tf.reduce_mean(y_pred)
                }
                
                model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print(f"   ✅ {model_name} chargé depuis: {model_path}")
            
            models[model_name] = model
            
        except Exception as e:
            print(f"   ❌ Erreur lors du chargement de {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return models

# Charger les modèles
models = load_models(MODEL_PATHS)
print(f"\n✅ {len(models)} modèles chargés avec succès")

if len(models) == 0:
    print("\n❌ Aucun modèle n'a pu être chargé. Vérifiez les chemins.")
    sys.exit(1)

# ============================================================================
# 3. ÉVALUATION DES MODÈLES
# ============================================================================

print("\n📊 PHASE 3: Évaluation des modèles")
print("-"*100)

# Importer le module de métriques corrigé
sys.path.insert(0, '/home/claude/evaluation_fixed')
from metrics_fixed import evaluate_model_on_dataset

results_dict = {}

for model_name, model in models.items():
    print(f"\n{'='*100}")
    print(f"ÉVALUATION DE {model_name.upper()}")
    print(f"{'='*100}")
    
    # Évaluer le modèle
    results = evaluate_model_on_dataset(
        model=model,
        images=test_images,
        masks_true=test_masks,
        batch_size=8,
        model_name=model_name
    )
    
    results_dict[model_name] = results
    
    # Afficher un résumé
    print(f"\n📊 Résumé {model_name}:")
    print(f"  Global:")
    print(f"    - IoU: {results['global']['iou']:.4f}")
    print(f"    - Dice: {results['global']['dice']:.4f}")
    print(f"    - Pixel Accuracy: {results['global']['pixel_accuracy']:.4f}")
    print(f"  Par classe:")
    for class_id, metrics in results['per_class'].items():
        class_name = CLASS_NAMES.get(class_id, f"Classe {class_id}") if class_id > 0 else "Background"
        print(f"    {class_name}:")
        print(f"      IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, " +
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

# ============================================================================
# 4. VISUALISATIONS
# ============================================================================

print("\n📊 PHASE 4: Génération des visualisations")
print("-"*100)

# Créer le dossier de sortie (dans le dossier du projet)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_dir = os.path.join(project_dir, "results", "evaluation")
os.makedirs(output_dir, exist_ok=True)

# 1. Comparaison globale
print("\n📊 Création du graphique de comparaison globale...")

model_names = list(results_dict.keys())
metrics_names = ['IoU', 'Dice', 'Pixel Accuracy']

iou_values = [results_dict[model]['global']['iou'] for model in model_names]
dice_values = [results_dict[model]['global']['dice'] for model in model_names]
acc_values = [results_dict[model]['global']['pixel_accuracy'] for model in model_names]

# Configuration du graphique
x = np.arange(len(metrics_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, model_name in enumerate(model_names):
    offset = (i - len(model_names)//2) * width
    values = [iou_values[i], dice_values[i], acc_values[i]]
    bars = ax.bar(x + offset, values, width, label=model_name, 
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
    
    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comparaison des Performances des Modèles', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
save_path = os.path.join(output_dir, 'comparison_global.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Sauvegardé: {save_path}")
plt.close()

# 2. Métriques par classe (IoU)
print("\n📊 Création du graphique IoU par classe...")

class_ids = [0, 1, 2, 4]
class_labels = [CLASS_NAMES.get(cid, f"Classe {cid}") if cid > 0 else "Background" for cid in class_ids]

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(class_ids))
width = 0.25

for i, model_name in enumerate(model_names):
    values = []
    for class_id in class_ids:
        # Trouver la métrique pour cette classe
        value = results_dict[model_name]['per_class'].get(class_id, {'iou': 0})['iou']
        values.append(value)
    
    offset = (i - len(model_names)//2) * width
    bars = ax.bar(x + offset, values, width, label=model_name,
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Type de Dommage', fontsize=12, fontweight='bold')
ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
ax.set_title('Performances par Classe - IoU', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(class_labels, rotation=15, ha='right')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
save_path = os.path.join(output_dir, 'comparison_iou_per_class.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Sauvegardé: {save_path}")
plt.close()

# 3. Matrices de confusion
print("\n📊 Création des matrices de confusion...")

fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 5))
if len(model_names) == 1:
    axes = [axes]

labels = ['BG', 'Long.', 'Trans.', 'Croco.', 'Pothole']

for idx, model_name in enumerate(model_names):
    conf_matrix = np.array(results_dict[model_name]['confusion_matrix'])
    
    # Normaliser
    conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
    
    sns.heatmap(conf_matrix_norm,
               annot=True,
               fmt='.2f',
               cmap='Blues',
               xticklabels=labels,
               yticklabels=labels,
               ax=axes[idx],
               cbar_kws={'label': 'Proportion'},
               vmin=0, vmax=1)
    
    axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Classe Prédite', fontsize=10)
    axes[idx].set_ylabel('Classe Vraie', fontsize=10)

plt.suptitle('Matrices de Confusion Normalisées', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

save_path = os.path.join(output_dir, 'confusion_matrices.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Sauvegardé: {save_path}")
plt.close()

# ============================================================================
# 5. SAUVEGARDE DES RÉSULTATS
# ============================================================================

print("\n💾 PHASE 5: Sauvegarde des résultats")
print("-"*100)

# Sauvegarder en JSON
results_file = os.path.join(output_dir, 'evaluation_results.json')
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)
print(f"✅ Résultats JSON sauvegardés: {results_file}")

# Créer un résumé texte
summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("RÉSUMÉ DE L'ÉVALUATION DES MODÈLES\n")
    f.write("="*100 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Nombre d'images évaluées: {len(test_images)}\n\n")
    
    f.write("PERFORMANCES GLOBALES\n")
    f.write("-"*100 + "\n\n")
    
    f.write(f"{'Modèle':<15} {'IoU':<10} {'Dice':<10} {'Pixel Accuracy':<15}\n")
    f.write("-"*100 + "\n")
    
    for model_name, results in results_dict.items():
        iou = results['global']['iou']
        dice = results['global']['dice']
        acc = results['global']['pixel_accuracy']
        f.write(f"{model_name:<15} {iou:<10.4f} {dice:<10.4f} {acc:<15.4f}\n")
    
    f.write("\n" + "="*100 + "\n")
    
    f.write("\nPERFORMANCES PAR CLASSE\n")
    f.write("-"*100 + "\n\n")
    
    for model_name, results in results_dict.items():
        f.write(f"\n{model_name}:\n")
        f.write("-"*50 + "\n")
        
        for class_id, metrics in results['per_class'].items():
            class_name = CLASS_NAMES.get(class_id, f"Classe {class_id}") if class_id > 0 else "Background"
            f.write(f"  {class_name:<20} IoU: {metrics['iou']:.4f}  Dice: {metrics['dice']:.4f}\n")

print(f"✅ Résumé texte sauvegardé: {summary_file}")

# ============================================================================
# 6. TABLEAU RÉCAPITULATIF
# ============================================================================

print("\n" + "="*100)
print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("="*100)

print(f"\n{'Modèle':<15} {'IoU':<10} {'Dice':<10} {'Pixel Accuracy':<15}")
print("-"*100)

for model_name, results in results_dict.items():
    iou = results['global']['iou']
    dice = results['global']['dice']
    acc = results['global']['pixel_accuracy']
    print(f"{model_name:<15} {iou:<10.4f} {dice:<10.4f} {acc:<15.4f}")

# Identifier le meilleur modèle
best_model_iou = max(results_dict.items(), key=lambda x: x[1]['global']['iou'])
best_model_dice = max(results_dict.items(), key=lambda x: x[1]['global']['dice'])

print("\n🏆 MEILLEURS MODÈLES:")
print(f"  - Meilleur IoU: {best_model_iou[0]} ({best_model_iou[1]['global']['iou']:.4f})")
print(f"  - Meilleur Dice: {best_model_dice[0]} ({best_model_dice[1]['global']['dice']:.4f})")

print("\n" + "="*100)
print("✅ ÉVALUATION COMPLÈTE TERMINÉE !")
print("="*100)

print(f"\n📁 Tous les résultats sont dans: {output_dir}")
print("\n📊 Graphiques créés:")
print("  - comparison_global.png")
print("  - comparison_iou_per_class.png")
print("  - confusion_matrices.png")

print("\n✅ Vous pouvez maintenant utiliser ces résultats pour votre rapport et présentation!")
