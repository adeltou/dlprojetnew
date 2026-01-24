"""
Configuration des chemins vers les modèles entraînés
Modifiez ce fichier pour pointer vers vos modèles entraînés
"""

from pathlib import Path

# ============================================================================
# CONFIGURATION DES CHEMINS - MODIFIEZ ICI SELON VOTRE INSTALLATION
# ============================================================================

# Chemin vers le dossier contenant les modèles U-Net et Hybrid (.keras)
MODELS_DIRECTORY = Path("C:/Users/DELL/Desktop/deeplearningproject-xo/modelss")

# Chemin vers le dossier contenant les poids YOLO (.pt)
YOLO_WEIGHTS_DIRECTORY = Path("C:/Users/DELL/Desktop/deeplearningproject-xo/yolo_temp_results/yolo_100img_20260115_215355/weights")

# ============================================================================
# NOMS DES FICHIERS DE MODÈLES
# ============================================================================

# Fichier du modèle U-Net
UNET_MODEL_FILENAME = "unet_100img_20260109_232107.keras"

# Fichier du modèle Hybrid
HYBRID_MODEL_FILENAME = "hybrid_100img_20260110_231216.keras"

# Fichier du modèle YOLO (best.pt ou last.pt)
YOLO_MODEL_FILENAME = "best.pt"

# ============================================================================
# CHEMINS COMPLETS (calculés automatiquement)
# ============================================================================

UNET_MODEL_PATH = MODELS_DIRECTORY / UNET_MODEL_FILENAME
HYBRID_MODEL_PATH = MODELS_DIRECTORY / HYBRID_MODEL_FILENAME
YOLO_MODEL_PATH = YOLO_WEIGHTS_DIRECTORY / YOLO_MODEL_FILENAME

# ============================================================================
# FONCTION DE VÉRIFICATION
# ============================================================================

def check_models():
    """Vérifie si les modèles sont accessibles"""
    print("=" * 60)
    print("VÉRIFICATION DES MODÈLES")
    print("=" * 60)

    models = {
        "U-Net": UNET_MODEL_PATH,
        "Hybrid": HYBRID_MODEL_PATH,
        "YOLO": YOLO_MODEL_PATH
    }

    all_found = True
    for name, path in models.items():
        exists = path.exists()
        status = "OK" if exists else "NON TROUVÉ"
        print(f"{name:10s}: {status:12s} - {path}")
        if not exists:
            all_found = False

    print("=" * 60)

    if all_found:
        print("Tous les modèles sont accessibles!")
    else:
        print("\nCertains modèles n'ont pas été trouvés.")
        print("Vérifiez les chemins dans model_paths.py")

    return all_found


if __name__ == "__main__":
    check_models()
