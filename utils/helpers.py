"""
Fonctions utilitaires pour le projet
Visualisation, sauvegarde, statistiques
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import de la configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *

def set_seeds(seed: int = RANDOM_SEED):
    """
    Configure les seeds pour la reproductibilité
    
    Args:
        seed: Valeur du seed
    """
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"✅ Seeds configurés: {seed}")


def plot_class_distribution(class_counts: Dict[int, int], 
                           save_path: str = None,
                           title: str = "Distribution des Classes"):
    """
    Visualise la distribution des classes
    
    Args:
        class_counts: Dictionnaire {class_id: count}
        save_path: Chemin pour sauvegarder la figure
        title: Titre du graphique
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Préparer les données
    class_names = [CLASS_NAMES[class_id] for class_id in class_counts.keys()]
    counts = list(class_counts.values())
    colors = [CLASS_COLORS[class_id] for class_id in class_counts.keys()]
    # Convertir BGR vers RGB pour matplotlib
    colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
    
    # Créer le bar plot
    bars = ax.bar(class_names, counts, color=colors_rgb, alpha=0.7, edgecolor='black')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Type de Dommage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre d\'Instances', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Distribution sauvegardée: {save_path}")
    
    plt.show()


def plot_training_history(history: Dict, save_path: str = None):
    """
    Visualise l'historique d'entraînement
    
    Args:
        history: Dictionnaire contenant les métriques (loss, accuracy, etc.)
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    if 'loss' in history and 'val_loss' in history:
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Plot Accuracy/IoU
    if 'accuracy' in history and 'val_accuracy' in history:
        axes[1].plot(history['accuracy'], label='Train Acc', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    elif 'iou' in history and 'val_iou' in history:
        axes[1].plot(history['iou'], label='Train IoU', linewidth=2)
        axes[1].plot(history['val_iou'], label='Val IoU', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('IoU', fontsize=12, fontweight='bold')
        axes[1].set_title('Évolution de l\'IoU', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Historique sauvegardé: {save_path}")
    
    plt.show()


def plot_comparison_bar(results: Dict[str, Dict[str, float]], 
                       save_path: str = None,
                       title: str = "Comparaison des Modèles"):
    """
    Crée un graphique de comparaison entre modèles
    
    Args:
        results: Dictionnaire {model_name: {metric_name: value}}
        save_path: Chemin pour sauvegarder
        title: Titre du graphique
    """
    # Préparer les données
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model, 
               color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparaison sauvegardée: {save_path}")
    
    plt.show()


def save_results_json(results: Dict, filepath: str):
    """
    Sauvegarde les résultats en JSON
    
    Args:
        results: Dictionnaire des résultats
        filepath: Chemin du fichier JSON
    """
    # Ajouter un timestamp
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés: {filepath}")


def load_results_json(filepath: str) -> Dict:
    """
    Charge les résultats depuis un JSON
    
    Args:
        filepath: Chemin du fichier JSON
        
    Returns:
        Dictionnaire des résultats
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ Résultats chargés: {filepath}")
    return results


def print_summary_table(results: Dict[str, Dict[str, float]]):
    """
    Affiche un tableau récapitulatif des résultats
    
    Args:
        results: Dictionnaire {model_name: {metric_name: value}}
    """
    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF DES RÉSULTATS")
    print("=" * 80)
    
    # En-tête
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    header = f"{'Métrique':<20}"
    for model in models:
        header += f"{model:<20}"
    print(header)
    print("-" * 80)
    
    # Lignes
    for metric in metrics:
        row = f"{metric:<20}"
        for model in models:
            value = results[model][metric]
            row += f"{value:<20.4f}"
        print(row)
    
    print("=" * 80)


def create_experiment_folder(experiment_name: str) -> str:
    """
    Crée un dossier pour une expérience avec timestamp
    
    Args:
        experiment_name: Nom de l'expérience
        
    Returns:
        Chemin du dossier créé
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{experiment_name}_{timestamp}"
    folder_path = os.path.join(RESULTS_ROOT, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'logs'), exist_ok=True)
    
    print(f"✅ Dossier d'expérience créé: {folder_path}")
    return folder_path


def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en format lisible
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        String formaté (ex: "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def test_helpers():
    """
    Fonction de test des helpers
    """
    print("=" * 80)
    print("TEST DES HELPERS")
    print("=" * 80)
    
    # Test de set_seeds
    print("\n🔢 Test des seeds...")
    set_seeds(42)
    
    # Test de plot_class_distribution
    print("\n📊 Test de visualisation de distribution...")
    test_counts = {0: 150, 1: 200, 2: 100, 4: 80}
    plot_class_distribution(test_counts, title="Test Distribution")
    
    # Test de création de dossier d'expérience
    print("\n📁 Test de création de dossier d'expérience...")
    exp_folder = create_experiment_folder("test_experiment")
    
    # Test de formatage du temps
    print("\n⏱️  Test de formatage du temps...")
    print(f"  - 65 secondes = {format_time(65)}")
    print(f"  - 3665 secondes = {format_time(3665)}")
    
    print("\n" + "=" * 80)
    print("✅ Tous les tests des helpers sont passés!")
    print("=" * 80)


if __name__ == "__main__":
    test_helpers()
