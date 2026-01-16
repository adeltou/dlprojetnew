"""
Interface Streamlit pour le projet de Deep Learning
Détection et Segmentation des Dommages Routiers
Master 2 HPC - 2025-2026
"""

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Projet Deep Learning - Dommages Routiers",
    page_icon="🛣️",
    layout="wide"
)

# Chemins des fichiers
BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "results" / "figures"
LOGS_DIR = BASE_DIR / "results" / "logs"

# Fichiers de métriques
UNET_METRICS = LOGS_DIR / "unet_100img_20260109_232107_metrics.json"
YOLO_METRICS = LOGS_DIR / "yolo_training_results.json"
HYBRID_METRICS = LOGS_DIR / "hybrid_100img_20260110_231216_metrics.json"


def load_metrics():
    """Charge les métriques des trois architectures"""
    metrics = {}

    # U-Net
    if UNET_METRICS.exists():
        with open(UNET_METRICS, 'r') as f:
            unet_data = json.load(f)
            metrics['U-Net'] = {
                'epochs': [d['epoch'] for d in unet_data],
                'accuracy': [d['accuracy'] for d in unet_data],
                'dice_coefficient': [d['dice_coefficient'] for d in unet_data],
                'iou': [d['iou'] for d in unet_data],
                'loss': [d['loss'] for d in unet_data],
                'val_accuracy': [d['val_accuracy'] for d in unet_data],
                'val_dice_coefficient': [d['val_dice_coefficient'] for d in unet_data],
                'val_iou': [d['val_iou'] for d in unet_data],
                'val_loss': [d['val_loss'] for d in unet_data]
            }

    # YOLO
    if YOLO_METRICS.exists():
        with open(YOLO_METRICS, 'r') as f:
            yolo_data = json.load(f)
            history = yolo_data['history']
            metrics['YOLO'] = {
                'epochs': list(range(1, len(history['accuracy']) + 1)),
                'accuracy': history['accuracy'],
                'dice_coefficient': history['dice_coefficient'],
                'iou': history['iou'],
                'loss': history['loss'],
                'val_accuracy': history['val_accuracy'],
                'val_dice_coefficient': history['val_dice_coefficient'],
                'val_iou': history['val_iou'],
                'val_loss': history['val_loss']
            }

    # Hybrid
    if HYBRID_METRICS.exists():
        with open(HYBRID_METRICS, 'r') as f:
            hybrid_data = json.load(f)
            metrics['Hybrid'] = {
                'epochs': [d['epoch'] for d in hybrid_data],
                'accuracy': [d['accuracy'] for d in hybrid_data],
                'dice_coefficient': [d['dice_coefficient'] for d in hybrid_data],
                'iou': [d['iou'] for d in hybrid_data],
                'loss': [d['loss'] for d in hybrid_data],
                'val_accuracy': [d['val_accuracy'] for d in hybrid_data],
                'val_dice_coefficient': [d['val_dice_coefficient'] for d in hybrid_data],
                'val_iou': [d['val_iou'] for d in hybrid_data],
                'val_loss': [d['val_loss'] for d in hybrid_data]
            }

    return metrics


def calculate_global_averages(metrics):
    """Calcule les moyennes globales pour chaque métrique et architecture"""
    averages = {}

    metric_names = ['accuracy', 'dice_coefficient', 'iou', 'loss',
                    'val_accuracy', 'val_dice_coefficient', 'val_iou', 'val_loss']

    for arch_name, arch_metrics in metrics.items():
        averages[arch_name] = {}
        for metric in metric_names:
            if metric in arch_metrics:
                averages[arch_name][metric] = np.mean(arch_metrics[metric])

    return averages


def display_preprocessing_results():
    """Affiche les résultats de prétraitement"""
    st.header("Résultats du Prétraitement")

    st.markdown("""
    Cette section présente les visualisations issues de la phase de prétraitement des données
    du dataset RDD2022 (Road Damage Detection).
    """)

    # Distribution des classes
    st.subheader("Distribution des Classes")

    col1, col2, col3 = st.columns(3)

    train_dist = FIGURES_DIR / "class_distribution_train.png"
    val_dist = FIGURES_DIR / "class_distribution_val.png"
    test_dist = FIGURES_DIR / "class_distribution_test.png"

    with col1:
        if train_dist.exists():
            st.image(str(train_dist), caption="Distribution - Ensemble d'entraînement", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col2:
        if val_dist.exists():
            st.image(str(val_dist), caption="Distribution - Ensemble de validation", use_container_width=True)
        else:
            st.warning("Image non disponible")

    with col3:
        if test_dist.exists():
            st.image(str(test_dist), caption="Distribution - Ensemble de test", use_container_width=True)
        else:
            st.warning("Image non disponible")

    # Augmentation et échantillons
    st.subheader("Augmentation des Données et Échantillons")

    col1, col2 = st.columns(2)

    with col1:
        augmentation_img = FIGURES_DIR / "test_augmentation.png"
        if augmentation_img.exists():
            st.image(str(augmentation_img), caption="Exemples d'augmentation de données", use_container_width=True)
        else:
            st.warning("Image d'augmentation non disponible")

    with col2:
        sample_img = FIGURES_DIR / "test_sample_original.png"
        if sample_img.exists():
            st.image(str(sample_img), caption="Échantillon original", use_container_width=True)
        else:
            st.warning("Image d'échantillon non disponible")


def display_metrics_histograms(metrics, averages):
    """Affiche les histogrammes des métriques"""
    st.header("Comparaison des Métriques - Histogrammes")

    architectures = list(averages.keys())
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    # Métriques principales pour l'histogramme de comparaison
    main_metrics = ['accuracy', 'dice_coefficient', 'iou']
    val_metrics = ['val_accuracy', 'val_dice_coefficient', 'val_iou']

    # Tableau récapitulatif des moyennes globales
    st.subheader("Moyennes Globales des Métriques")

    # Créer un DataFrame pour l'affichage
    df_data = []
    for arch in architectures:
        row = {'Architecture': arch}
        for metric in main_metrics + val_metrics + ['loss', 'val_loss']:
            if metric in averages[arch]:
                row[metric.replace('_', ' ').title()] = f"{averages[arch][metric]:.4f}"
        df_data.append(row)

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Histogrammes des métriques d'entraînement
    st.subheader("Métriques d'Entraînement (Moyennes)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(main_metrics):
        ax = axes[idx]
        values = [averages[arch][metric] for arch in architectures]
        bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur')
        ax.set_ylim(0, 1)

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Histogrammes des métriques de validation
    st.subheader("Métriques de Validation (Moyennes)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(val_metrics):
        ax = axes[idx]
        values = [averages[arch][metric] for arch in architectures]
        bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
        ax.set_title(metric.replace('val_', '').replace('_', ' ').title() + ' (Validation)',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur')
        ax.set_ylim(0, 1)

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Comparaison des pertes (Loss)
    st.subheader("Comparaison des Pertes (Loss)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss d'entraînement
    ax = axes[0]
    values = [averages[arch]['loss'] for arch in architectures]
    bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
    ax.set_title('Loss Entraînement (Moyenne)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Loss de validation
    ax = axes[1]
    values = [averages[arch]['val_loss'] for arch in architectures]
    bars = ax.bar(architectures, values, color=[colors[arch] for arch in architectures])
    ax.set_title('Loss Validation (Moyenne)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_training_evolution(metrics):
    """Affiche l'évolution des métriques pendant l'entraînement"""
    st.header("Évolution des Métriques pendant l'Entraînement")

    architectures = list(metrics.keys())
    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    # Sélection de la métrique à visualiser
    metric_options = {
        'Accuracy': 'accuracy',
        'Dice Coefficient': 'dice_coefficient',
        'IoU': 'iou',
        'Loss': 'loss',
        'Validation Accuracy': 'val_accuracy',
        'Validation Dice': 'val_dice_coefficient',
        'Validation IoU': 'val_iou',
        'Validation Loss': 'val_loss'
    }

    selected_metric_name = st.selectbox("Sélectionner la métrique", list(metric_options.keys()))
    selected_metric = metric_options[selected_metric_name]

    fig, ax = plt.subplots(figsize=(12, 6))

    for arch in architectures:
        if selected_metric in metrics[arch]:
            epochs = metrics[arch]['epochs']
            values = metrics[arch][selected_metric]
            ax.plot(epochs, values, marker='o', label=arch, color=colors[arch], linewidth=2)

    ax.set_xlabel('Époque', fontsize=12)
    ax.set_ylabel(selected_metric_name, fontsize=12)
    ax.set_title(f'Évolution de {selected_metric_name} par Époque', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_detailed_metrics(metrics):
    """Affiche les métriques détaillées par époque"""
    st.header("Métriques Détaillées par Architecture")

    architecture = st.selectbox("Sélectionner l'architecture", list(metrics.keys()))

    if architecture in metrics:
        arch_metrics = metrics[architecture]

        # Créer un DataFrame avec toutes les métriques
        df_data = {
            'Époque': arch_metrics['epochs'],
            'Accuracy': [f"{v:.4f}" for v in arch_metrics['accuracy']],
            'Dice': [f"{v:.4f}" for v in arch_metrics['dice_coefficient']],
            'IoU': [f"{v:.4f}" for v in arch_metrics['iou']],
            'Loss': [f"{v:.4f}" for v in arch_metrics['loss']],
            'Val Accuracy': [f"{v:.4f}" for v in arch_metrics['val_accuracy']],
            'Val Dice': [f"{v:.4f}" for v in arch_metrics['val_dice_coefficient']],
            'Val IoU': [f"{v:.4f}" for v in arch_metrics['val_iou']],
            'Val Loss': [f"{v:.4f}" for v in arch_metrics['val_loss']]
        }

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def display_final_comparison(averages):
    """Affiche une comparaison finale des architectures"""
    st.header("Synthèse et Comparaison Finale")

    architectures = list(averages.keys())

    # Radar chart pour comparer les architectures
    st.subheader("Comparaison Radar des Performances")

    metrics_to_compare = ['accuracy', 'dice_coefficient', 'iou',
                          'val_accuracy', 'val_dice_coefficient', 'val_iou']
    labels = ['Accuracy', 'Dice', 'IoU', 'Val Acc', 'Val Dice', 'Val IoU']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le polygone

    colors = {'U-Net': '#2ecc71', 'YOLO': '#e74c3c', 'Hybrid': '#3498db'}

    for arch in architectures:
        values = [averages[arch][m] for m in metrics_to_compare]
        values += values[:1]  # Fermer le polygone
        ax.plot(angles, values, 'o-', linewidth=2, label=arch, color=colors[arch])
        ax.fill(angles, values, alpha=0.25, color=colors[arch])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Comparaison des Performances', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Meilleure architecture par métrique
    st.subheader("Meilleure Architecture par Métrique")

    best_results = []
    for metric in metrics_to_compare + ['loss', 'val_loss']:
        best_arch = None
        best_val = None

        for arch in architectures:
            val = averages[arch][metric]
            if best_val is None:
                best_val = val
                best_arch = arch
            elif 'loss' in metric:  # Pour loss, on veut le minimum
                if val < best_val:
                    best_val = val
                    best_arch = arch
            else:  # Pour les autres métriques, on veut le maximum
                if val > best_val:
                    best_val = val
                    best_arch = arch

        best_results.append({
            'Métrique': metric.replace('_', ' ').title(),
            'Meilleure Architecture': best_arch,
            'Valeur': f"{best_val:.4f}"
        })

    df = pd.DataFrame(best_results)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Fonction principale de l'application"""

    # Titre principal
    st.title("Projet Deep Learning - Détection des Dommages Routiers")
    st.markdown("""
    **Master 2 HPC - Année 2025-2026**
    Segmentation sémantique des dégradations routières avec U-Net, YOLO et architecture Hybride.
    """)

    st.divider()

    # Menu de navigation
    menu = st.sidebar.radio(
        "Navigation",
        ["Prétraitement", "Histogrammes des Métriques", "Évolution de l'Entraînement",
         "Métriques Détaillées", "Synthèse Finale"]
    )

    # Charger les métriques
    metrics = load_metrics()

    if not metrics:
        st.error("Aucune métrique trouvée. Vérifiez que les fichiers JSON sont présents dans results/logs/")
        return

    # Calculer les moyennes globales
    averages = calculate_global_averages(metrics)

    # Afficher la section sélectionnée
    if menu == "Prétraitement":
        display_preprocessing_results()

    elif menu == "Histogrammes des Métriques":
        display_metrics_histograms(metrics, averages)

    elif menu == "Évolution de l'Entraînement":
        display_training_evolution(metrics)

    elif menu == "Métriques Détaillées":
        display_detailed_metrics(metrics)

    elif menu == "Synthèse Finale":
        display_final_comparison(averages)

    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### Types de dommages
    - **Classe 0**: Fissures longitudinales
    - **Classe 1**: Fissures transversales
    - **Classe 2**: Fissures crocodiles
    - **Classe 4**: Nids-de-poule
    """)

    st.sidebar.markdown("---")
    st.sidebar.info("Projet de Deep Learning - M2 HPC")


if __name__ == "__main__":
    main()
