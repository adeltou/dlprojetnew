"""
Callbacks pour l'entraînement des modèles
EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, etc.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from typing import List, Dict

# Import de la configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *


class EarlyStoppingWithRestore(keras.callbacks.Callback):
    """
    Early Stopping avec restauration des meilleurs poids
    Arrête l'entraînement si pas d'amélioration après 'patience' epochs
    """
    
    def __init__(self, monitor='val_loss', patience=15, verbose=1):
        """
        Args:
            monitor: Métrique à surveiller
            patience: Nombre d'epochs sans amélioration avant arrêt
            verbose: Niveau de verbosité
        """
        super(EarlyStoppingWithRestore, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if 'loss' in self.monitor else -np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        # Vérifier si amélioration
        if 'loss' in self.monitor:
            improved = current < self.best
        else:
            improved = current > self.best
        
        if improved:
            self.best = current
            self.wait = 0
            # Sauvegarder les meilleurs poids
            self.best_weights = self.model.get_weights()
            if self.verbose:
                print(f"\n✅ Epoch {epoch+1}: {self.monitor} improved to {current:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"\n⏳ Epoch {epoch+1}: {self.monitor} did not improve ({self.wait}/{self.patience})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered! Restoring best weights...")
                # Restaurer les meilleurs poids
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"\n✅ Training stopped at epoch {self.stopped_epoch + 1}")
            print(f"   Best {self.monitor}: {self.best:.4f}")


class LearningRateLogger(keras.callbacks.Callback):
    """
    Log le learning rate à chaque epoch
    """
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        # Handle different learning rate types (Variable, Schedule, etc.)
        if hasattr(lr, 'numpy'):
            lr = float(lr.numpy())
        elif callable(lr):
            # For learning rate schedules
            lr = float(lr(self.model.optimizer.iterations))
        else:
            lr = float(lr)
        logs['lr'] = lr
        print(f"\n📊 Learning Rate: {lr:.6f}")


class MetricsLogger(keras.callbacks.Callback):
    """
    Log les métriques détaillées à chaque epoch
    """
    
    def __init__(self, log_file=None):
        super(MetricsLogger, self).__init__()
        self.log_file = log_file
        self.history = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Créer un dictionnaire avec les métriques
        metrics = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        metrics.update(logs)
        
        self.history.append(metrics)
        
        # Affichage formaté
        print(f"\n📊 Epoch {epoch+1} Metrics:")
        print("-" * 60)
        for key, value in logs.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
        print("-" * 60)
        
        # Sauvegarder dans un fichier si spécifié
        if self.log_file:
            import json
            with open(self.log_file, 'w') as f:
                json.dump(self.history, f, indent=2)


def create_callbacks(model_name: str,
                     monitor: str = 'val_loss',
                     patience_early_stop: int = 15,
                     patience_reduce_lr: int = 5,
                     save_best_only: bool = True,
                     log_dir: str = LOGS_DIR,
                     models_dir: str = MODELS_DIR) -> List[keras.callbacks.Callback]:
    """
    Crée une liste de callbacks pour l'entraînement
    
    Args:
        model_name: Nom du modèle (pour les fichiers de sauvegarde)
        monitor: Métrique à surveiller
        patience_early_stop: Patience pour early stopping
        patience_reduce_lr: Patience pour reduce LR
        save_best_only: Sauvegarder seulement le meilleur modèle
        log_dir: Dossier pour les logs
        models_dir: Dossier pour sauvegarder les modèles
        
    Returns:
        Liste de callbacks
    """
    # Créer les dossiers si nécessaire
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Timestamp pour cette exécution
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Chemin pour sauvegarder le modèle
    model_path = os.path.join(models_dir, f'{model_name}_{timestamp}.keras')
    
    # Chemin pour les logs CSV
    csv_log_path = os.path.join(log_dir, f'{model_name}_{timestamp}.csv')
    
    # Chemin pour les logs détaillés
    metrics_log_path = os.path.join(log_dir, f'{model_name}_{timestamp}_metrics.json')
    # Déterminer le mode basé sur la métrique surveillée
    # Les métriques contenant 'loss' doivent être minimisées, les autres maximisés
    if 'loss' in monitor:
        mode = 'min'
    else:
        mode = 'max'

    callbacks = [
        # 1. ModelCheckpoint - Sauvegarde le meilleur modèle
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode=mode,
            verbose=1
        ),
        
        # 2. EarlyStopping - Arrête si pas d'amélioration
        EarlyStoppingWithRestore(
            monitor=monitor,
            patience=patience_early_stop,
            verbose=1
        ),
        
        # 3. ReduceLROnPlateau - Réduit le LR si plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-7,
            mode=mode,
            verbose=1
        ),
        
        # 4. CSVLogger - Log les métriques dans un CSV
        keras.callbacks.CSVLogger(
            filename=csv_log_path,
            separator=',',
            append=False
        ),
        
        # 5. TensorBoard - Visualisation dans TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, f'tensorboard_{model_name}_{timestamp}'),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),
        
        # 6. LearningRateLogger - Log le LR
        LearningRateLogger(),
        
        # 7. MetricsLogger - Log détaillé des métriques
        MetricsLogger(log_file=metrics_log_path)
    ]
    
    print("\n" + "=" * 80)
    print("✅ CALLBACKS CRÉÉS")
    print("=" * 80)
    print(f"📁 Modèle sera sauvegardé dans: {model_path}")
    print(f"📊 Logs CSV: {csv_log_path}")
    print(f"📈 TensorBoard: tensorboard --logdir={log_dir}")
    print(f"📝 Métriques détaillées: {metrics_log_path}")
    print("=" * 80 + "\n")
    
    return callbacks


class TrainingProgressCallback(keras.callbacks.Callback):
    """
    Affiche une barre de progression pendant l'entraînement
    """
    
    def __init__(self, epochs):
        super(TrainingProgressCallback, self).__init__()
        self.epochs = epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{'=' * 80}")
        print(f"🔄 EPOCH {epoch + 1}/{self.epochs}")
        print(f"{'=' * 80}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculer le pourcentage
        progress = ((epoch + 1) / self.epochs) * 100
        
        # Barre de progression
        bar_length = 40
        filled = int(bar_length * (epoch + 1) / self.epochs)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n📊 Progression globale: [{bar}] {progress:.1f}%")
        print(f"{'=' * 80}\n")


def test_callbacks():
    """
    Fonction de test des callbacks
    """
    print("=" * 80)
    print("TEST DES CALLBACKS")
    print("=" * 80)
    
    # Créer des callbacks de test
    callbacks = create_callbacks(
        model_name='test_model',
        monitor='val_loss',
        patience_early_stop=5,
        patience_reduce_lr=3
    )
    
    print(f"\n✅ {len(callbacks)} callbacks créés:")
    for i, cb in enumerate(callbacks, 1):
        print(f"  {i}. {cb.__class__.__name__}")
    
    print("\n" + "=" * 80)
    print("✅ TEST DES CALLBACKS RÉUSSI!")
    print("=" * 80)


if __name__ == "__main__":
    # Créer les dossiers nécessaires
    from utils.config import create_directories
    create_directories()
    
    # Tester les callbacks
    test_callbacks()
