"""
Package Training pour le projet RDD2022
Contient les scripts et utilitaires pour l'entraînement des modèles
"""

from .callbacks import (
    create_callbacks,
    EarlyStoppingWithRestore,
    LearningRateLogger
)

__all__ = [
    'create_callbacks',
    'EarlyStoppingWithRestore',
    'LearningRateLogger'
]
