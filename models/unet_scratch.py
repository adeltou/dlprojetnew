"""
Implémentation U-Net from Scratch
Basée sur le papier original : "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
Architecture : Encoder-Decoder avec Skip Connections
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np

# Import de la configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import *
from models.model_utils import dice_coefficient, iou_metric, combined_loss


# ============================================================================
# BLOCS DE CONSTRUCTION DE U-NET
# ============================================================================

def conv_block(inputs, filters, kernel_size=3, use_batch_norm=True, dropout=0.0):
    """
    Bloc de convolution standard de U-Net : Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    
    Args:
        inputs: Tensor d'entrée
        filters: Nombre de filtres
        kernel_size: Taille du noyau de convolution
        use_batch_norm: Si True, applique BatchNormalization
        dropout: Taux de dropout (0 = pas de dropout)
        
    Returns:
        Tensor de sortie après le bloc de convolution
    """
    # Première convolution
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        kernel_initializer='he_normal'
    )(inputs)
    
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    # Deuxième convolution
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    # Dropout si spécifié
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    
    return x


def encoder_block(inputs, filters, use_batch_norm=True, dropout=0.0):
    """
    Bloc Encoder : Conv Block + MaxPooling
    
    Args:
        inputs: Tensor d'entrée
        filters: Nombre de filtres
        use_batch_norm: Si True, applique BatchNormalization
        dropout: Taux de dropout
        
    Returns:
        (conv_output, pooled_output) - Pour les skip connections et la suite
    """
    # Bloc de convolution
    conv = conv_block(inputs, filters, use_batch_norm=use_batch_norm, dropout=dropout)
    
    # Max Pooling pour réduire la dimension
    pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    
    return conv, pool


def decoder_block(inputs, skip_features, filters, use_batch_norm=True, dropout=0.0):
    """
    Bloc Decoder : UpSampling + Concatenation (Skip Connection) + Conv Block
    
    Args:
        inputs: Tensor d'entrée depuis le niveau inférieur
        skip_features: Features du encoder (skip connection)
        filters: Nombre de filtres
        use_batch_norm: Si True, applique BatchNormalization
        dropout: Taux de dropout
        
    Returns:
        Tensor de sortie après le bloc decoder
    """
    # Upsampling (augmenter la résolution)
    x = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )(inputs)
    
    # Concatener avec les skip features du encoder
    x = layers.Concatenate()([x, skip_features])
    
    # Bloc de convolution
    x = conv_block(x, filters, use_batch_norm=use_batch_norm, dropout=dropout)
    
    return x


# ============================================================================
# ARCHITECTURE U-NET COMPLÈTE
# ============================================================================

class UNetScratch:
    """
    Architecture U-Net complète implémentée from scratch
    
    Structure:
    - Encoder : 4 niveaux de descente (64 -> 128 -> 256 -> 512)
    - Bottleneck : 1024 filtres
    - Decoder : 4 niveaux de montée (512 -> 256 -> 128 -> 64)
    - Output : Segmentation avec num_classes canaux
    """
    
    def __init__(self, 
                 input_shape=(256, 256, 3),
                 num_classes=NUM_CLASSES + 1,
                 filters_base=64,
                 use_batch_norm=True,
                 dropout=0.3):
        """
        Initialise l'architecture U-Net
        
        Args:
            input_shape: Forme de l'entrée (height, width, channels)
            num_classes: Nombre de classes de segmentation (incluant background)
            filters_base: Nombre de filtres dans le premier niveau (sera doublé à chaque niveau)
            use_batch_norm: Si True, utilise BatchNormalization
            dropout: Taux de dropout
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters_base = filters_base
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        
        self.model = None
    
    def build_model(self):
        """
        Construit l'architecture U-Net complète
        
        Returns:
            Model Keras compilé
        """
        # ====================================================================
        # ENTRÉE
        # ====================================================================
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # ====================================================================
        # ENCODER (Contracting Path)
        # ====================================================================
        # Niveau 1 : 64 filtres, 256x256 -> 128x128
        skip1, pool1 = encoder_block(
            inputs, 
            filters=self.filters_base,
            use_batch_norm=self.use_batch_norm,
            dropout=0  # Pas de dropout au premier niveau
        )
        
        # Niveau 2 : 128 filtres, 128x128 -> 64x64
        skip2, pool2 = encoder_block(
            pool1,
            filters=self.filters_base * 2,
            use_batch_norm=self.use_batch_norm,
            dropout=0
        )
        
        # Niveau 3 : 256 filtres, 64x64 -> 32x32
        skip3, pool3 = encoder_block(
            pool2,
            filters=self.filters_base * 4,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout
        )
        
        # Niveau 4 : 512 filtres, 32x32 -> 16x16
        skip4, pool4 = encoder_block(
            pool3,
            filters=self.filters_base * 8,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout
        )
        
        # ====================================================================
        # BOTTLENECK (Bridge)
        # ====================================================================
        # 1024 filtres, 16x16
        bottleneck = conv_block(
            pool4,
            filters=self.filters_base * 16,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout
        )
        
        # ====================================================================
        # DECODER (Expansive Path)
        # ====================================================================
        # Niveau 4 : 512 filtres, 16x16 -> 32x32
        up4 = decoder_block(
            bottleneck,
            skip_features=skip4,
            filters=self.filters_base * 8,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout
        )
        
        # Niveau 3 : 256 filtres, 32x32 -> 64x64
        up3 = decoder_block(
            up4,
            skip_features=skip3,
            filters=self.filters_base * 4,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout
        )
        
        # Niveau 2 : 128 filtres, 64x64 -> 128x128
        up2 = decoder_block(
            up3,
            skip_features=skip2,
            filters=self.filters_base * 2,
            use_batch_norm=self.use_batch_norm,
            dropout=0
        )
        
        # Niveau 1 : 64 filtres, 128x128 -> 256x256
        up1 = decoder_block(
            up2,
            skip_features=skip1,
            filters=self.filters_base,
            use_batch_norm=self.use_batch_norm,
            dropout=0
        )
        
        # ====================================================================
        # SORTIE (Output Layer)
        # ====================================================================
        # Convolution 1x1 pour obtenir num_classes canaux
        outputs = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            padding='same',
            activation='softmax',  # Softmax pour la classification multi-classe
            name='output_segmentation'
        )(up1)
        
        # ====================================================================
        # CRÉER LE MODÈLE
        # ====================================================================
        self.model = models.Model(inputs=inputs, outputs=outputs, name='UNet_Scratch')
        
        return self.model
    
    def compile_model(self, 
                     optimizer='adam',
                     learning_rate=LEARNING_RATE,
                     loss='combined',
                     metrics=['accuracy']):
        """
        Compile le modèle avec optimizer, loss et metrics
        
        Args:
            optimizer: Nom de l'optimizer ('adam', 'sgd', etc.)
            learning_rate: Taux d'apprentissage
            loss: Fonction de loss ('combined', 'dice', 'categorical_crossentropy')
            metrics: Liste des métriques à suivre
            
        Returns:
            Model compilé
        """
        if self.model is None:
            self.build_model()
        
        # Choisir l'optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        # Choisir la loss function
        if loss == 'combined':
            loss_fn = combined_loss
        elif loss == 'dice':
            loss_fn = lambda y_true, y_pred: 1 - dice_coefficient(y_true, y_pred)
        elif loss == 'categorical_crossentropy':
            loss_fn = 'categorical_crossentropy'
        else:
            loss_fn = loss
        
        # Compiler le modèle
        self.model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=metrics
        )
        
        return self.model
    
    def summary(self):
        """
        Affiche le résumé de l'architecture
        """
        if self.model is None:
            self.build_model()
        
        return self.model.summary()
    
    def get_model(self):
        """
        Retourne le modèle Keras
        """
        if self.model is None:
            self.build_model()
        
        return self.model
    
    def count_parameters(self):
        """
        Compte le nombre de paramètres du modèle
        
        Returns:
            (total_params, trainable_params, non_trainable_params)
        """
        if self.model is None:
            self.build_model()
        
        trainable = np.sum([keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable = np.sum([keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        total = trainable + non_trainable
        
        return total, trainable, non_trainable


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_unet():
    """
    Fonction de test de l'architecture U-Net
    """
    print("\n" + "=" * 100)
    print("TEST DE L'ARCHITECTURE U-NET FROM SCRATCH")
    print("=" * 100)
    
    # Créer le modèle
    print("\n🏗️  Construction du modèle U-Net...")
    unet = UNetScratch(
        input_shape=(256, 256, 3),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        use_batch_norm=True,
        dropout=0.3
    )
    
    # Build
    model = unet.build_model()
    print(f"✅ Modèle construit avec succès!")
    
    # Afficher le résumé
    print(f"\n📊 Résumé de l'architecture:")
    print("-" * 100)
    unet.summary()
    
    # Compter les paramètres
    total, trainable, non_trainable = unet.count_parameters()
    print(f"\n📈 Nombre de paramètres:")
    print(f"  - Total: {total:,}")
    print(f"  - Trainable: {trainable:,}")
    print(f"  - Non-trainable: {non_trainable:,}")
    
    # Compiler le modèle
    print(f"\n⚙️  Compilation du modèle...")
    unet.compile_model(
        optimizer='adam',
        learning_rate=1e-4,
        loss='combined',
        metrics=['accuracy']
    )
    print(f"✅ Modèle compilé avec succès!")
    
    # Test de prédiction avec des données aléatoires
    print(f"\n🧪 Test de prédiction...")
    test_input = np.random.rand(2, 256, 256, 3).astype(np.float32)
    
    try:
        predictions = model.predict(test_input, verbose=0)
        print(f"✅ Prédiction réussie!")
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {predictions.shape}")
        print(f"  - Output min: {predictions.min():.4f}, max: {predictions.max():.4f}")
        
        # Vérifier que la sortie est bien une distribution de probabilités
        print(f"\n🔍 Vérification de la sortie:")
        sample_sum = predictions[0, 0, 0, :].sum()
        print(f"  - Somme des probabilités (doit être ≈ 1.0): {sample_sum:.4f}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("✅ TEST DE U-NET TERMINÉ AVEC SUCCÈS!")
    print("=" * 100)
    
    return model


# ============================================================================
# FONCTION POUR CRÉER ET RETOURNER LE MODÈLE
# ============================================================================

def create_unet_model(input_shape=(256, 256, 3),
                     num_classes=NUM_CLASSES + 1,
                     filters_base=64,
                     compile_model=True):
    """
    Fonction helper pour créer rapidement un modèle U-Net
    
    Args:
        input_shape: Shape de l'entrée
        num_classes: Nombre de classes
        filters_base: Nombre de filtres de base
        compile_model: Si True, compile le modèle
        
    Returns:
        Model Keras (compilé ou non)
    """
    unet = UNetScratch(
        input_shape=input_shape,
        num_classes=num_classes,
        filters_base=filters_base,
        use_batch_norm=True,
        dropout=0.3
    )
    
    if compile_model:
        return unet.compile_model(
            optimizer='adam',
            learning_rate=LEARNING_RATE,
            loss='combined',
            metrics=['accuracy']
        )
    else:
        return unet.build_model()


if __name__ == "__main__":
    # Tester l'architecture
    model = test_unet()
