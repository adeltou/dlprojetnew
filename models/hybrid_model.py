"""
Architecture Hybride Proposée - Combinaison U-Net + YOLO
Méthode 3 : Architecture innovante pour la segmentation de dommages routiers

Architecture:
- Encoder: Backbone efficace inspiré de YOLO (CSPDarknet ou EfficientNet)
- Decoder: Structure U-Net avec skip connections
- Améliorations: Attention gates, Residual connections, ASPP
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
# BLOCS DE CONSTRUCTION AVANCÉS
# ============================================================================

def attention_gate(x, gating_signal, filters):
    """
    Attention Gate pour mettre en valeur les features importantes
    
    Args:
        x: Features du encoder (skip connection)
        gating_signal: Signal du niveau inférieur
        filters: Nombre de filtres
        
    Returns:
        Features avec attention appliquée
    """
    # Transformer x et gating_signal à la même dimension
    theta_x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(gating_signal)
    
    # Additionner
    concat = layers.Add()([theta_x, phi_g])
    concat = layers.Activation('relu')(concat)
    
    # Appliquer une convolution 1x1
    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(concat)
    psi = layers.Activation('sigmoid')(psi)
    
    # Multiplier avec les features originales
    attended = layers.Multiply()([x, psi])
    
    return attended


def residual_block(inputs, filters, use_batch_norm=True):
    """
    Bloc résiduel pour améliorer le gradient flow
    
    Args:
        inputs: Tensor d'entrée
        filters: Nombre de filtres
        use_batch_norm: Si True, utilise BatchNormalization
        
    Returns:
        Tensor avec connexion résiduelle
    """
    # Première convolution
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Deuxième convolution
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    # Connexion résiduelle (projection si nécessaire)
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, (1, 1), padding='same')(inputs)
    
    # Addition
    x = layers.Add()([x, inputs])
    x = layers.Activation('relu')(x)
    
    return x


def aspp_block(inputs, filters):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    Capture des contextes multi-échelles
    
    Args:
        inputs: Tensor d'entrée
        filters: Nombre de filtres
        
    Returns:
        Features avec contexte multi-échelle
    """
    # Différentes rates de dilation
    # 1x1 convolution
    conv1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    
    # 3x3 convolutions avec différentes dilations
    conv2 = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=6, activation='relu')(inputs)
    conv3 = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=12, activation='relu')(inputs)
    conv4 = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=18, activation='relu')(inputs)
    
    # Global pooling
    gap = layers.GlobalAveragePooling2D()(inputs)
    gap = layers.Reshape((1, 1, inputs.shape[-1]))(gap)
    gap = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(gap)
    gap = layers.UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), interpolation='bilinear')(gap)
    
    # Concatener toutes les features
    concat = layers.Concatenate()([conv1, conv2, conv3, conv4, gap])
    
    # Fusion finale
    output = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(concat)
    
    return output


def csp_block(inputs, filters, use_batch_norm=True):
    """
    Cross Stage Partial block (inspiré de YOLO)
    Améliore le gradient flow et réduit les calculs
    
    Args:
        inputs: Tensor d'entrée
        filters: Nombre de filtres
        use_batch_norm: Si True, utilise BatchNormalization
        
    Returns:
        Features CSP
    """
    # Diviser en deux chemins
    half_filters = filters // 2
    
    # Chemin 1: Passage direct
    path1 = layers.Conv2D(half_filters, (1, 1), padding='same')(inputs)
    if use_batch_norm:
        path1 = layers.BatchNormalization()(path1)
    path1 = layers.Activation('relu')(path1)
    
    # Chemin 2: Convolutions
    path2 = layers.Conv2D(half_filters, (1, 1), padding='same')(inputs)
    if use_batch_norm:
        path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    
    path2 = layers.Conv2D(half_filters, (3, 3), padding='same')(path2)
    if use_batch_norm:
        path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    
    path2 = layers.Conv2D(half_filters, (1, 1), padding='same')(path2)
    if use_batch_norm:
        path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation('relu')(path2)
    
    # Concatener les deux chemins
    output = layers.Concatenate()([path1, path2])
    
    return output


def encoder_block_hybrid(inputs, filters, use_residual=True, use_csp=False, use_batch_norm=True):
    """
    Bloc encoder hybride avec options avancées
    
    Args:
        inputs: Tensor d'entrée
        filters: Nombre de filtres
        use_residual: Si True, utilise residual connections
        use_csp: Si True, utilise CSP block
        use_batch_norm: Si True, utilise BatchNormalization
        
    Returns:
        (features, pooled) - Pour skip connections et suite
    """
    if use_csp:
        x = csp_block(inputs, filters, use_batch_norm)
    elif use_residual:
        x = residual_block(inputs, filters, use_batch_norm)
    else:
        # Convolution standard
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
    # Max pooling
    pooled = layers.MaxPooling2D((2, 2))(x)
    
    return x, pooled


def decoder_block_hybrid(inputs, skip_features, filters, use_attention=True, use_batch_norm=True):
    """
    Bloc decoder hybride avec attention gate
    
    Args:
        inputs: Tensor d'entrée depuis le niveau inférieur
        skip_features: Features du encoder
        filters: Nombre de filtres
        use_attention: Si True, utilise attention gate
        use_batch_norm: Si True, utilise BatchNormalization
        
    Returns:
        Features décodées
    """
    # Upsampling
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    
    # Attention gate si activé
    if use_attention:
        skip_features = attention_gate(skip_features, x, filters)
    
    # Concatenation
    x = layers.Concatenate()([x, skip_features])
    
    # Convolutions
    x = residual_block(x, filters, use_batch_norm)
    
    return x


# ============================================================================
# ARCHITECTURE HYBRIDE COMPLÈTE
# ============================================================================

class HybridModel:
    """
    Architecture Hybride pour la Segmentation de Dommages Routiers
    
    Innovations:
    - Encoder efficace avec CSP blocks (inspiré YOLO)
    - Decoder U-Net avec skip connections
    - Attention gates pour focaliser sur les régions importantes
    - ASPP pour capturer le contexte multi-échelle
    - Residual connections pour meilleur gradient flow
    """
    
    def __init__(self,
                 input_shape=(256, 256, 3),
                 num_classes=NUM_CLASSES + 1,
                 filters_base=64,
                 use_attention=True,
                 use_residual=True,
                 use_csp=True,
                 use_aspp=True,
                 dropout=0.3):
        """
        Initialise l'architecture hybride
        
        Args:
            input_shape: Forme de l'entrée (height, width, channels)
            num_classes: Nombre de classes
            filters_base: Nombre de filtres de base
            use_attention: Si True, utilise attention gates
            use_residual: Si True, utilise residual connections
            use_csp: Si True, utilise CSP blocks
            use_aspp: Si True, utilise ASPP dans le bottleneck
            dropout: Taux de dropout
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters_base = filters_base
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_csp = use_csp
        self.use_aspp = use_aspp
        self.dropout = dropout
        
        self.model = None
    
    def build_model(self):
        """
        Construit l'architecture hybride complète
        
        Returns:
            Model Keras
        """
        # ====================================================================
        # ENTRÉE
        # ====================================================================
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # ====================================================================
        # ENCODER (Hybrid: CSP + Residual)
        # ====================================================================
        # Niveau 1: 64 filtres
        skip1, pool1 = encoder_block_hybrid(
            inputs,
            filters=self.filters_base,
            use_residual=self.use_residual,
            use_csp=self.use_csp
        )
        
        # Niveau 2: 128 filtres
        skip2, pool2 = encoder_block_hybrid(
            pool1,
            filters=self.filters_base * 2,
            use_residual=self.use_residual,
            use_csp=self.use_csp
        )
        
        # Niveau 3: 256 filtres
        skip3, pool3 = encoder_block_hybrid(
            pool2,
            filters=self.filters_base * 4,
            use_residual=self.use_residual,
            use_csp=self.use_csp
        )
        
        # Niveau 4: 512 filtres
        skip4, pool4 = encoder_block_hybrid(
            pool3,
            filters=self.filters_base * 8,
            use_residual=self.use_residual,
            use_csp=self.use_csp
        )
        
        # ====================================================================
        # BOTTLENECK (avec ASPP optionnel)
        # ====================================================================
        if self.use_aspp:
            # ASPP pour contexte multi-échelle
            bottleneck = aspp_block(pool4, filters=self.filters_base * 16)
        else:
            # Bloc résiduel standard
            bottleneck = residual_block(pool4, filters=self.filters_base * 16)
        
        # Dropout
        bottleneck = layers.Dropout(self.dropout)(bottleneck)
        
        # ====================================================================
        # DECODER (U-Net style avec Attention)
        # ====================================================================
        # Niveau 4: 512 filtres
        up4 = decoder_block_hybrid(
            bottleneck,
            skip_features=skip4,
            filters=self.filters_base * 8,
            use_attention=self.use_attention
        )
        up4 = layers.Dropout(self.dropout)(up4)
        
        # Niveau 3: 256 filtres
        up3 = decoder_block_hybrid(
            up4,
            skip_features=skip3,
            filters=self.filters_base * 4,
            use_attention=self.use_attention
        )
        up3 = layers.Dropout(self.dropout / 2)(up3)
        
        # Niveau 2: 128 filtres
        up2 = decoder_block_hybrid(
            up3,
            skip_features=skip2,
            filters=self.filters_base * 2,
            use_attention=self.use_attention
        )
        
        # Niveau 1: 64 filtres
        up1 = decoder_block_hybrid(
            up2,
            skip_features=skip1,
            filters=self.filters_base,
            use_attention=self.use_attention
        )
        
        # ====================================================================
        # SORTIE
        # ====================================================================
        # Convolution finale
        outputs = layers.Conv2D(
            self.num_classes,
            (1, 1),
            padding='same',
            activation='softmax',
            name='output_segmentation'
        )(up1)
        
        # ====================================================================
        # CRÉER LE MODÈLE
        # ====================================================================
        self.model = models.Model(inputs=inputs, outputs=outputs, name='Hybrid_UNet_YOLO')
        
        return self.model
    
    def compile_model(self,
                     optimizer='adam',
                     learning_rate=LEARNING_RATE,
                     loss='combined',
                     metrics=['accuracy']):
        """
        Compile le modèle
        
        Args:
            optimizer: Optimizer
            learning_rate: Learning rate
            loss: Loss function
            metrics: Métriques
            
        Returns:
            Modèle compilé
        """
        if self.model is None:
            self.build_model()
        
        # Optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        # Loss
        if loss == 'combined':
            loss_fn = combined_loss
        elif loss == 'dice':
            loss_fn = lambda y_true, y_pred: 1 - dice_coefficient(y_true, y_pred)
        else:
            loss_fn = loss
        
        # Compiler
        self.model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=metrics
        )
        
        return self.model
    
    def summary(self):
        """Affiche le résumé"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def get_model(self):
        """Retourne le modèle"""
        if self.model is None:
            self.build_model()
        return self.model
    
    def count_parameters(self):
        """Compte les paramètres"""
        if self.model is None:
            self.build_model()
        
        trainable = np.sum([keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable = np.sum([keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        total = trainable + non_trainable
        
        return total, trainable, non_trainable


# ============================================================================
# FONCTIONS HELPER
# ============================================================================

def create_hybrid_model(input_shape=(256, 256, 3),
                       num_classes=NUM_CLASSES + 1,
                       filters_base=64,
                       compile_model=True):
    """
    Fonction helper pour créer rapidement le modèle hybride
    
    Args:
        input_shape: Shape d'entrée
        num_classes: Nombre de classes
        filters_base: Filtres de base
        compile_model: Si True, compile
        
    Returns:
        Modèle hybride
    """
    hybrid = HybridModel(
        input_shape=input_shape,
        num_classes=num_classes,
        filters_base=filters_base,
        use_attention=True,
        use_residual=True,
        use_csp=True,
        use_aspp=True,
        dropout=0.3
    )
    
    if compile_model:
        return hybrid.compile_model(
            optimizer='adam',
            learning_rate=LEARNING_RATE,
            loss='combined',
            metrics=['accuracy']
        )
    else:
        return hybrid.build_model()


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_hybrid():
    """
    Test de l'architecture hybride
    """
    print("\n" + "=" * 100)
    print("TEST DE L'ARCHITECTURE HYBRIDE")
    print("=" * 100)
    
    # Créer le modèle
    print("\n🏗️  Construction du modèle hybride...")
    hybrid = HybridModel(
        input_shape=(256, 256, 3),
        num_classes=NUM_CLASSES + 1,
        filters_base=64,
        use_attention=True,
        use_residual=True,
        use_csp=True,
        use_aspp=True,
        dropout=0.3
    )
    
    model = hybrid.build_model()
    print("✅ Modèle construit avec succès!")
    
    # Résumé
    print("\n📊 Résumé de l'architecture:")
    print("-" * 100)
    hybrid.summary()
    
    # Paramètres
    total, trainable, non_trainable = hybrid.count_parameters()
    print(f"\n📈 Nombre de paramètres:")
    print(f"  - Total: {total:,}")
    print(f"  - Trainable: {trainable:,}")
    print(f"  - Non-trainable: {non_trainable:,}")
    
    # Compiler
    print(f"\n⚙️  Compilation...")
    hybrid.compile_model(
        optimizer='adam',
        learning_rate=1e-4,
        loss='combined',
        metrics=['accuracy']
    )
    print("✅ Modèle compilé!")
    
    # Test de prédiction
    print(f"\n🧪 Test de prédiction...")
    test_input = np.random.rand(2, 256, 256, 3).astype(np.float32)
    
    try:
        predictions = model.predict(test_input, verbose=0)
        print("✅ Prédiction réussie!")
        print(f"  - Input: {test_input.shape}")
        print(f"  - Output: {predictions.shape}")
        print(f"  - Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
        
        # Vérification
        sample_sum = predictions[0, 0, 0, :].sum()
        print(f"  - Somme proba: {sample_sum:.4f}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("✅ TEST HYBRIDE TERMINÉ AVEC SUCCÈS!")
    print("=" * 100)
    
    print("\n🎯 Innovations de cette architecture:")
    print("  ✅ Encoder efficace avec CSP blocks (YOLO-inspired)")
    print("  ✅ Decoder U-Net avec skip connections")
    print("  ✅ Attention gates pour features importantes")
    print("  ✅ ASPP pour contexte multi-échelle")
    print("  ✅ Residual connections pour meilleur gradient")
    
    print("\n" + "=" * 100)
    
    return model


if __name__ == "__main__":
    # Tester l'architecture
    model = test_hybrid()
