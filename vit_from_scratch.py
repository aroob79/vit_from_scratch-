import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, Input

# Patch Embedding Layer
class PatchEmbedding(layers.Layer):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # Number of patches
        self.proj = layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim), initializer=initializer, trainable=True
        )
        self.pos_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, embed_dim), initializer=initializer, trainable=True
        )


    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = self.proj(x)
        x = tf.reshape(x, [batch_size, -1, x.shape[-1]])

        # Add CLS Token
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, x.shape[-1]])
        x = tf.concat([cls_tokens, x], axis=1)

        # Add Positional Encoding
        x += self.pos_embedding
        x = tf.clip_by_value(x, -5.0, 5.0)
        return x

# Multi-Head Self-Attention
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, x):
        return self.mha(x, x)

# Feed-Forward Network
class FeedForward(layers.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            Dense(hidden_dim, activation="gelu", kernel_initializer=tf.keras.initializers.GlorotUniform()),
            Dropout(dropout),
            Dense(dim, kernel_initializer=tf.keras.initializers.GlorotUniform()),
            Dropout(dropout),
        ])

    def call(self, x):
        return self.ffn(x)

# Vision Transformer Model
class ViTModel(tf.keras.Model):
    def __init__(self, ch, img_size, patch_size, emb_dim, n_layers, dropout, heads, num_classes):
        super().__init__()
        self.ch = ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.heads = heads
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=ch, embed_dim=emb_dim)
        self.mha = MultiHeadSelfAttention(embed_dim=emb_dim, num_heads=heads)
        self.ffn = FeedForward(dim=emb_dim, hidden_dim=emb_dim * 4, dropout=dropout) 

        self.layer_norm1 = LayerNormalization(epsilon=1e-6)  # ✅ Define once
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)  # ✅ Define once 
        self.outLayer = Dense(self.num_classes, activation="softmax")
        
        self.dropout_layer = layers.Dropout(self.dropout)

    def build_single_block(self, x):
        """Single Transformer Encoder Block"""
        x1 = self.layer_norm1(x)
        x2 = self.mha(x1)
        x3 = self.dropout_layer(layers.Add()([x, x2]))  # Dropout before residual
        x4 = self.layer_norm2(x3)
        x5 = self.ffn(x4)
        x6 = self.dropout_layer(layers.Add()([x3, x5]))  # Dropout before residual
        return x6

    def call(self, inputs):
        """Build the Full Vision Transformer Model"""
        x = self.patch_embedding(inputs)

        for _ in range(self.n_layers):
            x = self.build_single_block(x)

        x = layers.GlobalAveragePooling1D()(x)
        outputs = self.outLayer(x)
        return outputs
    
    
if __name__ == "__main__":
    # Example Model Instantiation
    img_size = 224
    patch_size = 16
    vit_model = ViTModel(
        ch=3, img_size=img_size, patch_size=patch_size, emb_dim=768, 
        n_layers=12, out_dim=768, dropout=0.1, heads=8, num_classes=10
    )

    # Compile and Print Model Summary
    inputs = Input(shape=(img_size, img_size, 3))
    outputs = vit_model(inputs)
    model = Model(inputs, outputs)
    model.summary()

