import math
import tensorflow as tf
# from tensorflow import keras
# import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
#from keras_multi_head import MultiHeadAttention


class TransformerLayer(layers.Layer):
    def __init__(self, embed_dim, ff_dim=2048, num_heads=8, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
#        self.ffn1 = layers.Dense(ff_dim, activation="relu")
#        self.ffn2 = layers.Dense(embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, pos_emb=None):
        if pos_emb is not None:
            inputs = inputs + pos_emb
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        #ffn_output = self.ffn(out1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads=8, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = [TransformerLayer(embed_dim, num_heads=num_heads) for i in range(num_layers)]
        self.num_layers = num_layers
        self.norm = layers.LayerNormalization(epsilon=1e-6) if norm is not None else None

    def call(self, inputs, training, pos_emb=None):
        output = inputs
        for i in range(len(self.layers)):
            output = self.layers[i](output, training, pos_emb)
        if self.norm is not None:
            output = self.norm(output)
        return output


class PositionEmbeddingSine(layers.Layer):
    """
    Refer to DETR/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be true if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, x):
        y_embed = K.cumsum(K.ones_like(x[:, :, :, 0]), 1)
        x_embed = K.cumsum(K.ones_like(x[:, :, :, 0]), 2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = K.arange(self.num_pos_feats, dtype='float')
        dim = self.temperature ** (2 * (dim_t // 2) /self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = K.stack((K.sin(pos_x[:, :, :, 0::2]), K.cos(pos_x[:, :, :, 1::2])), axis=4)
        b, h, w, d, c = K.int_shape(pos_x)
        pos_x = K.reshape(pos_x, (-1, h, w, d*c))
        pos_y = K.stack((K.sin(pos_y[:, :, :, 0::2]), K.cos(pos_y[:, :, :, 1::2])), axis=4)
        pos_y = K.reshape(pos_y, (-1, h, w, d*c))
        pos = K.permute_dimensions(K.concatenate((pos_y, pos_x), axis=3), (0, 3, 1, 2))
        return pos


