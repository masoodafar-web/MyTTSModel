# MyTTSModel.py
"""
MyTTSModel - Transformer-based Text-to-Speech Model

This module implements a complete Transformer architecture for text-to-speech synthesis.
It includes:
- Positional encoding utilities
- Encoder and Decoder with multi-head attention
- PostNet for mel-spectrogram refinement
- Greedy generation for inference

The model uses teacher forcing during training and autoregressive generation during inference.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# =========================
# Positional Encoding
# =========================
def positional_encoding(length, d_model):
    """
    Generate sinusoidal positional encoding for Transformer models.
    
    Uses alternating sine and cosine functions at different frequencies to encode
    position information that can be added to token embeddings.
    
    Args:
        length (int): Maximum sequence length to generate encodings for.
        d_model (int): Dimension of the model (embedding size).
        
    Returns:
        tf.Tensor: Positional encoding tensor of shape (length, d_model).
        
    Example:
        >>> pos_enc = positional_encoding(100, 256)
        >>> pos_enc.shape
        TensorShape([100, 256])
    """
    positions = np.arange(length)[:, None]
    dims = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates
    pos = np.zeros((length, d_model), dtype=np.float32)
    pos[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.convert_to_tensor(pos, dtype=tf.float32)

# =========================
# Embeddings
# =========================
class PositionalEmbedding(layers.Layer):
    """
    Embedding layer with added positional encoding for text tokens.
    
    Combines learned token embeddings with fixed sinusoidal positional encodings,
    scaled by sqrt(d_model) as in the original Transformer paper.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the embedding space.
        max_length (int): Maximum sequence length supported. Default: 2048.
        name (str, optional): Layer name.
        
    Attributes:
        embedding (layers.Embedding): Token embedding layer.
        pos_encoding (tf.Tensor): Pre-computed positional encodings.
    """
    
    def __init__(self, vocab_size, d_model, max_length=2048, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=False, name="tok_emb")
        self.pos_encoding = positional_encoding(max_length, d_model)

    def call(self, x):
        """
        Apply embedding and positional encoding to input token IDs.
        
        Args:
            x (tf.Tensor): Input token IDs of shape (batch_size, seq_len).
            
        Returns:
            tf.Tensor: Embedded tokens with positional encoding, shape (batch_size, seq_len, d_model).
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)                                  # (B,L,D)
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class MelPositionalProjection(layers.Layer):
    """
    Projects mel-spectrogram features to model dimension with positional encoding.
    
    Uses an optional prenet (bottleneck layers with dropout for better convergence)
    to project mel features to d_model dimension, then adds positional encoding.
    
    Args:
        n_mels (int): Number of mel-spectrogram bands.
        d_model (int): Target model dimension.
        max_length (int): Maximum sequence length. Default: 4096.
        name (str, optional): Layer name.
        use_prenet (bool): Whether to use prenet layers. Default: True.
        prenet_drop (float): Dropout rate for prenet layers. Default: 0.5.
        
    Attributes:
        prenet (tf.keras.Sequential or layers.Dense): Feature projection layer(s).
        pos_encoding (tf.Tensor): Pre-computed positional encodings.
    """
    
    def __init__(self, n_mels, d_model, max_length=4096, name=None, use_prenet=True, prenet_drop=0.5):
        super().__init__(name=name)
        self.d_model = d_model
        self.use_prenet = use_prenet
        if use_prenet:
            self.prenet = tf.keras.Sequential([
                layers.Dense(256, activation="relu"),
                layers.Dropout(prenet_drop),
                layers.Dense(256, activation="relu"),
                layers.Dropout(prenet_drop),
                layers.Dense(d_model),
            ], name="prenet")
        else:
            self.prenet = layers.Dense(d_model, name="prenet_linear")
        self.pos_encoding = positional_encoding(max_length, d_model)

    def call(self, mel):
        """
        Project mel-spectrogram to model dimension and add positional encoding.
        
        Args:
            mel (tf.Tensor): Input mel-spectrogram of shape (batch_size, time, n_mels).
            
        Returns:
            tf.Tensor: Projected and positionally encoded features of shape (batch_size, time, d_model).
        """
        # Feed float32 to prenet; it will cast internally if using mixed precision
        x = tf.cast(mel, tf.float32)
        x = self.prenet(x)                                     # (B,T,D), may be float16 under mixed precision
        scale = tf.math.sqrt(tf.cast(tf.shape(x)[-1], x.dtype))
        x = x * scale
        length = tf.shape(x)[1]
        pos = tf.cast(self.pos_encoding[tf.newaxis, :length, :], x.dtype)
        x = x + pos
        return x

# =========================
# Mask helpers
# =========================
def make_padding_bool(ids, pad_id):
    """
    Create boolean mask indicating valid (non-padding) positions.
    
    Args:
        ids (tf.Tensor): Input token IDs of shape (batch_size, seq_len).
        pad_id (int): Padding token ID.
        
    Returns:
        tf.Tensor: Boolean mask where True indicates valid positions, shape (batch_size, seq_len).
    """
    return tf.not_equal(ids, tf.cast(pad_id, ids.dtype))

def key_mask_from_valid(valid_bool):
    """
    Convert boolean valid mask to key mask format for MultiHeadAttention.
    
    Args:
        valid_bool (tf.Tensor): Boolean mask of shape (batch_size, seq_len).
        
    Returns:
        tf.Tensor: Key mask of shape (batch_size, 1, seq_len) where True indicates allowed positions.
    """
    # Keras MHA expects True where allowed
    return valid_bool[:, None, :]  # (B,1,S)

def make_causal_mask(T):
    """
    Create causal (lower-triangular) mask for autoregressive attention.
    
    Args:
        T (int): Sequence length.
        
    Returns:
        tf.Tensor: Boolean causal mask of shape (T, T) where True indicates allowed positions.
    """
    return tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)

def combine_causal_and_keypadding(dec_valid_bool):
    """
    Combine causal mask with key padding mask for decoder self-attention.
    
    Args:
        dec_valid_bool (tf.Tensor): Boolean valid mask of shape (batch_size, seq_len).
        
    Returns:
        tf.Tensor: Combined mask of shape (batch_size, seq_len, seq_len) where True indicates
                   allowed positions (both causal and not padding).
    """
    T = tf.shape(dec_valid_bool)[1]
    causal = make_causal_mask(T)                  # (T,T) True at <=
    key_mask = key_mask_from_valid(dec_valid_bool)  # (B,1,T)
    # Broadcast to (B,T,T), True where allowed
    return tf.logical_and(key_mask, causal[None, :, :])        # (B,T,T)

# =========================
# DropPath
# =========================
class DropPath(layers.Layer):
    """
    Stochastic depth layer that randomly drops entire residual paths during training.
    
    Implements DropPath/Stochastic Depth for better gradient flow and regularization
    in deep networks. During training, randomly drops entire paths with probability drop_prob.
    
    Args:
        drop_prob (float): Probability of dropping a path. Default: 0.0.
        name (str, optional): Layer name.
    """
    
    def __init__(self, drop_prob=0.0, name=None):
        super().__init__(name=name)
        self.drop_prob = float(drop_prob)
        self.supports_masking = True

    def call(self, x, training=False):
        """
        Apply DropPath to input tensor.
        Ensures dtype consistency under mixed precision by casting to x.dtype.
        
        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Whether in training mode.
            
        Returns:
            tf.Tensor: Output tensor, possibly with paths dropped.
        """
        if (not training) or self.drop_prob == 0.0:
            return x
        keep_prob = tf.cast(1.0 - self.drop_prob, x.dtype)
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * binary_tensor

# =========================
# Attention Blocks (Pre-Norm)
# =========================
class BaseAttentionPreNorm(layers.Layer):
    """
    Base attention layer with pre-normalization and residual connection.
    
    Implements a self-attention or cross-attention layer with:
    - Pre-LayerNorm for better training stability
    - Optional DropPath for regularization
    - Residual connection
    - Support for attention weight extraction
    
    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Model dimension.
        dropout (float): Dropout rate.
        droppath (float): DropPath rate. Default: 0.0.
        name (str, optional): Layer name.
    """
    
    def __init__(self, *, num_heads, d_model, dropout, droppath=0.0, name=None):
        super().__init__(name=name)
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="ln")
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="mha"
        )
        self.res_dropout = layers.Dropout(dropout, name="res_drop")
        self.drop_path = DropPath(droppath, name="droppath") if droppath > 0 else None
        self.add = layers.Add(name="add")

    def call(self, x, *, context=None, attn_mask=None, q_valid=None, training=False, need_attn=False):
        """
        Apply attention with pre-normalization.
        
        Args:
            x (tf.Tensor): Query input of shape (batch_size, seq_len, d_model).
            context (tf.Tensor, optional): Key/value input for cross-attention. If None, performs self-attention.
            attn_mask (tf.Tensor, optional): Attention mask.
            q_valid (tf.Tensor, optional): Query validity mask.
            training (bool): Whether in training mode.
            need_attn (bool): Whether to return attention scores.
            
        Returns:
            tf.Tensor or tuple: Output tensor, optionally with attention scores if need_attn=True.
        """
        xn = self.norm(x)
        # Normalize attention_mask semantics across TF versions:
        # Convert boolean "allowed" masks to additive bias: 0 for allowed, -1e9 for blocked.
        attn_bias = None
        if attn_mask is not None:
            if attn_mask.dtype == tf.bool:
                # Here, True means "allowed" in our code. Convert to bias where blocked -> -1e9
                zeros = tf.zeros_like(tf.cast(attn_mask, tf.float32))
                neginf = tf.fill(tf.shape(attn_mask), tf.constant(-1e9, tf.float32))
                attn_bias = tf.where(attn_mask, zeros, neginf)
            else:
                attn_bias = tf.cast(attn_mask, tf.float32)

        if context is None:
            if need_attn:
                h, scores = self.mha(query=xn, key=xn, value=xn, attention_mask=attn_bias,
                                     training=training, return_attention_scores=True)
            else:
                h = self.mha(query=xn, key=xn, value=xn, attention_mask=attn_bias, training=training)
                scores = None
        else:
            if need_attn:
                h, scores = self.mha(query=xn, key=context, value=context, attention_mask=attn_bias,
                                     training=training, return_attention_scores=True)
            else:
                h = self.mha(query=xn, key=context, value=context, attention_mask=attn_bias, training=training)
                scores = None
        if q_valid is not None:
            h *= tf.cast(q_valid[..., None], h.dtype)
        h = self.res_dropout(h, training=training)
        if self.drop_path is not None:
            h = self.drop_path(h, training=training)
        out = self.add([x, h])
        if need_attn:
            return out, scores
        return out

class FeedForwardPreNorm(layers.Layer):
    """
    Feed-forward network with pre-normalization and residual connection.
    
    Implements a two-layer FFN with GELU activation, pre-LayerNorm,
    dropout, and optional DropPath for regularization.
    
    Args:
        d_model (int): Model dimension.
        dff (int): Feed-forward intermediate dimension.
        dropout (float): Dropout rate. Default: 0.1.
        droppath (float): DropPath rate. Default: 0.0.
        name (str, optional): Layer name.
    """
    
    def __init__(self, d_model, dff, dropout=0.1, droppath=0.0, name=None):
        super().__init__(name=name)
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="ln")
        self.seq = tf.keras.Sequential([
            layers.Dense(dff, activation=tf.keras.activations.gelu, name="fc1"),
            layers.Dropout(dropout, name="drop"),
            layers.Dense(d_model, name="fc2"),
        ], name="ff")
        self.res_dropout = layers.Dropout(dropout, name="res_drop")
        self.drop_path = DropPath(droppath, name="droppath") if droppath > 0 else None
        self.add = layers.Add(name="add")

    def call(self, x, training=False):
        """
        Apply feed-forward network with pre-normalization.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            training (bool): Whether in training mode.
            
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        h = self.norm(x)
        h = self.seq(h, training=training)
        h = self.res_dropout(h, training=training)
        if self.drop_path is not None:
            h = self.drop_path(h, training=training)
        return self.add([x, h])

# =========================
# Encoder / Decoder Layers
# =========================
class EncoderLayer(layers.Layer):
    """
    Single Transformer encoder layer with self-attention and feed-forward network.
    
    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dff (int): Feed-forward intermediate dimension.
        dropout_rate (float): Dropout rate. Default: 0.1.
        droppath (float): DropPath rate. Default: 0.0.
        name (str, optional): Layer name.
    """
    
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, droppath=0.0, name=None):
        super().__init__(name=name)
        self.self_attention = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, droppath=droppath, name="self_attn"
        )
        self.ffn = FeedForwardPreNorm(d_model, dff, dropout=dropout_rate, droppath=droppath, name="ffn")

    def call(self, x, key_mask=None, q_valid=None, training=False):
        """
        Process input through encoder layer.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_mask (tf.Tensor, optional): Key padding mask.
            q_valid (tf.Tensor, optional): Query validity mask.
            training (bool): Whether in training mode.
            
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.self_attention(x, attn_mask=key_mask, q_valid=q_valid, training=training)
        x = self.ffn(x, training=training)
        return x

class DecoderLayer(layers.Layer):
    """
    Single Transformer decoder layer with causal self-attention, cross-attention, and FFN.
    
    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dff (int): Feed-forward intermediate dimension.
        dropout_rate (float): Dropout rate. Default: 0.1.
        droppath (float): DropPath rate. Default: 0.0.
        name (str, optional): Layer name.
    """
    
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, droppath=0.0, name=None):
        super().__init__(name=name)
        self.causal_self_attention = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, droppath=droppath, name="causal_self_attn"
        )
        self.cross_attention = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, droppath=droppath, name="cross_attn"
        )
        self.ffn = FeedForwardPreNorm(d_model, dff, dropout=dropout_rate, droppath=droppath, name="ffn")

    def call(self, x, context, dec_causal_key_mask=None, enc_key_mask=None, q_valid=None, training=False, need_attn=False):
        """
        Process input through decoder layer.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            context (tf.Tensor): Encoder output for cross-attention of shape (batch_size, src_len, d_model).
            dec_causal_key_mask (tf.Tensor, optional): Causal mask for decoder self-attention.
            enc_key_mask (tf.Tensor, optional): Encoder key padding mask.
            q_valid (tf.Tensor, optional): Query validity mask.
            training (bool): Whether in training mode.
            need_attn (bool): Whether to return cross-attention scores.
            
        Returns:
            tf.Tensor or tuple: Output tensor, optionally with attention scores if need_attn=True.
        """
        x = self.causal_self_attention(x, attn_mask=dec_causal_key_mask, q_valid=q_valid, training=training)
        if need_attn:
            x, scores = self.cross_attention(x, context=context, attn_mask=enc_key_mask, q_valid=q_valid,
                                             training=training, need_attn=True)
        else:
            x = self.cross_attention(x, context=context, attn_mask=enc_key_mask, q_valid=q_valid, training=training)
            scores = None
        x = self.ffn(x, training=training)
        if need_attn:
            return x, scores
        return x

# =========================
# Encoder / Decoder Stacks
# =========================
class Encoder(tf.keras.Model):
    """
    Transformer encoder stack for text encoding.
    
    Stacks multiple encoder layers with optional stochastic depth (DropPath).
    Includes token embedding with positional encoding and input dropout.
    
    Args:
        num_layers (int): Number of encoder layers.
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dff (int): Feed-forward intermediate dimension.
        vocab_size (int): Vocabulary size.
        dropout_rate (float): Dropout rate. Default: 0.1.
        droppath_rate (float): Maximum DropPath rate (linearly increases across layers). Default: 0.0.
        name (str): Model name. Default: "encoder".
    """
    
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1, droppath_rate=0.0, name="encoder"):
        super().__init__(name=name)
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model, name="tokpos")
        self.dropout = layers.Dropout(dropout_rate, name="dropout")
        self.layers_ = []
        for i in range(num_layers):
            dp = 0.0 if droppath_rate == 0.0 else droppath_rate * (i / max(1, num_layers - 1))
            self.layers_.append(
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff,
                             dropout_rate=dropout_rate, droppath=dp, name=f"block_{i}")
            )

    def call(self, ids, key_mask=None, q_valid=None, training=False):
        """
        Encode input token IDs.
        
        Args:
            ids (tf.Tensor): Input token IDs of shape (batch_size, seq_len).
            key_mask (tf.Tensor, optional): Key padding mask.
            q_valid (tf.Tensor, optional): Query validity mask.
            training (bool): Whether in training mode.
            
        Returns:
            tf.Tensor: Encoded representations of shape (batch_size, seq_len, d_model).
        """
        x = self.pos_embedding(ids)
        x = self.dropout(x, training=training)
        for lyr in self.layers_:
            x = lyr(x, key_mask=key_mask, q_valid=q_valid, training=training)
        return x  # (B,S,D)

class DecoderTTS(tf.keras.Model):
    """
    Transformer decoder stack for mel-spectrogram generation.
    
    Stacks multiple decoder layers with causal self-attention and cross-attention
    to encoder outputs. Includes mel prenet for feature projection.
    
    Args:
        num_layers (int): Number of decoder layers.
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dff (int): Feed-forward intermediate dimension.
        n_mels (int): Number of mel-spectrogram bands.
        dropout_rate (float): Dropout rate. Default: 0.1.
        droppath_rate (float): Maximum DropPath rate. Default: 0.0.
        name (str): Model name. Default: "decoder_tts".
        use_prenet (bool): Whether to use prenet layers. Default: True.
        prenet_drop (float): Prenet dropout rate. Default: 0.5.
    """
    
    def __init__(self, *, num_layers, d_model, num_heads, dff, n_mels,
                 dropout_rate=0.1, droppath_rate=0.0, name="decoder_tts",
                 use_prenet=True, prenet_drop=0.5):
        super().__init__(name=name)
        self.mel_proj = MelPositionalProjection(n_mels=n_mels, d_model=d_model,
                                                name="melpos", use_prenet=use_prenet, prenet_drop=prenet_drop)
        self.dropout = layers.Dropout(dropout_rate, name="dropout")
        self.layers_ = []
        for i in range(num_layers):
            dp = 0.0 if droppath_rate == 0.0 else droppath_rate * (i / max(1, num_layers - 1))
            self.layers_.append(
                DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff,
                             dropout_rate=dropout_rate, droppath=dp, name=f"block_{i}")
            )

    def call(self, mel_in, context, dec_causal_key_mask=None, enc_key_mask=None, q_valid=None, training=False, return_attn=False):
        """
        Decode mel-spectrogram autoregressively.
        
        Args:
            mel_in (tf.Tensor): Input mel-spectrogram of shape (batch_size, time, n_mels).
            context (tf.Tensor): Encoder output of shape (batch_size, src_len, d_model).
            dec_causal_key_mask (tf.Tensor, optional): Causal mask for self-attention.
            enc_key_mask (tf.Tensor, optional): Encoder key padding mask.
            q_valid (tf.Tensor, optional): Query validity mask.
            training (bool): Whether in training mode.
            return_attn (bool): Whether to return attention weights from last layer.
            
        Returns:
            tf.Tensor or tuple: Decoded features of shape (batch_size, time, d_model),
                               optionally with attention weights if return_attn=True.
        """
        x = self.mel_proj(mel_in)
        x = self.dropout(x, training=training)
        last_scores = None
        for i, lyr in enumerate(self.layers_):
            need_attn = bool(return_attn) and (i == len(self.layers_) - 1)
            if need_attn:
                x, last_scores = lyr(x, context, dec_causal_key_mask=dec_causal_key_mask,
                                     enc_key_mask=enc_key_mask, q_valid=q_valid, training=training, need_attn=True)
            else:
                x = lyr(x, context, dec_causal_key_mask=dec_causal_key_mask,
                        enc_key_mask=enc_key_mask, q_valid=q_valid, training=training)
        if return_attn:
            # Keras MHA scores shape: (B, num_heads, T_dec, S_enc). Reduce heads → (B, T_dec, S_enc)
            if last_scores is not None:
                last_scores = tf.reduce_mean(tf.cast(last_scores, tf.float32), axis=1)
        return (x, last_scores) if return_attn else x  # (B,T,D) and optional (B,T,S)

# =========================
# PostNet
# =========================
class PostNet(layers.Layer):
    """
    Convolutional PostNet for refining mel-spectrogram predictions.
    
    Applies 4 convolutional layers with batch normalization, tanh activation, and dropout,
    followed by a final convolutional layer. The output is added as a residual to the
    initial mel prediction for refinement.
    
    Architecture: 4×(Conv1D->BN->tanh->Dropout) + 1×(Conv1D->BN) → n_mels
    
    Args:
        n_mels (int): Number of mel-spectrogram bands.
        channels (int): Number of channels in hidden layers. Default: 512.
        num_layers (int): Total number of convolutional layers. Default: 5.
        kernel_size (int): Convolution kernel size. Default: 5.
        dropout (float): Dropout rate. Default: 0.5.
        name (str): Layer name. Default: "postnet".
    """
    def __init__(self, n_mels, channels=512, num_layers=5, kernel_size=5, dropout=0.5, name="postnet"):
        super().__init__(name=name)
        assert num_layers >= 2, "PostNet needs at least 2 layers"
        self.layers_ = []
        for i in range(num_layers - 1):
            self.layers_.append(layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", name=f"conv_{i}"))
            self.layers_.append(layers.BatchNormalization(name=f"bn_{i}"))
            self.layers_.append(layers.Activation("tanh", name=f"tanh_{i}"))
            self.layers_.append(layers.Dropout(dropout, name=f"drop_{i}"))
        self.layers_.append(layers.Conv1D(filters=n_mels, kernel_size=kernel_size, padding="same", name=f"conv_out"))
        self.layers_.append(layers.BatchNormalization(name=f"bn_out"))

    def call(self, mel_pred, training=False):
        """
        Apply PostNet to refine mel-spectrogram.
        
        Args:
            mel_pred (tf.Tensor): Input mel-spectrogram of shape (batch_size, time, n_mels).
            training (bool): Whether in training mode.
            
        Returns:
            tf.Tensor: Residual refinement of shape (batch_size, time, n_mels).
        """
        x = mel_pred
        for layer in self.layers_:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x  # (B,T,n_mels)

# =========================
# TransformerTTS
# =========================
class TransformerTTS(tf.keras.Model):
    """
    Complete Transformer-based Text-to-Speech model.
    
    Combines encoder, decoder, mel prediction heads, stop token prediction, and PostNet
    for end-to-end text-to-speech synthesis. Supports both teacher-forcing training
    and autoregressive inference.
    
    Architecture:
        Text → Encoder → Decoder ← Mel (shifted right)
                    ↓
        mel_head → mel_pre → PostNet → mel_post
                    ↓
        stop_head → stop_logits
    
    Args:
        num_layers (int): Number of encoder/decoder layers.
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dff (int): Feed-forward intermediate dimension.
        input_vocab_size (int): Input vocabulary size.
        n_mels (int): Number of mel-spectrogram bands.
        dropout_rate (float): Dropout rate. Default: 0.1.
        pad_id (int): Padding token ID. Default: 1.
        droppath_rate (float): Maximum DropPath rate. Default: 0.0.
        max_length (int): Maximum sequence length. Default: 4096.
        use_prenet (bool): Whether to use prenet in decoder. Default: True.
        prenet_drop (float): Prenet dropout rate. Default: 0.5.
        name (str): Model name. Default: "TransformerTTS".
    
    Attributes:
        encoder (Encoder): Text encoder.
        decoder (DecoderTTS): Mel decoder.
        mel_head (layers.Dense): Mel prediction head.
        stop_head (layers.Dense): Stop token prediction head.
        postnet (PostNet): Mel refinement network.
        go_frame (tf.Variable): Learnable go-frame for inference.
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, n_mels,
                 dropout_rate=0.1, pad_id=1, droppath_rate=0.0, max_length=4096,
                 use_prenet=True, prenet_drop=0.5, cross_win=None, name="TransformerTTS"):
        super().__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.pad_id = int(pad_id)
        self.max_length = max_length
        self.n_mels = int(n_mels)
        # عرض پنجرهٔ قطری برای cross-attention (0..1). None یعنی بدون محدودیت اضافی
        self.cross_win = None if (cross_win is None) else float(cross_win)

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate,
                               droppath_rate=droppath_rate, name="encoder")
        self.decoder = DecoderTTS(num_layers=num_layers, d_model=d_model,
                                  num_heads=num_heads, dff=dff,
                                  n_mels=n_mels,
                                  dropout_rate=dropout_rate,
                                  droppath_rate=droppath_rate,
                                  use_prenet=use_prenet, prenet_drop=prenet_drop, name="decoder")

        self.mel_head = layers.Dense(n_mels, name="mel_head")
        self.stop_head = layers.Dense(1, name="stop_head")
        self.postnet = PostNet(n_mels=n_mels, channels=512, num_layers=5, kernel_size=5, dropout=0.5, name="postnet")

        # go-frame (قابل یادگیری) برای اینفرنس پایدارتر
        self.go_frame = self.add_weight(name="go_frame", shape=(1, 1, self.n_mels),
                                        initializer="zeros", trainable=True)

    @staticmethod
    def shift_right_mel(mel, pad_val=0.0):
        """
        Shift mel-spectrogram right by one frame for teacher forcing.
        
        Args:
            mel (tf.Tensor): Mel-spectrogram of shape (batch_size, time, n_mels).
            pad_val (float): Padding value for the first frame. Default: 0.0.
            
        Returns:
            tf.Tensor: Right-shifted mel of shape (batch_size, time, n_mels).
        """
        B = tf.shape(mel)[0]
        n_mels = tf.shape(mel)[2]
        pad = tf.fill([B, 1, n_mels], tf.cast(pad_val, mel.dtype))
        return tf.concat([pad, mel[:, :-1, :]], axis=1)

    @staticmethod
    def valid_from_mel(mel, eps=1e-6):
        """
        Create boolean mask indicating valid (non-zero) mel frames.
        
        Args:
            mel (tf.Tensor): Mel-spectrogram of shape (batch_size, time, n_mels).
            eps (float): Threshold for considering a frame as non-zero. Default: 1e-6.
            
        Returns:
            tf.Tensor: Boolean mask of shape (batch_size, time).
        """
        return tf.reduce_any(tf.math.abs(mel) > eps, axis=-1)  # (B,T)

    def _build_masks(self, enc_ids, dec_mel, enc_len=None, mel_len=None):
        """
        Build attention masks for encoder and decoder.
        
        Args:
            enc_ids (tf.Tensor): Encoder input IDs of shape (batch_size, src_len).
            dec_mel (tf.Tensor): Decoder input mel of shape (batch_size, tgt_len, n_mels).
            enc_len (tf.Tensor, optional): Encoder sequence lengths.
            mel_len (tf.Tensor, optional): Mel sequence lengths.
            
        Returns:
            tuple: (enc_valid, enc_key_mask, dec_valid, dec_causal_key) masks.
        """
        enc_valid = make_padding_bool(enc_ids, self.pad_id)           # (B,S)
        enc_key_mask = key_mask_from_valid(enc_valid)                 # (B,1,S)

        if mel_len is not None:
            dec_valid = tf.sequence_mask(mel_len, maxlen=tf.shape(dec_mel)[1])  # (B,T)
        else:
            dec_valid = self.valid_from_mel(dec_mel)                  # (B,T)
        dec_causal_key = combine_causal_and_keypadding(dec_valid)     # (B,T,T)
        return enc_valid, enc_key_mask, dec_valid, dec_causal_key

    def call(self, inputs, training=False, return_attn=False):
        """
        Forward pass for training or evaluation.
        
        Args:
            inputs (tuple, list, or dict): Either (enc_ids, dec_mel) or dict with keys
                'enc_ids', 'dec_mel', and optionally 'enc_len', 'mel_len'.
            training (bool): Whether in training mode.
            return_attn (bool): Whether to return attention weights.
            
        Returns:
            tuple: (mel_pred_pre, mel_pred_post, stop_logits) or 
                   (mel_pred_pre, mel_pred_post, stop_logits, attn_last) if return_attn=True.
                   
        Shapes:
            mel_pred_pre: (batch_size, time, n_mels) - Pre-PostNet mel prediction
            mel_pred_post: (batch_size, time, n_mels) - Post-PostNet refined mel
            stop_logits: (batch_size, time, 1) - Stop token logits
            attn_last: (batch_size, time, src_len) - Cross-attention weights (if return_attn=True)
        """
        if isinstance(inputs, (list, tuple)):
            enc_ids, dec_mel = inputs
            enc_len = None
            mel_len = None
        elif isinstance(inputs, dict):
            enc_ids = inputs["enc_ids"]
            dec_mel = inputs["dec_mel"]
            enc_len = inputs.get("enc_len", None)
            mel_len = inputs.get("mel_len", None)
        else:
            raise ValueError("inputs must be (enc_ids, dec_mel) or a dict with keys enc_ids, dec_mel")

        # جایگزینی فریم آغازین با go-frame برای پایداری بیشتر
        B = tf.shape(dec_mel)[0]
        go = tf.cast(tf.broadcast_to(self.go_frame, [B, 1, self.n_mels]), dec_mel.dtype)
        dec_mel = tf.concat([go, dec_mel[:, 1:, :]], axis=1)

        enc_valid, enc_key_mask, dec_valid, dec_causal_key = self._build_masks(enc_ids, dec_mel, enc_len, mel_len)

        enc_out = self.encoder(enc_ids, key_mask=enc_key_mask, q_valid=enc_valid, training=training)

        # اگر cross_win تنظیم شده باشد، ماسک قطری (B,T,S) برای cross-attention بساز
        if self.cross_win is not None:
            T = tf.shape(dec_mel)[1]
            S = tf.shape(enc_ids)[1]
            # طول‌ها
            if enc_len is None:
                enc_len_val = tf.reduce_sum(tf.cast(enc_valid, tf.int32), axis=1)
            else:
                enc_len_val = tf.cast(enc_len, tf.int32)
            if mel_len is None:
                dec_len_val = tf.reduce_sum(tf.cast(dec_valid, tf.int32), axis=1)
            else:
                dec_len_val = tf.cast(mel_len, tf.int32)

            t_idx = tf.cast(tf.range(T)[None, :, None], tf.float32)  # (1,T,1)
            s_idx = tf.cast(tf.range(S)[None, None, :], tf.float32)  # (1,1,S)
            t_norm = t_idx / tf.maximum(tf.cast(dec_len_val[:, None, None], tf.float32) - 1.0, 1.0)
            s_norm = s_idx / tf.maximum(tf.cast(enc_len_val[:, None, None], tf.float32) - 1.0, 1.0)
            allow_diag = tf.abs(t_norm - s_norm) <= tf.cast(self.cross_win, tf.float32)  # (B,T,S)
            enc_valid_b = enc_valid[:, None, :]  # (B,1,S)
            enc_mask_for_cross = tf.logical_and(allow_diag, enc_valid_b)  # (B,T,S)
        else:
            enc_mask_for_cross = enc_key_mask

        dec_res = self.decoder(dec_mel, enc_out,
                               dec_causal_key_mask=dec_causal_key,
                               enc_key_mask=enc_mask_for_cross,
                               q_valid=dec_valid,
                               training=training,
                               return_attn=return_attn)
        if return_attn:
            dec_out, attn_last = dec_res
        else:
            dec_out = dec_res
            attn_last = None

        mel_pred_pre = tf.cast(self.mel_head(dec_out), tf.float32)
        stop_logits  = tf.cast(self.stop_head(dec_out), tf.float32)

        mel_residual = tf.cast(self.postnet(mel_pred_pre, training=training), tf.float32)
        mel_pred_post = mel_pred_pre + mel_residual

        if return_attn:
            return mel_pred_pre, mel_pred_post, stop_logits, attn_last
        return mel_pred_pre, mel_pred_post, stop_logits

    def build_for_load(self, max_src_len, max_tgt_len, dtype_ids=tf.int32, dtype_mel=tf.float32):
        """
        Build model with specific input shapes for loading weights.
        
        Args:
            max_src_len (int): Maximum source sequence length.
            max_tgt_len (int): Maximum target sequence length.
            dtype_ids (tf.dtype): Data type for input IDs. Default: tf.int32.
            dtype_mel (tf.dtype): Data type for mel-spectrogram. Default: tf.float32.
            
        Raises:
            ValueError: If max_src_len or max_tgt_len exceeds max_length.
        """
        if max_src_len > self.max_length or max_tgt_len > self.max_length:
            raise ValueError(f"max_src_len/max_tgt_len نباید از max_length ({self.max_length}) بزرگ‌تر باشند.")
        enc_in = tf.keras.Input(shape=(max_src_len,), dtype=dtype_ids, name="enc_ids")
        dec_in = tf.keras.Input(shape=(max_tgt_len, self.n_mels), dtype=dtype_mel, name="dec_mel")
        _ = self({"enc_ids": enc_in, "dec_mel": dec_in}, training=False)
        self.summary(expand_nested=True)

    # -------- greedy inference (go-frame) --------
    def greedy_generate_fast(self, enc_ids, *,
                             max_steps=600, min_steps=120,
                             stop_threshold=0.9, window=8, patience=4,
                             check_stop_every=5, verbose=True,
                             use_postnet=True, return_pre=False):
        """
        Generate mel-spectrogram autoregressively using greedy decoding.
        
        Generates mel frames one at a time starting from a learnable go-frame,
        stopping when the model predicts high stop probability consistently.
        
        Args:
            enc_ids (tf.Tensor): Encoder input IDs of shape (batch_size, src_len).
            max_steps (int): Maximum generation steps. Default: 600.
            min_steps (int): Minimum generation steps before checking stop condition. Default: 40.
            stop_threshold (float): Threshold for stop probability. Default: 0.6.
            window (int): Number of recent frames to average for stop decision. Default: 5.
            patience (int): Number of consecutive windows above threshold before stopping. Default: 3.
            check_stop_every (int): Check stop condition every N steps. Default: 10.
            verbose (bool): Whether to print progress. Default: True.
            
        Returns:
            tuple: If return_pre=False → (mel_hat, stop_probs)
                   If return_pre=True  → (mel_pre, mel_hat, stop_probs)
                   where mel_hat is postnet output if use_postnet=True, otherwise mel_pre.
        """
        B = tf.shape(enc_ids)[0]

        enc_valid   = make_padding_bool(enc_ids, self.pad_id)
        enc_keymask = key_mask_from_valid(enc_valid)
        enc_out     = self.encoder(enc_ids, key_mask=enc_keymask, q_valid=enc_valid, training=False)

        # شروع با go-frame قابل یادگیری
        mel_step   = tf.tile(self.go_frame, [B, 1, 1])
        mel_list   = []
        stop_list  = []
        strike     = 0

        for t in range(int(max_steps)):
            cur_len = tf.shape(mel_step)[1]
            dec_valid     = tf.ones([B, cur_len], dtype=tf.bool)
            dec_causalkey = combine_causal_and_keypadding(dec_valid)

            dec_h = self.decoder(mel_step, enc_out,
                                 dec_causal_key_mask=dec_causalkey,
                                 enc_key_mask=enc_keymask,
                                 q_valid=dec_valid,
                                 training=False)

            mel_next_pre = self.mel_head(dec_h)[:, -1:, :]
            stop_prob    = tf.sigmoid(self.stop_head(dec_h))[:, -1:, :]

            mel_list.append(mel_next_pre)
            stop_list.append(stop_prob)

            if verbose and (t % 50 == 49):
                tf.print("gen step", t+1, "stop~", tf.reduce_mean(stop_prob))

            if (t + 1) >= min_steps and ((t + 1) % check_stop_every) == 0:
                k = min(window, t + 1)
                last_probs = tf.concat(stop_list[-k:], axis=1)
                mean_prob  = tf.reduce_mean(last_probs, axis=1)
                if float(tf.reduce_all(mean_prob > stop_threshold).numpy()):
                    strike += 1
                    if strike >= patience:
                        if verbose: print(f"[stop @ step {t+1}]")
                        break
                else:
                    strike = 0

            mel_step = tf.concat([mel_step, mel_next_pre], axis=1)

        mel_pre  = tf.concat(mel_list, axis=1)  if mel_list  else tf.zeros([B, 0, self.n_mels], tf.float32)
        if use_postnet:
            mel_hat = mel_pre + self.postnet(mel_pre, training=False)
        else:
            mel_hat = mel_pre
        stop_probs = tf.concat(stop_list, axis=1) if stop_list else tf.zeros([B, 0, 1], tf.float32)
        if return_pre:
            return mel_pre, mel_hat, stop_probs
        return mel_hat, stop_probs
