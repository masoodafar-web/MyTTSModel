# MyTTSModel.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# =========================
# Positional Encoding
# =========================
def positional_encoding(length, d_model):
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
    def __init__(self, vocab_size, d_model, max_length=2048, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=False, name="tok_emb")
        self.pos_encoding = positional_encoding(max_length, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)                                  # (B,L,D)
        x = tf.cast(x, tf.float32)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class MelPositionalProjection(layers.Layer):
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
        x = tf.cast(mel, tf.float32)
        x = self.prenet(x)                                     # (B,T,D)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        length = tf.shape(x)[1]
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

# =========================
# Mask helpers
# =========================
def make_padding_bool(ids, pad_id):
    return tf.not_equal(ids, tf.cast(pad_id, ids.dtype))

def key_mask_from_valid(valid_bool):
    # Keras MHA expects True where allowed
    return valid_bool[:, None, :]  # (B,1,S)

def make_causal_mask(T):
    return tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)

def combine_causal_and_keypadding(dec_valid_bool):
    T = tf.shape(dec_valid_bool)[1]
    causal = make_causal_mask(T)                  # (T,T) True at <=
    key_mask = key_mask_from_valid(dec_valid_bool)  # (B,1,T)
    # Broadcast to (B,T,T), True where allowed
    return tf.logical_and(key_mask, causal[None, :, :])        # (B,T,T)

# =========================
# DropPath
# =========================
class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, name=None):
        super().__init__(name=name)
        self.drop_prob = float(drop_prob)
        self.supports_masking = True

    def call(self, x, training=False):
        if (not training) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        binary_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * binary_tensor

# =========================
# Attention Blocks (Pre-Norm)
# =========================
class BaseAttentionPreNorm(layers.Layer):
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

    def call(self, x, *, context=None, attn_mask=None, q_valid=None, training=False):
        xn = self.norm(x)
        if context is None:
            h = self.mha(query=xn, key=xn, value=xn, attention_mask=attn_mask, training=training)
        else:
            h = self.mha(query=xn, key=context, value=context, attention_mask=attn_mask, training=training)
        if q_valid is not None:
            h *= tf.cast(q_valid[..., None], h.dtype)
        h = self.res_dropout(h, training=training)
        if self.drop_path is not None:
            h = self.drop_path(h, training=training)
        return self.add([x, h])

class FeedForwardPreNorm(layers.Layer):
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
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, droppath=0.0, name=None):
        super().__init__(name=name)
        self.self_attention = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, droppath=droppath, name="self_attn"
        )
        self.ffn = FeedForwardPreNorm(d_model, dff, dropout=dropout_rate, droppath=droppath, name="ffn")

    def call(self, x, key_mask=None, q_valid=None, training=False):
        x = self.self_attention(x, attn_mask=key_mask, q_valid=q_valid, training=training)
        x = self.ffn(x, training=training)
        return x

class DecoderLayer(layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, droppath=0.0, name=None):
        super().__init__(name=name)
        self.causal_self_attention = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, droppath=droppath, name="causal_self_attn"
        )
        self.cross_attention = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, droppath=droppath, name="cross_attn"
        )
        self.ffn = FeedForwardPreNorm(d_model, dff, dropout=dropout_rate, droppath=droppath, name="ffn")

    def call(self, x, context, dec_causal_key_mask=None, enc_key_mask=None, q_valid=None, training=False):
        x = self.causal_self_attention(x, attn_mask=dec_causal_key_mask, q_valid=q_valid, training=training)
        x = self.cross_attention(x, context=context, attn_mask=enc_key_mask, q_valid=q_valid, training=training)
        x = self.ffn(x, training=training)
        return x

# =========================
# Encoder / Decoder Stacks
# =========================
class Encoder(tf.keras.Model):
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
        x = self.pos_embedding(ids)
        x = self.dropout(x, training=training)
        for lyr in self.layers_:
            x = lyr(x, key_mask=key_mask, q_valid=q_valid, training=training)
        return x  # (B,S,D)

class DecoderTTS(tf.keras.Model):
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

    def call(self, mel_in, context, dec_causal_key_mask=None, enc_key_mask=None, q_valid=None, training=False):
        x = self.mel_proj(mel_in)
        x = self.dropout(x, training=training)
        for lyr in self.layers_:
            x = lyr(x, context, dec_causal_key_mask=dec_causal_key_mask,
                    enc_key_mask=enc_key_mask, q_valid=q_valid, training=training)
        return x  # (B,T,D)

# =========================
# PostNet
# =========================
class PostNet(layers.Layer):
    """
    4×(Conv1D->BN->tanh->Dropout) + 1×(Conv1D->BN) → n_mels
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
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, n_mels,
                 dropout_rate=0.1, pad_id=1, droppath_rate=0.0, max_length=4096,
                 use_prenet=True, prenet_drop=0.5, name="TransformerTTS"):
        super().__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.pad_id = int(pad_id)
        self.max_length = max_length
        self.n_mels = int(n_mels)

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
        B = tf.shape(mel)[0]
        n_mels = tf.shape(mel)[2]
        pad = tf.fill([B, 1, n_mels], tf.cast(pad_val, mel.dtype))
        return tf.concat([pad, mel[:, :-1, :]], axis=1)

    @staticmethod
    def valid_from_mel(mel, eps=1e-6):
        return tf.reduce_any(tf.math.abs(mel) > eps, axis=-1)  # (B,T)

    def _build_masks(self, enc_ids, dec_mel, enc_len=None, mel_len=None):
        enc_valid = make_padding_bool(enc_ids, self.pad_id)           # (B,S)
        enc_key_mask = key_mask_from_valid(enc_valid)                 # (B,1,S)

        if mel_len is not None:
            dec_valid = tf.sequence_mask(mel_len, maxlen=tf.shape(dec_mel)[1])  # (B,T)
        else:
            dec_valid = self.valid_from_mel(dec_mel)                  # (B,T)
        dec_causal_key = combine_causal_and_keypadding(dec_valid)     # (B,T,T)
        return enc_valid, enc_key_mask, dec_valid, dec_causal_key

    def call(self, inputs, training=False):
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

        enc_valid, enc_key_mask, dec_valid, dec_causal_key = self._build_masks(enc_ids, dec_mel, enc_len, mel_len)

        enc_out = self.encoder(enc_ids, key_mask=enc_key_mask, q_valid=enc_valid, training=training)

        dec_out = self.decoder(dec_mel, enc_out,
                               dec_causal_key_mask=dec_causal_key,
                               enc_key_mask=enc_key_mask,
                               q_valid=dec_valid,
                               training=training)

        mel_pred_pre = tf.cast(self.mel_head(dec_out), tf.float32)
        stop_logits  = tf.cast(self.stop_head(dec_out), tf.float32)

        mel_residual = self.postnet(mel_pred_pre, training=training)
        mel_pred_post = mel_pred_pre + mel_residual

        return mel_pred_pre, mel_pred_post, stop_logits

    def build_for_load(self, max_src_len, max_tgt_len, dtype_ids=tf.int32, dtype_mel=tf.float32):
        if max_src_len > self.max_length or max_tgt_len > self.max_length:
            raise ValueError(f"max_src_len/max_tgt_len نباید از max_length ({self.max_length}) بزرگ‌تر باشند.")
        enc_in = tf.keras.Input(shape=(max_src_len,), dtype=dtype_ids, name="enc_ids")
        dec_in = tf.keras.Input(shape=(max_tgt_len, self.n_mels), dtype=dtype_mel, name="dec_mel")
        _ = self({"enc_ids": enc_in, "dec_mel": dec_in}, training=False)
        self.summary(expand_nested=True)

    # -------- greedy inference (go-frame) --------
    def greedy_generate_fast(self, enc_ids, *,
                             max_steps=600, min_steps=40,
                             stop_threshold=0.6, window=5, patience=3,
                             check_stop_every=10, verbose=True):
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
        mel_post = mel_pre + self.postnet(mel_pre, training=False)
        stop_probs = tf.concat(stop_list, axis=1) if stop_list else tf.zeros([B, 0, 1], tf.float32)
        return mel_post, stop_probs
