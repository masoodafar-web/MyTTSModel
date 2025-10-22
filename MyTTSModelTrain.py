# MyTTSModelTrain.py
"""
Training script for MyTTSModel - Transformer-based Text-to-Speech

This module provides:
- Data preprocessing and loading for TTS training
- Custom training loop with masked losses
- Guided Attention Loss (GAL) with ramp-up
- Multiple metrics: L1 loss per mel-band, stop accuracy, within-epsilon-dB
- EMA (Exponential Moving Average) of weights
- Learning rate scheduling (Noam schedule)
- Multi-GPU support via MirroredStrategy
- Mixed precision training

The training uses:
- NLLB tokenizer for multilingual text encoding
- Mel-spectrogram preprocessing with caching
- Teacher forcing with right-shifted decoder input
- Dynamic stop token weighting based on class imbalance
"""

import os, datetime, logging, warnings, tensorflow as tf, numpy as np
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
from TTSDataLoader import (AudioCfg, TextCfg, preprocess_dataset, TTSDataset)

# GPU & logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
import absl.logging
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")
# Quiet transformers deprecation/info logs
hf_logging.set_verbosity_error()

# ---- Mixed precision (speeds up on modern GPUs) ----
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Using mixed precision: float16 compute, float32 vars")
except Exception as _e:
    print("Mixed precision not enabled:", _e)

# ---- Multi-GPU strategy (MirroredStrategy) ----
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} GPUs (MirroredStrategy)")
except Exception as e:
    print("MirroredStrategy unavailable, falling back to default. Reason:", e)
    strategy = tf.distribute.get_strategy()

# ---- Dataset configs
ROOT = "../dataset/dataset_train"

LANG_CODE = "eng_Latn"
tok = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", use_fast=False, src_lang=LANG_CODE
)

audio_cfg = AudioCfg(
    sample_rate=16000, target_sample_rate=16000,
    n_fft=1024, hop_length=256, win_length=1024,
    n_mels=80, fmin=0.0, fmax=8000.0, trim_silence=False
)
text_cfg  = TextCfg(
    pad_id=tok.pad_token_id,
    bos_id=tok.bos_token_id,
    eos_id=tok.eos_token_id,
    max_text_len=256,
    lang_code=LANG_CODE,
)



print("Preprocessing (tokenize + wav→mel) ...")
# فعال‌سازی کشِ مِل + پردازش موازی برای سرعت بهتر
cache_dir = os.path.join("checkpoints", "mel_cache")
os.makedirs(cache_dir, exist_ok=True)
num_workers = max(1, (os.cpu_count() or 2) - 1)
items, text_ids, mels, mel_lens = preprocess_dataset(
    ROOT, audio_cfg, text_cfg, tok,
    metadata_name="metadata_train.csv",
    num_workers=num_workers,
    cache_dir=cache_dir,
)
print("Examples:", len(items), " | First mel:", mels[0].shape, "| First text len:", len(text_ids[0]))
try:
    max_tok_id = int(max((max(t) if len(t)>0 else 0) for t in text_ids)) if len(text_ids)>0 else 0
except Exception:
    max_tok_id = 0
print(f"Tokenizer size: vocab={tok.vocab_size} | len(tokenizer)={len(tok)} | max_id_in_data={max_tok_id}")

# ---- Split train/val (بدون نشت)
BATCH_SIZE   = 4
# طول‌ها را بر اساس صدک‌ها تعیین کن تا پَدینگ کمتر شود
text_lens = np.asarray([len(t) for t in text_ids], dtype=np.int32)
mel_lens  = np.asarray(mel_lens, dtype=np.int32)
MAX_SRC_LEN  = int(min(256, max(8, np.percentile(text_lens, 99) + 8)))
MAX_MEL_LEN  = int(min(2000, max(64, np.percentile(mel_lens, 99) + 16)))
N_MELS       = audio_cfg.n_mels

N = len(items)
val_ratio = 0.02
# حداقل 1 نمونه برای val، و اطمینان از یک batch برای train
n_val = max(1, int(N * val_ratio))
if (N - n_val) < BATCH_SIZE:
    n_val = max(1, N - BATCH_SIZE)
perm = np.random.RandomState(42).permutation(N)
tr_idx, va_idx = perm[: N - n_val], perm[N - n_val :]

items_tr = [items[i] for i in tr_idx]
items_va = [items[i] for i in va_idx]
text_ids_tr = [text_ids[i] for i in tr_idx]
text_ids_va = [text_ids[i] for i in va_idx]
mels_tr = [mels[i] for i in tr_idx]
mels_va = [mels[i] for i in va_idx]
mel_lens_tr = mel_lens[tr_idx]
mel_lens_va = mel_lens[va_idx]

# اطمینان از سازگاری اندازه‌ی بچ با تعداد GPUها (global batch size)
try:
    num_repl = strategy.num_replicas_in_sync
    if BATCH_SIZE % max(1, num_repl) != 0:
        new_bs = num_repl * ((BATCH_SIZE + num_repl - 1) // num_repl)
        print(f"Adjusting BATCH_SIZE {BATCH_SIZE} -> {new_bs} for {num_repl} replicas")
        BATCH_SIZE = new_bs
except Exception:
    pass

train_gen = TTSDataset(
    text_ids_list=text_ids_tr,
    mels_list=mels_tr,
    batch_size=BATCH_SIZE,
    pad_id=text_cfg.pad_id,
    n_mels=N_MELS,
    max_src_len=MAX_SRC_LEN,
    max_mel_len=MAX_MEL_LEN,
    shuffle=True
)
val_gen = TTSDataset(
    text_ids_list=text_ids_va,
    mels_list=mels_va,
    batch_size=BATCH_SIZE,
    pad_id=text_cfg.pad_id,
    n_mels=N_MELS,
    max_src_len=MAX_SRC_LEN,
    max_mel_len=MAX_MEL_LEN,
    shuffle=False
)

# ---- Model
from MyTTSModel import TransformerTTS

core_path = "checkpoints/tts_core_last.weights.h5"
# مدل را با اندازه واژگان len(tokenizer) می‌سازیم تا added tokens را پوشش دهد
NUM_LAYERS = 12
D_MODEL = 512
DFF = 2024
NUM_HEADS = 8
with strategy.scope():
    model_core = TransformerTTS(
        num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF,
        input_vocab_size=len(tok),
        n_mels=N_MELS, dropout_rate=0.1, pad_id=text_cfg.pad_id,
        use_prenet=True, prenet_drop=0.5
    )
    model_core.build_for_load(max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_MEL_LEN)
    if os.path.exists(core_path):
        try:
            model_core.load_weights(core_path)
            print("✅ Weights loaded.")
        except Exception as e:
            print("⚠️ Failed to load weights strictly:", e)
            try:
                model_core.load_weights(core_path, skip_mismatch=True)
                print("✅ Weights loaded with skip_mismatch.")
            except Exception as e2:
                print("⚠️ Skipped loading weights due to mismatch:", e2)
    else:
        print("⚠️ Checkpoint not found:", core_path)

# ---- Learner (custom train/test step + ماسک)
AUTO_SHIFT = False  # اگر دیتالودر dec_mel شیفت‌شده می‌دهد، False بماند

class TTSLearner(tf.keras.Model):
    """
    Custom training wrapper for TransformerTTS with masked losses and metrics.
    
    Implements custom train_step and test_step with:
    - Masked L1 loss for mel-spectrogram (mean per band)
    - Weighted binary cross-entropy for stop tokens with dynamic class weighting
    - Guided Attention Loss (GAL) with configurable weight and sigma
    - Multiple training metrics including within-epsilon-dB accuracy
    
    Args:
        core (TransformerTTS): The core TTS model.
        loss_weights (dict, optional): Weights for different loss components.
            Default: {"mel_pre": 0.5, "mel_post": 1.0, "stop": 0.5}.
        stop_pos_weight (float, optional): Positive class weight for stop BCE.
            If None or <=0, computed dynamically from batch statistics.
        ga_weight (float): Initial weight for Guided Attention Loss. Default: 0.2.
        ga_sigma (float): Sigma parameter for GAL diagonal penalty. Default: 0.2.
        
    Attributes:
        core (TransformerTTS): The wrapped TTS model.
        ga_weight_var (tf.Variable): Current GAL weight (can be ramped during training).
        train_loss, val_loss (tf.keras.metrics.Mean): Training and validation loss trackers.
        mae_pre, mae_post (tf.keras.metrics.Mean): L1 loss trackers for pre/post mel.
        bce_stop (tf.keras.metrics.Mean): BCE loss tracker for stop tokens.
        stop_acc (tf.keras.metrics.BinaryAccuracy): Stop token prediction accuracy.
        within2db (tf.keras.metrics.Mean): Percentage of mel bins within 2dB error.
        ga_metric (tf.keras.metrics.Mean): Guided attention penalty tracker.
    """
    def __init__(self, core, loss_weights=None, stop_pos_weight=None, ga_weight=0.2, ga_sigma=0.2):
        super().__init__()
        self.core = core
        self.loss_weights = loss_weights or {"mel_pre": 0.5, "mel_post": 1.0, "stop": 0.5}
        # None یا <=0 یعنی محاسبه پویا از نسبت کلاس‌ها در هر بچ
        self.stop_pos_weight = None if (stop_pos_weight is None) else float(stop_pos_weight)
        # Guided Attention
        self.ga_weight = float(ga_weight)
        self.ga_sigma = float(ga_sigma)
        # وزن GAL به‌صورت متغیر تا در طول آموزش ramp شود
        self.ga_weight_var = tf.Variable(self.ga_weight, trainable=False, dtype=tf.float32, name="ga_weight")

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name="loss")
        self.val_loss   = tf.keras.metrics.Mean(name="val_loss")
        self.mae_pre    = tf.keras.metrics.Mean(name="l1_pre")
        self.mae_post   = tf.keras.metrics.Mean(name="l1_post")
        self.bce_stop   = tf.keras.metrics.Mean(name="bce_stop")
        self.stop_acc   = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="stop_acc")
        self.within2db  = tf.keras.metrics.Mean(name="within2db")
        self.ga_metric  = tf.keras.metrics.Mean(name="gal")

    # ---------- helpers ----------
    @staticmethod
    def _frame_mask_from_targets(targets):
        """ماسک فریم برای لُس مل از خود تارگت مل (جلوگیری از حذف فریم پایان)"""
        y = None
        if isinstance(targets, dict):
            if "mel_post" in targets:
                y = targets["mel_post"]
            elif "mel_pre" in targets:
                y = targets["mel_pre"]
        if y is None:
            raise ValueError("targets must contain 'mel_post' or 'mel_pre' for mask building.")
        mask_bt = tf.reduce_any(tf.math.abs(y) > 1e-6, axis=-1)   # (B,T) bool
        return tf.cast(mask_bt[..., None], tf.float32)            # (B,T,1)

    def _stop_mask(self, inputs, targets):
        """ماسک stop مبتنی بر mel_len: فریم‌های [0..mel_len] در محاسبه لحاظ می‌شود.
        یک فریم پس از پایان را نیز شامل می‌کنیم تا برچسب مثبت stop دیده شود."""
        if not (isinstance(targets, dict) and "stop" in targets):
            raise ValueError("targets must contain 'stop' for stop-mask.")
        stop = tf.cast(targets["stop"], tf.float32)
        T = tf.shape(stop)[1]
        mel_len = None
        if isinstance(inputs, dict) and ("mel_len" in inputs):
            mel_len = tf.cast(inputs["mel_len"], tf.int32)
        if mel_len is None:
            if len(stop.shape) == 3 and stop.shape[-1] == 1:
                stop = tf.squeeze(stop, -1)
            return tf.ones_like(stop, dtype=tf.float32)
        mask = tf.sequence_mask(mel_len + 1, maxlen=T, dtype=tf.float32)  # (B,T)
        return mask

    @staticmethod
    def _masked_mae(y_true, y_pred, mask):
        """
        Compute masked mean absolute error.
        
        Args:
            y_true (tf.Tensor): Ground truth values.
            y_pred (tf.Tensor): Predicted values.
            mask (tf.Tensor): Binary mask indicating valid positions.
            
        Returns:
            tf.Tensor: Scalar MAE loss over valid positions.
        """
        mask = tf.cast(mask, y_pred.dtype)                        # (B,T,1) یا (B,T,80)
        diff = tf.abs(y_pred - y_true)
        num  = tf.reduce_sum(diff * mask)
        den  = tf.reduce_sum(mask) + 1e-8
        return num / den

    @staticmethod
    def _masked_mae_mean_per_band(y_true, y_pred, mask):
        """میانگین L1 روی محور مل برای خواناییِ متریک، سپس ماسک فریم.
        توجه: فقط برای متریک استفاده می‌شود؛ لُس اصلی تغییر نمی‌کند."""
        mask = tf.cast(mask, y_pred.dtype)                        # (B,T,1) یا (B,T,80)
        diff = tf.abs(y_pred - y_true)
        diff = tf.reduce_mean(diff, axis=-1, keepdims=True)       # (B,T,1)
        num  = tf.reduce_sum(diff * mask)
        den  = tf.reduce_sum(mask) + 1e-8
        return num / den

    def _weighted_bce_logits(self, y_true, logits, stop_mask):
        """
        Compute weighted binary cross-entropy with dynamic positive class weighting.
        
        Automatically balances positive/negative classes by computing pos_weight
        as the ratio of negative to positive samples in the batch.
        
        Args:
            y_true (tf.Tensor): Ground truth stop labels of shape (B, T, 1) or (B, T).
            logits (tf.Tensor): Predicted stop logits of shape (B, T, 1) or (B, T).
            stop_mask (tf.Tensor): Binary mask of shape (B, T) indicating valid frames.
            
        Returns:
            tf.Tensor: Scalar masked BCE loss.
        """
        # y_true: (B,T,1) | logits: (B,T,1) | stop_mask: (B,T)
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 3 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, -1)                       # (B,T)
        logits = tf.cast(logits, tf.float32)
        if len(logits.shape) == 3 and logits.shape[-1] == 1:
            logits = tf.squeeze(logits, -1)                       # (B,T)
        stop_mask = tf.cast(stop_mask, tf.float32)                # (B,T)

        # محاسبهٔ پویا: pos_weight = neg/pos روی فریم‌های ماسک‌شده
        if (self.stop_pos_weight is None) or (self.stop_pos_weight <= 0.0):
            pos = tf.reduce_sum(y_true * stop_mask) + 1e-8
            neg = tf.reduce_sum((1.0 - y_true) * stop_mask) + 1e-8
            pos_weight_val = neg / pos
        else:
            pos_weight_val = tf.constant(self.stop_pos_weight, dtype=tf.float32)

        loss_el = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=logits, pos_weight=pos_weight_val
        )  # (B,T)
        num = tf.reduce_sum(loss_el * stop_mask)
        den = tf.reduce_sum(stop_mask) + 1e-8
        return num / den

    @staticmethod
    def _within_eps_db(y_true, y_pred, frame_mask, eps=2.0):
        """درصد بن‌های مِل که در بازه eps dB هستند، روی همه فریم‌های معتبر.
        فقط برای متریک (بدون اثر روی آموزش)."""
        # اطمینان از محدودهٔ ورودی‌ها برای تبدیل dB
        x_t = tf.clip_by_value(tf.cast(y_true, tf.float32), -1.0, 1.0)
        x_p = tf.clip_by_value(tf.cast(y_pred, tf.float32), -1.0, 1.0)
        to_db = lambda x: ((x + 1.0) * 0.5) * 100.0 - 100.0
        t_db = to_db(x_t)                       # (B,T,M)
        p_db = to_db(x_p)                       # (B,T,M)
        diff_db = tf.abs(p_db - t_db)           # (B,T,M)
        # ماسک فریم‌ها و نرمال‌سازی بر تعداد بن‌ها
        mask_bt1 = tf.cast(frame_mask, tf.float32)                  # (B,T,1)
        good_btm = tf.cast(diff_db <= eps, tf.float32) * mask_bt1   # (B,T,M)
        num = tf.reduce_sum(good_btm)
        den = tf.reduce_sum(mask_bt1) * tf.cast(tf.shape(diff_db)[-1], tf.float32) + 1e-8
        return num / den

    def _assert_shift_ok(self, inputs, targets):
        """
        Sanity check that decoder input is properly right-shifted from targets.
        
        Verifies that dec_mel[:, 1:3] ≈ mel_target[:, 0:2] to ensure correct
        teacher forcing setup.
        
        Args:
            inputs (dict): Input dictionary containing 'dec_mel'.
            targets (dict): Target dictionary containing 'mel_pre' or 'mel_post'.
            
        Raises:
            tf.errors.InvalidArgumentError: If assertion fails.
        """
        # چک سبک: dec_mel باید شیفت y باشد (dec[:,1] ≈ y[:,0])
        y_any = None
        if isinstance(targets, dict):
            if "mel_post" in targets:
                y_any = targets["mel_post"]
            elif "mel_pre" in targets:
                y_any = targets["mel_pre"]
        if y_any is None:
            return
        dec = inputs["dec_mel"]
        dec_ = dec[:, 1:3, :]
        y_   = y_any[:, 0:2, :]
        tf.debugging.assert_near(
            dec_, y_, atol=1e-3,
            message="[assert] Decoder input must be right-shifted targets."
        )

    # ---------- Guided Attention Terms ----------
    def _guided_attn_terms(self, attn, enc_ids, mel_len, sigma=0.2):
        """
        برمی‌گرداند:
          - ga_loss: فرمول کلاسیک GAL نرمال‌شده روی T×S (کوچک است)
          - ga_metric: میانگین پنالتی فقط روی زمان (B,T) → [0..1] برای نمایش بهتر
        """
        if attn is None:
            z = tf.constant(0.0, tf.float32)
            return z, z
        attn = tf.cast(attn, tf.float32)  # (B,T,S)

        B = tf.shape(attn)[0]
        T = tf.shape(attn)[1]
        S = tf.shape(attn)[2]

        # lengths
        enc_valid = tf.not_equal(tf.cast(enc_ids, tf.int32), tf.cast(self.core.pad_id, tf.int32))  # (B,S)
        enc_len = tf.reduce_sum(tf.cast(enc_valid, tf.int32), axis=1)  # (B,)
        dec_len = tf.cast(mel_len, tf.int32)

        # normalized coordinates
        t_idx = tf.cast(tf.range(T)[None, :, None], tf.float32)  # (1,T,1)
        s_idx = tf.cast(tf.range(S)[None, None, :], tf.float32)  # (1,1,S)
        t_norm = t_idx / tf.maximum(tf.cast(dec_len[:, None, None], tf.float32) - 1.0, 1.0)  # (B,T,1)
        s_norm = s_idx / tf.maximum(tf.cast(enc_len[:, None, None], tf.float32) - 1.0, 1.0)  # (B,1,S)
        diff = t_norm - s_norm  # (B,T,S)
        W = 1.0 - tf.exp(- (diff * diff) / (2.0 * (sigma ** 2)))  # (B,T,S)

        # masks
        dec_mask_bt = tf.sequence_mask(dec_len, maxlen=T, dtype=tf.float32)  # (B,T)
        enc_mask_bs = tf.sequence_mask(enc_len, maxlen=S, dtype=tf.float32)  # (B,S)
        mask_bts = dec_mask_bt[:, :, None] * enc_mask_bs[:, None, :]        # (B,T,S)

        # classic GA loss normalized over T×S
        num = tf.reduce_sum(W * attn * mask_bts)
        den = tf.reduce_sum(mask_bts) + 1e-8
        ga_loss = num / den

        # time-averaged metric: sum_s W*A in [0,1], average over valid T
        attn_penalty_bt = tf.reduce_sum(W * attn, axis=-1)                  # (B,T)
        num_m = tf.reduce_sum(attn_penalty_bt * dec_mask_bt)
        den_m = tf.reduce_sum(dec_mask_bt) + 1e-8
        ga_metric = num_m / den_m
        return ga_loss, ga_metric

    def call(self, inputs, training=False):
        """
        Forward pass through the model.
        
        Args:
            inputs (dict): Input dictionary with 'enc_ids', 'dec_mel', etc.
            training (bool): Whether in training mode.
            
        Returns:
            dict: Output dictionary with 'mel_pre', 'mel_post', 'stop', and 'attn' keys.
        """
        x = inputs
        if training and AUTO_SHIFT:
            x = dict(inputs)
            x["dec_mel"] = self.core.shift_right_mel(inputs["dec_mel"], pad_val=0.0)
        # درخواست attn در train/val برای GAL
        if training:
            outs = self.core(x, training=training, return_attn=True)
            mel_pre, mel_post, stop_logits, attn = outs
            return {"mel_pre": mel_pre, "mel_post": mel_post, "stop": stop_logits, "attn": attn}
        else:
            mel_pre, mel_post, stop_logits = self.core(x, training=training)
            return {"mel_pre": mel_pre, "mel_post": mel_post, "stop": stop_logits, "attn": None}

    # ---------- train/test ----------
    def train_step(self, data):
        """
        Custom training step with masked losses and guided attention.
        
        Computes:
        - Masked L1 loss for mel_pre and mel_post (mean per mel-band)
        - Weighted BCE for stop tokens with dynamic class balancing
        - Guided Attention Loss with configurable weight
        - Multiple metrics: stop accuracy, within-2dB accuracy, GAL penalty
        
        Args:
            data (tuple): (inputs, targets) or (inputs, targets, sample_weights).
            
        Returns:
            dict: Dictionary of metric names and values.
        """
        if len(data) == 3:
            inputs, targets, _ = data
        else:
            inputs, targets = data

        self._assert_shift_ok(inputs, targets)

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)

            # ماسک‌ها از تارگت (نه dec_mel)
            frame_mask = self._frame_mask_from_targets(targets)   # (B,T,1)
            stop_mask  = self._stop_mask(inputs, targets)         # (B,T)

            # --- Losses (ماسک‌دار)
            # برای سازگاری با متریک‌ها، لُس را هم به میانگین هر باند تبدیل می‌کنیم
            l1_pre_loss  = self._masked_mae_mean_per_band(targets["mel_pre"],  outputs["mel_pre"],  frame_mask)
            l1_post_loss = self._masked_mae_mean_per_band(targets["mel_post"], outputs["mel_post"], frame_mask)
            bce_stop= self._weighted_bce_logits(targets["stop"], outputs["stop"], stop_mask)

            ga_loss, ga_vis = self._guided_attn_terms(outputs.get("attn"), inputs["enc_ids"], inputs["mel_len"], sigma=self.ga_sigma)
            loss = (
                self.loss_weights["mel_pre"]  * l1_pre_loss +
                self.loss_weights["mel_post"] * l1_post_loss +
                self.loss_weights["stop"]     * bce_stop +
                tf.cast(self.ga_weight_var, tf.float32) * ga_loss
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # --- Metrics
        self.train_loss.update_state(loss)
        l1_pre_mean  = self._masked_mae_mean_per_band(targets["mel_pre"],  outputs["mel_pre"],  frame_mask)
        l1_post_mean = self._masked_mae_mean_per_band(targets["mel_post"], outputs["mel_post"], frame_mask)
        self.mae_pre.update_state(l1_pre_mean)
        self.mae_post.update_state(l1_post_mean)
        self.bce_stop.update_state(bce_stop)
        self.ga_metric.update_state(ga_vis)

        # BinaryAccuracy با ماسک stop
        stop_prob   = tf.sigmoid(outputs["stop"])                                 # (B,T,1)
        y_true_stop = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)   # (B,T)
        y_pred_stop = tf.squeeze(tf.cast(stop_prob > 0.5, tf.float32), axis=-1)   # (B,T)
        self.stop_acc.update_state(y_true_stop, y_pred_stop, sample_weight=stop_mask)

        # within-ε
        self.within2db.update_state(self._within_eps_db(targets["mel_post"], outputs["mel_post"], frame_mask, eps=2.0))

        return {
            "loss": self.train_loss.result(),
            "l1_pre": self.mae_pre.result(),   # میانگین هر باند
            "l1_post": self.mae_post.result(), # میانگین هر باند
            "bce_stop": self.bce_stop.result(),
            "stop_acc": self.stop_acc.result(),
            "within2db": self.within2db.result(),
            "gal": self.ga_metric.result(),
        }

    def test_step(self, data):
        """
        Custom validation/test step with masked losses and metrics.
        
        Similar to train_step but runs model in eval mode and computes metrics
        for monitoring validation performance.
        
        Args:
            data (tuple): (inputs, targets) or (inputs, targets, sample_weights).
            
        Returns:
            dict: Dictionary of validation metric names and values.
        """
        if len(data) == 3:
            inputs, targets, _ = data
        else:
            inputs, targets = data

        # Run core in eval mode but request attention for GA monitoring
        mel_pre_core, mel_post_core, stop_logits_core, attn_core = self.core(inputs, training=False, return_attn=True)
        outputs = {"mel_pre": mel_pre_core, "mel_post": mel_post_core, "stop": stop_logits_core, "attn": attn_core}

        frame_mask = self._frame_mask_from_targets(targets)   # (B,T,1)
        stop_mask  = self._stop_mask(inputs, targets)         # (B,T)

        l1_pre_loss  = self._masked_mae_mean_per_band(targets["mel_pre"],  outputs["mel_pre"],  frame_mask)
        l1_post_loss = self._masked_mae_mean_per_band(targets["mel_post"], outputs["mel_post"], frame_mask)
        bce_stop= self._weighted_bce_logits(targets["stop"], outputs["stop"], stop_mask)

        ga_loss, ga_vis = self._guided_attn_terms(outputs.get("attn"), inputs["enc_ids"], inputs["mel_len"], sigma=self.ga_sigma)
        loss = (
            self.loss_weights["mel_pre"]  * l1_pre_loss +
            self.loss_weights["mel_post"] * l1_post_loss +
            self.loss_weights["stop"]     * bce_stop +
            tf.cast(self.ga_weight_var, tf.float32) * ga_loss
        )

        self.val_loss.update_state(loss)

        # stop accuracy (وزن‌دار)
        stop_prob   = tf.sigmoid(outputs["stop"])
        y_true_stop = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)
        y_pred_stop = tf.squeeze(tf.cast(stop_prob > 0.5, tf.float32), axis=-1)
        correct = tf.cast(tf.equal(y_pred_stop, y_true_stop), tf.float32)
        val_stop_acc = tf.reduce_sum(correct * stop_mask) / (tf.reduce_sum(stop_mask) + 1e-8)

        # within-ε
        val_within2db = self._within_eps_db(targets["mel_post"], outputs["mel_post"], frame_mask, eps=2.0)

        # کلیدها بدون پیشوند "val_" برگردند؛ Keras خودش val_ اضافه می‌کند
        return {
            "loss": self.val_loss.result(),
            "l1_pre": self._masked_mae_mean_per_band(targets["mel_pre"],  outputs["mel_pre"],  frame_mask),
            "l1_post": self._masked_mae_mean_per_band(targets["mel_post"], outputs["mel_post"], frame_mask),
            "bce_stop": bce_stop,
            "stop_acc": val_stop_acc,
            "within2db": val_within2db,
            "gal": ga_vis,
        }

# ---- Optimizer / LR schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Noam learning rate schedule as used in "Attention is All You Need".
    
    Implements warm-up followed by inverse square root decay:
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    Args:
        d_model (int): Model dimension, used for scaling the learning rate.
        warmup_steps (int): Number of warmup steps. Default: 4000.
    """
    
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__(); self.d_model = tf.cast(d_model, tf.float32); self.warmup = float(max(1, warmup_steps))
        
    def __call__(self, step):
        """
        Compute learning rate for given training step.
        
        Args:
            step (int or tf.Tensor): Current training step.
            
        Returns:
            tf.Tensor: Learning rate for this step.
        """
        step = tf.cast(step, tf.float32)
        warm = tf.maximum(tf.constant(self.warmup, tf.float32), tf.constant(1.0, tf.float32))
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.rsqrt(step), step * (warm ** -1.5))

steps_per_epoch = max(1, len(train_gen))
with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(D_MODEL, warmup_steps=max(1, 8*steps_per_epoch)),
        beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
    )

    learner = TTSLearner(
        model_core,
        loss_weights={"mel_pre": 0.5, "mel_post": 1.0, "stop": 0.5},
        stop_pos_weight=None,
        ga_weight=0.2,
        ga_sigma=0.2,
    )
    learner.compile(optimizer=optimizer, run_eagerly=False)

# ---- Callbacks
class EMAAndSaveCore(tf.keras.callbacks.Callback):
    """
    Callback to maintain Exponential Moving Average (EMA) of model weights.
    
    Tracks EMA of all trainable variables and saves both regular and EMA
    checkpoints at the end of each epoch. EMA weights often provide better
    generalization and smoother inference.
    
    Args:
        decay (float): EMA decay rate. Default: 0.999.
        core_path (str): Path to save regular model weights.
            Default: "checkpoints/tts_core_last.weights.h5".
            
    Attributes:
        ema_path (str): Path for EMA weights (derived from core_path).
    """
    
    def __init__(self, decay=0.999, core_path="checkpoints/tts_core_last.weights.h5"):
        super().__init__()
        self.decay = float(decay)
        self._pairs = None
        self.core_path = core_path
        self.ema_path = "checkpoints/tts_core_ema_last.weights.h5"

    def on_train_begin(self, logs=None):
        """
        Initialize EMA shadow variables at the start of training.
        
        Args:
            logs (dict, optional): Training logs.
        """
        core = self.model.core
        pairs = []
        for v in core.trainable_variables:
            ema_v = tf.Variable(tf.zeros_like(v), trainable=False)
            ema_v.assign(v)
            pairs.append((ema_v, v))
        self._pairs = pairs

    def on_train_batch_end(self, batch, logs=None):
        if not self._pairs:
            return
        d = self.decay
        for ema_v, v in self._pairs:
            ema_v.assign(d * ema_v + (1.0 - d) * tf.cast(v, ema_v.dtype))

    def on_epoch_end(self, epoch, logs=None):
        core = self.model.core
        # Save regular core
        core.save_weights(self.core_path)
        msg = "\n[saved core → checkpoints/tts_core_last.weights.h5]"
        # Save EMA snapshot by swapping weights temporarily
        if self._pairs:
            orig_vals = [tf.identity(v) for _, v in self._pairs]
            for ema_v, v in self._pairs:
                v.assign(tf.cast(ema_v, v.dtype))
            core.save_weights(self.ema_path)
            for (ema_v, v), val in zip(self._pairs, orig_vals):
                v.assign(val)
            msg += " | [ema saved → checkpoints/tts_core_ema_last.weights.h5]"
        print(msg)

tb   = tf.keras.callbacks.TensorBoard(
    log_dir=f"logs/tts/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    update_freq='batch'
)
early= tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

ckpt_learner = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/tts_learner_best.weights.h5",
    save_weights_only=True, save_best_only=True,
    monitor="val_loss", mode="min", verbose=1
)

class GAWeightRamp(tf.keras.callbacks.Callback):
    """
    Callback to gradually ramp up Guided Attention Loss weight during training.
    
    Linearly increases GAL weight from start value to target value over a
    specified number of epochs. This helps stabilize early training before
    enforcing strong attention alignment.
    
    Args:
        start (float): Initial GAL weight. Default: 0.0.
        target (float): Final GAL weight. Default: 0.2.
        ramp_epochs (int): Number of epochs to ramp up. Default: 3.
    """
    
    def __init__(self, start=0.0, target=0.2, ramp_epochs=3):
        super().__init__()
        self.start = float(start)
        self.target = float(target)
        self.ramp_epochs = int(ramp_epochs)

    def on_train_begin(self, logs=None):
        """
        Set initial GAL weight at the start of training.
        
        Args:
            logs (dict, optional): Training logs.
        """
        try:
            self.model.ga_weight_var.assign(tf.cast(self.start, tf.float32))
        except Exception:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        if self.ramp_epochs <= 0:
            w = self.target
        else:
            p = min(1.0, max(0.0, (epoch + 1) / float(self.ramp_epochs)))
            w = self.start + (self.target - self.start) * p
        try:
            self.model.ga_weight_var.assign(tf.cast(w, tf.float32))
        except Exception:
            pass

callbacks = [GAWeightRamp(start=0.0, target=0.2, ramp_epochs=3), EMAAndSaveCore(decay=0.999, core_path=core_path), ckpt_learner, tb, early]

print("Fitting ...")
val_data = val_gen if len(val_gen) > 0 else None
learner.fit(train_gen, validation_data=val_data, epochs=50, callbacks=callbacks)
