# MyTTSModelTrain.py
import os, datetime, logging, warnings, tensorflow as tf
from transformers import AutoTokenizer
from TTSDataLoader import (AudioCfg, TextCfg, preprocess_dataset, TTSDataset)

# GPU & logs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass
import absl.logging
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")

# ---- Dataset configs
ROOT = "../dataset/dataset_train"

tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False)
tok.src_lang = "src_lang"

audio_cfg = AudioCfg(
    sample_rate=16000, target_sample_rate=16000,
    n_fft=1024, hop_length=256, win_length=1024,
    n_mels=80, fmin=0.0, fmax=8000.0, trim_silence=False
)
text_cfg  = TextCfg(pad_id=tok.pad_token_id, bos_id=tok.bos_token_id, eos_id=tok.eos_token_id, max_text_len=256, lang_code=tok.src_lang)



print("Preprocessing (tokenize + wav→mel) ...")
items, text_ids, mels, mel_lens = preprocess_dataset(
    ROOT, audio_cfg, text_cfg, tok, metadata_name="metadata_train.csv"
)
print("Examples:", len(items), " | First mel:", mels[0].shape, "| First text len:", len(text_ids[0]))

# ---- Split train/val (بدون نشت)
BATCH_SIZE   = 4
MAX_SRC_LEN  = 256
MAX_MEL_LEN  = 2000
N_MELS       = audio_cfg.n_mels

N = len(items)
val_ratio = 0.02
n_val = max(max(1, int(N * val_ratio)), BATCH_SIZE)

train_slice = slice(0, N - n_val)
val_slice   = slice(N - n_val, N)

items_tr, items_va         = items[train_slice],   items[val_slice]
text_ids_tr, text_ids_va   = text_ids[train_slice], text_ids[val_slice]
mels_tr, mels_va           = mels[train_slice],    mels[val_slice]
mel_lens_tr, mel_lens_va   = mel_lens[train_slice], mel_lens[val_slice]

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

model_core = TransformerTTS(
    num_layers=6, d_model=256, num_heads=8, dff=1024,
    input_vocab_size=tok.vocab_size,
    n_mels=N_MELS, dropout_rate=0.5, pad_id=text_cfg.pad_id,
    use_prenet=True, prenet_drop=0.5
)
core_path="checkpoints/tts_core_last.weights.h5"
model_core.build_for_load(max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_MEL_LEN)
if os.path.exists(core_path):
    model_core.load_weights(core_path)
    print("✅ Weights loaded.")
else:
    print("⚠️ Checkpoint not found:", core_path)

# ---- Learner (custom train/test step + ماسک)
AUTO_SHIFT = False  # اگر دیتالودر dec_mel شیفت‌شده می‌دهد، False بماند

class TTSLearner(tf.keras.Model):
    def __init__(self, core, loss_weights=None, stop_pos_weight=5.0):
        super().__init__()
        self.core = core
        self.loss_weights = loss_weights or {"mel_pre": 0.5, "mel_post": 1.0, "stop": 0.5}
        self.stop_pos_weight = float(stop_pos_weight)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name="loss")
        self.val_loss   = tf.keras.metrics.Mean(name="val_loss")
        self.mae_pre    = tf.keras.metrics.Mean(name="l1_pre")
        self.mae_post   = tf.keras.metrics.Mean(name="l1_post")
        self.bce_stop   = tf.keras.metrics.Mean(name="bce_stop")
        self.stop_acc   = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="stop_acc")
        self.within2db  = tf.keras.metrics.Mean(name="within2db")

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

    @staticmethod
    def _stop_mask_from_targets(targets):
        """ماسک stop: هر جا لیبل stop موجود است، شمرده شود (معمولاً تمام فریم‌های معتبر + فریم پایان)"""
        if not (isinstance(targets, dict) and "stop" in targets):
            raise ValueError("targets must contain 'stop' for stop-mask.")
        stop = tf.cast(targets["stop"], tf.float32)               # (B,T,1) یا (B,T)
        if len(stop.shape) == 3 and stop.shape[-1] == 1:
            stop = tf.squeeze(stop, -1)                           # (B,T)
        return tf.ones_like(stop, dtype=tf.float32)               # (B,T)

    @staticmethod
    def _masked_mae(y_true, y_pred, mask):
        mask = tf.cast(mask, y_pred.dtype)                        # (B,T,1) یا (B,T,80)
        diff = tf.abs(y_pred - y_true)
        num  = tf.reduce_sum(diff * mask)
        den  = tf.reduce_sum(mask) + 1e-8
        return num / den

    def _weighted_bce_logits(self, y_true, logits, stop_mask):
        # y_true: (B,T,1) | logits: (B,T,1) | stop_mask: (B,T)
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 3 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, -1)                       # (B,T)
        logits = tf.cast(logits, tf.float32)
        if len(logits.shape) == 3 and logits.shape[-1] == 1:
            logits = tf.squeeze(logits, -1)                       # (B,T)
        stop_mask = tf.cast(stop_mask, tf.float32)                # (B,T)

        loss_el = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=logits, pos_weight=self.stop_pos_weight
        )                                                         # (B,T)
        num = tf.reduce_sum(loss_el * stop_mask)
        den = tf.reduce_sum(stop_mask) + 1e-8
        return num / den

    @staticmethod
    def _within_eps_db(y_true, y_pred, frame_mask, eps=0.1):
        """در صورتی که روی dB نیستی eps=0.1 مناسب‌تره؛ اگر dB هستی، eps~2.0 بذار."""
        err = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)    # (B,T)
        mask2 = tf.squeeze(tf.cast(frame_mask, tf.float32), -1)   # (B,T)
        good = tf.cast(err <= eps, tf.float32) * mask2
        return tf.reduce_sum(good) / (tf.reduce_sum(mask2) + 1e-8)

    def _assert_shift_ok(self, inputs, targets):
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

    def call(self, inputs, training=False):
        x = inputs
        if training and AUTO_SHIFT:
            x = dict(inputs)
            x["dec_mel"] = self.core.shift_right_mel(inputs["dec_mel"], pad_val=0.0)
        mel_pre, mel_post, stop_logits = self.core(x, training=training)
        return {"mel_pre": mel_pre, "mel_post": mel_post, "stop": stop_logits}

    # ---------- train/test ----------
    def train_step(self, data):
        if len(data) == 3:
            inputs, targets, _ = data
        else:
            inputs, targets = data

        self._assert_shift_ok(inputs, targets)

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)

            # ماسک‌ها از تارگت (نه dec_mel)
            frame_mask = self._frame_mask_from_targets(targets)   # (B,T,1)
            stop_mask  = self._stop_mask_from_targets(targets)    # (B,T)

            # --- Losses (ماسک‌دار)
            l1_pre  = self._masked_mae(targets["mel_pre"],  outputs["mel_pre"],  frame_mask)
            l1_post = self._masked_mae(targets["mel_post"], outputs["mel_post"], frame_mask)
            bce_stop= self._weighted_bce_logits(targets["stop"], outputs["stop"], stop_mask)

            loss = (
                self.loss_weights["mel_pre"]  * l1_pre +
                self.loss_weights["mel_post"] * l1_post +
                self.loss_weights["stop"]     * bce_stop
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # --- Metrics
        self.train_loss.update_state(loss)
        self.mae_pre.update_state(l1_pre)
        self.mae_post.update_state(l1_post)
        self.bce_stop.update_state(bce_stop)

        # BinaryAccuracy با ماسک stop
        stop_prob   = tf.sigmoid(outputs["stop"])                                 # (B,T,1)
        y_true_stop = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)   # (B,T)
        y_pred_stop = tf.squeeze(tf.cast(stop_prob > 0.5, tf.float32), axis=-1)   # (B,T)
        self.stop_acc.update_state(y_true_stop, y_pred_stop, sample_weight=stop_mask)

        # within-ε
        self.within2db.update_state(self._within_eps_db(targets["mel_post"], outputs["mel_post"], frame_mask, eps=0.1))

        return {
            "loss": self.train_loss.result(),
            "l1_pre": self.mae_pre.result(),
            "l1_post": self.mae_post.result(),
            "bce_stop": self.bce_stop.result(),
            "stop_acc": self.stop_acc.result(),
            "within2db": self.within2db.result(),
        }

    def test_step(self, data):
        if len(data) == 3:
            inputs, targets, _ = data
        else:
            inputs, targets = data

        outputs = self(inputs, training=False)

        frame_mask = self._frame_mask_from_targets(targets)   # (B,T,1)
        stop_mask  = self._stop_mask_from_targets(targets)    # (B,T)

        l1_pre  = self._masked_mae(targets["mel_pre"],  outputs["mel_pre"],  frame_mask)
        l1_post = self._masked_mae(targets["mel_post"], outputs["mel_post"], frame_mask)
        bce_stop= self._weighted_bce_logits(targets["stop"], outputs["stop"], stop_mask)

        loss = (
            self.loss_weights["mel_pre"]  * l1_pre +
            self.loss_weights["mel_post"] * l1_post +
            self.loss_weights["stop"]     * bce_stop
        )

        self.val_loss.update_state(loss)

        # stop accuracy (وزن‌دار)
        stop_prob   = tf.sigmoid(outputs["stop"])
        y_true_stop = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)
        y_pred_stop = tf.squeeze(tf.cast(stop_prob > 0.5, tf.float32), axis=-1)
        correct = tf.cast(tf.equal(y_pred_stop, y_true_stop), tf.float32)
        val_stop_acc = tf.reduce_sum(correct * stop_mask) / (tf.reduce_sum(stop_mask) + 1e-8)

        # within-ε
        val_within2db = self._within_eps_db(targets["mel_post"], outputs["mel_post"], frame_mask, eps=0.1)

        # کلیدها بدون پیشوند "val_" برگردند؛ Keras خودش val_ اضافه می‌کند
        return {
            "loss": self.val_loss.result(),
            "l1_pre": l1_pre,
            "l1_post": l1_post,
            "bce_stop": bce_stop,
            "stop_acc": val_stop_acc,
            "within2db": val_within2db,
        }

# ---- Optimizer / LR schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__(); self.d_model = tf.cast(d_model, tf.float32); self.warmup = float(warmup_steps)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup ** -1.5))

steps_per_epoch = len(train_gen)
optimizer = tf.keras.optimizers.Adam(
    CustomSchedule(256, warmup_steps=8*steps_per_epoch),
    beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
)

learner = TTSLearner(model_core, loss_weights={"mel_pre":0.5, "mel_post":1.0, "stop":0.5}, stop_pos_weight=5.0)
learner.compile(optimizer=optimizer, run_eagerly=False)

# ---- Callbacks
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

class SaveCore(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.core.save_weights(core_path)
        print("\n[saved core → checkpoints/tts_core_last.weights.h5]")

callbacks = [ckpt_learner, SaveCore(), tb, early]

print("Fitting ...")
learner.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks)
