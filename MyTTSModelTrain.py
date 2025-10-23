"""
MyTTSModel Training Script

A comprehensive training pipeline for Transformer-based Text-to-Speech synthesis.

Features:
- Multi-GPU training with MirroredStrategy
- Mixed precision training for performance
- Custom masked losses with dynamic weighting
- Guided Attention Loss with gradual ramp-up
- Exponential Moving Average (EMA) for model weights
- Noam learning rate scheduling
- Comprehensive metrics and monitoring

Dependencies:
- TensorFlow 2.x with mixed precision support
- Transformers library for NLLB tokenizer
- Custom TTSDataLoader for data preprocessing
"""

import os
import datetime
import logging
import warnings
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
from TTSDataLoader import AudioCfg, TextCfg, preprocess_dataset, TTSDataset


def setup_environment():
    """Configure GPU memory growth and logging levels."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # Suppress verbose logging
    import absl.logging
    tf.get_logger().setLevel(logging.ERROR)
    absl.logging.set_verbosity(absl.logging.ERROR)
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()


def setup_mixed_precision():
    """Enable mixed precision training if available."""
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("✅ Mixed precision enabled: float16 compute, float32 variables")
        return True
    except Exception as e:
        print(f"⚠️ Mixed precision not available: {e}")
        return False


def setup_strategy():
    """Initialize distributed training strategy."""
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"✅ Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        return strategy
    except Exception as e:
        print(f"⚠️ MirroredStrategy unavailable, using default strategy: {e}")
        return tf.distribute.get_strategy()


# Initialize environment
setup_environment()
mixed_precision_enabled = setup_mixed_precision()
strategy = setup_strategy()

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Dataset
    dataset_root: str = "../dataset/dataset_train"
    metadata_name: str = "metadata_train.csv"
    language_code: str = "eng_Latn"

    # Audio processing
    audio_preset: str = "base16k"

    # Training
    batch_size: int = 4
    epochs: int = 50
    validation_split: float = 0.02

    # Model
    model_preset: str = "normal"
    checkpoint_path: str = "checkpoints/tts_core_last.weights.h5"

    # Optimization
    learning_rate_warmup_steps: int = 4000
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    # Losses
    mel_pre_loss_weight: float = 0.5
    mel_post_loss_weight: float = 1.0
    stop_loss_weight: float = 0.5
    guided_attention_weight: float = 0.2
    guided_attention_sigma: float = 0.2

    # EMA
    ema_decay: float = 0.999

    # Callbacks
    early_stopping_patience: int = 8
    guided_attention_ramp_epochs: int = 3

    def __post_init__(self):
        """Override defaults with environment variables."""
        self.audio_preset = os.environ.get("TTS_AUDIO_PRESET", self.audio_preset)
        self.model_preset = os.environ.get("TTS_MODEL_PRESET", self.model_preset)


def load_tokenizer(language_code: str):
    """Load and configure NLLB tokenizer."""
    print(f"Loading NLLB tokenizer for language: {language_code}")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        use_fast=False,
        src_lang=language_code
    )
    return tokenizer


def setup_configs(tokenizer, config: TrainingConfig):
    """Create audio and text configurations."""
    from TTSConfig import make_audio_cfg, make_text_cfg

    audio_cfg = make_audio_cfg(config.audio_preset)
    text_cfg = make_text_cfg(tokenizer, config.language_code, max_text_len=256)

    print(f"Audio config: preset={config.audio_preset}, "
          f"sr={audio_cfg.target_sample_rate}, n_fft={audio_cfg.n_fft}, "
          f"hop={audio_cfg.hop_length}, n_mels={audio_cfg.n_mels}, fmax={audio_cfg.fmax}")

    return audio_cfg, text_cfg


def preprocess_data(config: TrainingConfig, audio_cfg, text_cfg, tokenizer):
    """Preprocess dataset with caching and parallel processing."""
    print("🔄 Starting data preprocessing (tokenization + mel-spectrogram extraction)...")

    # Setup caching
    cache_dir = os.path.join("checkpoints", "mel_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Determine number of workers
    num_workers = max(1, (os.cpu_count() or 2) - 1)

    # Preprocess dataset
    items, text_ids, mels, mel_lens = preprocess_dataset(
        config.dataset_root, audio_cfg, text_cfg, tokenizer,
        metadata_name=config.metadata_name,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )

    # Log statistics
    print(f"✅ Preprocessing complete:")
    print(f"   - Total examples: {len(items)}")
    print(f"   - Sample mel shape: {mels[0].shape if mels else 'N/A'}")
    print(f"   - Sample text length: {len(text_ids[0]) if text_ids else 'N/A'}")

    # Analyze tokenizer usage
    if text_ids:
        try:
            max_token_id = max((max(seq) if len(seq) > 0 else 0) for seq in text_ids)
            print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")
            print(f"   - Max token ID in data: {max_token_id}")
        except Exception as e:
            print(f"   - Warning: Could not analyze token IDs: {e}")

    return items, text_ids, mels, mel_lens


# Initialize configuration
config = TrainingConfig()
tokenizer = load_tokenizer(config.language_code)
audio_cfg, text_cfg = setup_configs(tokenizer, config)
items, text_ids, mels, mel_lens = preprocess_data(config, audio_cfg, text_cfg, tokenizer)

# Log training setup summary
print("\n" + "="*60)
print("🎯 TRAINING SETUP SUMMARY")
print("="*60)
print(f"📊 Dataset: {config.dataset_root}")
print(f"🎵 Audio: {config.audio_preset} ({audio_cfg.target_sample_rate}Hz, {audio_cfg.n_mels} mels)")
from TTSConfig import get_model_preset
model_preset_info = get_model_preset(config.model_preset)
print(f"🤖 Model: {config.model_preset} (layers={model_preset_info.num_layers})")
print(f"📈 Batch size: {config.batch_size} (adjusted for {strategy.num_replicas_in_sync} GPUs)")
print(f"🎯 Loss weights: mel_pre={config.mel_pre_loss_weight}, mel_post={config.mel_post_loss_weight}, stop={config.stop_loss_weight}")
print(f"⏰ Training: {config.epochs} epochs")
print(f"💾 Checkpoints: {config.checkpoint_path}")
print(f"📊 Monitoring: Samples every 200 steps")
print("="*60 + "\n")

def create_data_split(config: TrainingConfig, items, text_ids, mels, mel_lens, strategy):
    """Create train/validation split with proper batch size adjustment."""
    print("🔄 Creating train/validation split...")

    # Calculate sequence length statistics for padding optimization
    text_lengths = np.array([len(seq) for seq in text_ids], dtype=np.int32)
    mel_lengths = np.array(mel_lens, dtype=np.int32)

    # Use 99th percentile for max lengths to handle outliers
    max_src_len = int(min(256, max(8, np.percentile(text_lengths, 99) + 8)))
    max_mel_len = int(min(2000, max(64, np.percentile(mel_lengths, 99) + 16)))
    n_mels = audio_cfg.n_mels

    print(f"   - Max source length: {max_src_len} (99th percentile)")
    print(f"   - Max mel length: {max_mel_len} (99th percentile)")
    print(f"   - Number of mel bins: {n_mels}")

    # Create train/validation split
    total_samples = len(items)
    val_samples = max(1, int(total_samples * config.validation_split))

    # Ensure at least one batch for training
    if (total_samples - val_samples) < config.batch_size:
        val_samples = max(1, total_samples - config.batch_size)

    # Shuffle with fixed seed for reproducibility
    indices = np.random.RandomState(42).permutation(total_samples)
    train_indices = indices[:total_samples - val_samples]
    val_indices = indices[total_samples - val_samples:]

    # Split data
    train_items = [items[i] for i in train_indices]
    val_items = [items[i] for i in val_indices]
    train_text_ids = [text_ids[i] for i in train_indices]
    val_text_ids = [text_ids[i] for i in val_indices]
    train_mels = [mels[i] for i in train_indices]
    val_mels = [mels[i] for i in val_indices]
    train_mel_lens = mel_lengths[train_indices]
    val_mel_lens = mel_lengths[val_indices]

    print(f"✅ Data split complete:")
    print(f"   - Training samples: {len(train_items)}")
    print(f"   - Validation samples: {len(val_items)}")

    # Adjust batch size for distributed training
    adjusted_batch_size = adjust_batch_size_for_strategy(config.batch_size, strategy)

    return {
        'train_items': train_items,
        'val_items': val_items,
        'train_text_ids': train_text_ids,
        'val_text_ids': val_text_ids,
        'train_mels': train_mels,
        'val_mels': val_mels,
        'train_mel_lens': train_mel_lens,
        'val_mel_lens': val_mel_lens,
        'max_src_len': max_src_len,
        'max_mel_len': max_mel_len,
        'n_mels': n_mels,
        'batch_size': adjusted_batch_size
    }


def adjust_batch_size_for_strategy(batch_size: int, strategy) -> int:
    """Adjust batch size to be divisible by number of replicas."""
    try:
        num_replicas = strategy.num_replicas_in_sync
        if batch_size % max(1, num_replicas) != 0:
            new_batch_size = num_replicas * ((batch_size + num_replicas - 1) // num_replicas)
            print(f"🔧 Adjusting batch size {batch_size} -> {new_batch_size} for {num_replicas} replicas")
            return new_batch_size
    except Exception:
        pass
    return batch_size


# Create data split
data_split = create_data_split(config, items, text_ids, mels, mel_lens, strategy)

def create_data_generators(data_split, text_cfg):
    """Create training and validation data generators."""
    print("🔄 Creating data generators...")

    train_generator = TTSDataset(
        text_ids_list=data_split['train_text_ids'],
        mels_list=data_split['train_mels'],
        batch_size=data_split['batch_size'],
        pad_id=text_cfg.pad_id,
        n_mels=data_split['n_mels'],
        max_src_len=data_split['max_src_len'],
        max_mel_len=data_split['max_mel_len'],
        shuffle=True
    )

    val_generator = TTSDataset(
        text_ids_list=data_split['val_text_ids'],
        mels_list=data_split['val_mels'],
        batch_size=data_split['batch_size'],
        pad_id=text_cfg.pad_id,
        n_mels=data_split['n_mels'],
        max_src_len=data_split['max_src_len'],
        max_mel_len=data_split['max_mel_len'],
        shuffle=False
    )

    print(f"✅ Data generators created:")
    print(f"   - Training batches: {len(train_generator)}")
    print(f"   - Validation batches: {len(val_generator)}")

    return train_generator, val_generator


# Create data generators
train_generator, val_generator = create_data_generators(data_split, text_cfg)

def create_model(config: TrainingConfig, data_split, strategy, tokenizer, text_cfg):
    """Create and initialize the TTS model."""
    print("🔄 Creating TTS model...")

    from MyTTSModel import TransformerTTS
    from TTSConfig import get_model_preset

    # Get model configuration
    model_config = get_model_preset(config.model_preset)
    print(f"Model preset: {config.model_preset}")
    print(f"   - Layers: {model_config.num_layers}")
    print(f"   - Model dimension: {model_config.d_model}")
    print(f"   - Attention heads: {model_config.num_heads}")
    print(f"   - Feed-forward dimension: {model_config.dff}")
    print(f"   - Dropout rate: {model_config.dropout_rate}")

    with strategy.scope():
        model = TransformerTTS(
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            dff=model_config.dff,
            input_vocab_size=len(tokenizer),
            n_mels=data_split['n_mels'],
            dropout_rate=model_config.dropout_rate,
            pad_id=text_cfg.pad_id,
            use_prenet=getattr(model_config, 'use_prenet', True),
            prenet_drop=getattr(model_config, 'prenet_drop', 0.5),
            droppath_rate=getattr(model_config, 'droppath_rate', 0.0),
            cross_win=getattr(model_config, 'cross_win', None)
        )

        # Build model with actual input shapes
        model.build_for_load(
            max_src_len=data_split['max_src_len'],
            max_tgt_len=data_split['max_mel_len']
        )

        # Load weights if available
        if os.path.exists(config.checkpoint_path):
            success = load_model_weights(model, config.checkpoint_path)
            if not success:
                print("⚠️ Training will start from random initialization")
        else:
            print("ℹ️ No checkpoint found, starting from random initialization")

    return model


def load_model_weights(model, checkpoint_path: str) -> bool:
    """Safely load model weights with fallback options."""
    try:
        model.load_weights(checkpoint_path)
        print(f"✅ Successfully loaded weights from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to load weights (strict mode): {e}")
        try:
            model.load_weights(checkpoint_path, skip_mismatch=True)
            print(f"✅ Loaded weights with skip_mismatch from {checkpoint_path}")
            return True
        except Exception as e2:
            print(f"❌ Failed to load weights (skip_mismatch): {e2}")
            return False


# Create model
model = create_model(config, data_split, strategy, tokenizer, text_cfg)

class TTSLearner(tf.keras.Model):
    """
    Custom training wrapper for TransformerTTS with masked losses and metrics.

    This class implements custom training and validation steps with:
    - Masked L1 loss for mel-spectrograms (computed per frequency band)
    - Weighted binary cross-entropy for stop tokens with dynamic class balancing
    - Guided Attention Loss (GAL) with configurable weight and sigma
    - Comprehensive metrics including stop accuracy and dB error analysis

    Args:
        model (TransformerTTS): The core TTS transformer model.
        loss_weights (dict): Weights for different loss components.
        stop_pos_weight (float, optional): Positive class weight for stop BCE.
            If None, computed dynamically from batch statistics.
        guided_attention_weight (float): Initial weight for GAL.
        guided_attention_sigma (float): Sigma parameter for GAL diagonal penalty.
    """

    def __init__(self, model, loss_weights=None, stop_pos_weight=None,
                 guided_attention_weight=0.2, guided_attention_sigma=0.2):
        super().__init__()
        self.core = model
        self.loss_weights = loss_weights or {
            "mel_pre": 0.5,
            "mel_post": 1.0,
            "stop": 0.5
        }

        # Dynamic stop token weighting
        self.stop_pos_weight = stop_pos_weight

        # Guided Attention Loss parameters
        self.ga_weight = float(guided_attention_weight)
        self.ga_sigma = float(guided_attention_sigma)
        self.ga_weight_var = tf.Variable(
            self.ga_weight,
            trainable=False,
            dtype=tf.float32,
            name="ga_weight"
        )

        # Training metrics
        self.train_loss = tf.keras.metrics.Mean(name="loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.mae_pre = tf.keras.metrics.Mean(name="l1_pre")
        self.mae_post = tf.keras.metrics.Mean(name="l1_post")
        self.bce_stop = tf.keras.metrics.Mean(name="bce_stop")
        self.stop_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="stop_acc")
        self.within2db = tf.keras.metrics.Mean(name="within2db")
        self.ga_metric = tf.keras.metrics.Mean(name="gal")

    # ---------- Helper Methods ----------
    @staticmethod
    def _create_frame_mask(targets):
        """
        Create frame mask from mel targets to avoid penalizing padding regions.

        The mask is True for frames that contain actual mel-spectrogram data
        (where absolute values exceed a small threshold).
        """
        mel_target = targets.get("mel_post")
        if mel_target is None:
            mel_target = targets.get("mel_pre")

        if mel_target is None:
            raise ValueError("Targets must contain 'mel_post' or 'mel_pre' for mask creation.")

        # Create mask where frames have non-zero energy
        valid_frames = tf.reduce_any(tf.abs(mel_target) > 1e-6, axis=-1)  # (B, T) boolean
        return tf.cast(valid_frames[..., None], tf.float32)  # (B, T, 1)

    def _create_stop_mask(self, inputs, targets):
        """
        Create mask for stop token loss based on actual mel sequence lengths.

        Includes one extra frame after the sequence ends to ensure the stop
        token (positive label) is visible during training.
        """
        if not (isinstance(targets, dict) and "stop" in targets):
            raise ValueError("Targets must contain 'stop' for stop mask creation.")

        stop_targets = tf.cast(targets["stop"], tf.float32)
        seq_length = tf.shape(stop_targets)[1]

        # Get actual mel lengths from inputs
        mel_lengths = None
        if isinstance(inputs, dict) and "mel_len" in inputs:
            mel_lengths = tf.cast(inputs["mel_len"], tf.int32)

        if mel_lengths is None:
            # Fallback: use all frames if lengths not available
            if len(stop_targets.shape) == 3 and stop_targets.shape[-1] == 1:
                stop_targets = tf.squeeze(stop_targets, -1)
            return tf.ones_like(stop_targets, dtype=tf.float32)

        # Mask includes frames up to mel_len + 1 for stop token visibility
        return tf.sequence_mask(mel_lengths + 1, maxlen=seq_length, dtype=tf.float32)

    @staticmethod
    def _compute_masked_mae(y_true, y_pred, mask):
        """
        Compute masked Mean Absolute Error over valid positions.

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            mask: Binary mask tensor indicating valid positions

        Returns:
            Scalar MAE loss averaged over valid positions
        """
        mask = tf.cast(mask, y_pred.dtype)
        absolute_errors = tf.abs(y_pred - y_true)
        numerator = tf.reduce_sum(absolute_errors * mask)
        denominator = tf.reduce_sum(mask) + 1e-8
        return numerator / denominator

    @staticmethod
    def _compute_masked_mae_per_band(y_true, y_pred, mask):
        """
        Compute masked MAE averaged across mel frequency bands.

        This provides a more interpretable metric by averaging across
        frequency dimensions before applying the temporal mask.
        """
        mask = tf.cast(mask, y_pred.dtype)
        absolute_errors = tf.abs(y_pred - y_true)
        # Average across frequency bands (last dimension)
        band_averaged_errors = tf.reduce_mean(absolute_errors, axis=-1, keepdims=True)
        numerator = tf.reduce_sum(band_averaged_errors * mask)
        denominator = tf.reduce_sum(mask) + 1e-8
        return numerator / denominator

    def _compute_weighted_bce_logits(self, y_true, logits, stop_mask):
        """
        Compute weighted binary cross-entropy with dynamic class balancing.

        Automatically computes positive class weight as the ratio of negative
        to positive samples in the batch to handle class imbalance in stop tokens.

        Args:
            y_true: Ground truth stop labels (B, T, 1) or (B, T)
            logits: Predicted stop logits (B, T, 1) or (B, T)
            stop_mask: Binary mask indicating valid frames (B, T)

        Returns:
            Scalar masked BCE loss
        """
        # Ensure proper tensor shapes
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 3 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, -1)  # (B, T)

        logits = tf.cast(logits, tf.float32)
        if len(logits.shape) == 3 and logits.shape[-1] == 1:
            logits = tf.squeeze(logits, -1)  # (B, T)

        stop_mask = tf.cast(stop_mask, tf.float32)  # (B, T)

        # Dynamic positive class weighting
        if self.stop_pos_weight is None or self.stop_pos_weight <= 0.0:
            # Compute class ratio from batch statistics
            positive_samples = tf.reduce_sum(y_true * stop_mask) + 1e-6
            negative_samples = tf.reduce_sum((1.0 - y_true) * stop_mask) + 1e-6
            pos_weight = tf.clip_by_value(negative_samples / positive_samples, 1.0, 100.0)
        else:
            pos_weight = tf.constant(self.stop_pos_weight, dtype=tf.float32)

        # Compute weighted cross-entropy
        elementwise_loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=logits, pos_weight=pos_weight
        )  # (B, T)

        # Apply mask and compute mean
        masked_loss = elementwise_loss * stop_mask
        numerator = tf.reduce_sum(masked_loss)
        denominator = tf.reduce_sum(stop_mask) + 1e-8
        return numerator / denominator

    @staticmethod
    def _compute_within_epsilon_db(y_true, y_pred, frame_mask, eps=2.0):
        """
        Compute percentage of mel bins within epsilon dB error.

        This metric evaluates prediction accuracy in decibel space,
        which is more perceptually relevant than linear amplitude space.

        Args:
            y_true: Ground truth mel-spectrogram
            y_pred: Predicted mel-spectrogram
            frame_mask: Binary mask for valid frames
            eps: Error threshold in dB (default: 2.0)

        Returns:
            Percentage of bins within epsilon dB (0-1 range)
        """
        # Ensure inputs are in valid range for dB conversion
        y_true_clipped = tf.clip_by_value(tf.cast(y_true, tf.float32), -1.0, 1.0)
        y_pred_clipped = tf.clip_by_value(tf.cast(y_pred, tf.float32), -1.0, 1.0)

        # Convert normalized mel to dB scale
        def to_db_scale(x):
            return ((x + 1.0) * 0.5) * 100.0 - 100.0

        true_db = to_db_scale(y_true_clipped)  # (B, T, M)
        pred_db = to_db_scale(y_pred_clipped)  # (B, T, M)

        # Compute absolute dB difference
        db_errors = tf.abs(pred_db - true_db)  # (B, T, M)

        # Apply frame mask and count accurate predictions
        frame_mask_float = tf.cast(frame_mask, tf.float32)  # (B, T, 1)
        accurate_predictions = tf.cast(db_errors <= eps, tf.float32) * frame_mask_float  # (B, T, M)

        # Compute percentage across all valid bins
        total_accurate = tf.reduce_sum(accurate_predictions)
        total_valid_frames = tf.reduce_sum(frame_mask_float)
        num_mel_bins = tf.cast(tf.shape(db_errors)[-1], tf.float32)

        return total_accurate / (total_valid_frames * num_mel_bins + 1e-8)

    def _validate_teacher_forcing_shift(self, inputs, targets):
        """
        Validate that decoder inputs are properly right-shifted from targets.

        This sanity check ensures teacher forcing is implemented correctly.
        Only runs in eager mode to avoid graph compilation issues.
        """
        if not tf.executing_eagerly():
            return

        # Get target mel-spectrogram
        target_mel = None
        if isinstance(targets, dict):
            target_mel = targets.get("mel_post") or targets.get("mel_pre")

        if target_mel is None:
            return

        # Check decoder input alignment
        decoder_input = inputs["dec_mel"]
        decoder_slice = decoder_input[:, 1:3, :]  # Next 2 frames from decoder input
        target_slice = target_mel[:, 0:2, :]      # First 2 frames from target

        try:
            tf.debugging.assert_near(
                decoder_slice, target_slice, atol=1e-3,
                message="Decoder input must be right-shifted targets for teacher forcing."
            )
        except tf.errors.InvalidArgumentError:
            tf.print("⚠️ Warning: Decoder input not properly shifted for teacher forcing.")

    def _compute_guided_attention_loss(self, attention_weights, encoder_ids, mel_lengths, sigma=0.2):
        """
        Compute Guided Attention Loss (GAL) to encourage diagonal attention patterns.

        GAL penalizes attention weights that deviate from the diagonal, encouraging
        monotonic alignment between source and target sequences.

        Args:
            attention_weights: Attention weights tensor (B, T, S)
            encoder_ids: Encoder input token IDs (B, S)
            mel_lengths: Target sequence lengths (B,)
            sigma: Standard deviation for Gaussian penalty

        Returns:
            Tuple of (ga_loss, ga_metric) where:
            - ga_loss: Normalized GAL for training
            - ga_metric: Time-averaged penalty for monitoring
        """
        if attention_weights is None:
            zero_tensor = tf.constant(0.0, tf.float32)
            return zero_tensor, zero_tensor

        attention_weights = tf.cast(attention_weights, tf.float32)  # (B, T, S)
        batch_size, target_len, source_len = tf.shape(attention_weights)[0], tf.shape(attention_weights)[1], tf.shape(attention_weights)[2]

        # Compute sequence lengths
        encoder_mask = tf.not_equal(tf.cast(encoder_ids, tf.int32), tf.cast(self.core.pad_id, tf.int32))  # (B, S)
        encoder_lengths = tf.reduce_sum(tf.cast(encoder_mask, tf.int32), axis=1)  # (B,)
        decoder_lengths = tf.cast(mel_lengths, tf.int32)  # (B,)

        # Create normalized coordinate grids
        target_indices = tf.cast(tf.range(target_len)[None, :, None], tf.float32)  # (1, T, 1)
        source_indices = tf.cast(tf.range(source_len)[None, None, :], tf.float32)  # (1, 1, S)

        # Normalize by sequence lengths
        target_norm = target_indices / tf.maximum(tf.cast(decoder_lengths[:, None, None], tf.float32) - 1.0, 1.0)  # (B, T, 1)
        source_norm = source_indices / tf.maximum(tf.cast(encoder_lengths[:, None, None], tf.float32) - 1.0, 1.0)  # (B, 1, S)

        # Compute Gaussian penalty matrix
        coordinate_diff = target_norm - source_norm  # (B, T, S)
        gaussian_penalty = 1.0 - tf.exp(-tf.square(coordinate_diff) / (2.0 * (sigma ** 2)))  # (B, T, S)

        # Create validity masks
        decoder_mask = tf.sequence_mask(decoder_lengths, maxlen=target_len, dtype=tf.float32)  # (B, T)
        encoder_mask_float = tf.sequence_mask(encoder_lengths, maxlen=source_len, dtype=tf.float32)  # (B, S)
        combined_mask = decoder_mask[:, :, None] * encoder_mask_float[:, None, :]  # (B, T, S)

        # Compute normalized GAL
        weighted_attention = gaussian_penalty * attention_weights * combined_mask
        ga_loss = tf.reduce_sum(weighted_attention) / (tf.reduce_sum(combined_mask) + 1e-8)

        # Compute time-averaged metric for monitoring
        attention_penalty_per_frame = tf.reduce_sum(gaussian_penalty * attention_weights, axis=-1)  # (B, T)
        ga_metric = tf.reduce_sum(attention_penalty_per_frame * decoder_mask) / (tf.reduce_sum(decoder_mask) + 1e-8)

        return ga_loss, ga_metric

    def call(self, inputs, training=False):
        """
        Forward pass through the TTS model.

        Args:
            inputs: Dictionary containing 'enc_ids', 'dec_mel', 'mel_len', etc.
            training: Whether in training mode (affects attention computation)

        Returns:
            Dictionary with model outputs: 'mel_pre', 'mel_post', 'stop', 'attn'
        """
        model_inputs = inputs

        # Teacher forcing shift handled by data loader, not here
        # Request attention weights during training for GAL computation
        if training:
            outputs = self.core(model_inputs, training=training, return_attn=True)
            mel_pre, mel_post, stop_logits, attention = outputs
            return {
                "mel_pre": mel_pre,
                "mel_post": mel_post,
                "stop": stop_logits,
                "attn": attention
            }
        else:
            mel_pre, mel_post, stop_logits = self.core(model_inputs, training=training)
            return {
                "mel_pre": mel_pre,
                "mel_post": mel_post,
                "stop": stop_logits,
                "attn": None
            }

    def train_step(self, data):
        """
        Custom training step implementing masked losses and guided attention.

        Computes and combines multiple loss terms:
        - Masked L1 loss for mel-spectrograms (pre and post net)
        - Weighted BCE for stop tokens with dynamic class balancing
        - Guided Attention Loss for alignment regularization

        Args:
            data: Tuple of (inputs, targets) or (inputs, targets, sample_weights)

        Returns:
            Dictionary of training metrics
        """
        inputs, targets = data[0], data[1] if len(data) >= 2 else (data[0], data[1])

        # Validate teacher forcing setup
        self._validate_teacher_forcing_shift(inputs, targets)

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)

            # Create masks from targets
            frame_mask = self._create_frame_mask(targets)  # (B, T, 1)
            stop_mask = self._create_stop_mask(inputs, targets)  # (B, T)

            # Compute masked losses
            mel_pre_loss = self._compute_masked_mae_per_band(
                targets["mel_pre"], outputs["mel_pre"], frame_mask
            )
            mel_post_loss = self._compute_masked_mae_per_band(
                targets["mel_post"], outputs["mel_post"], frame_mask
            )
            stop_loss = self._compute_weighted_bce_logits(
                targets["stop"], outputs["stop"], stop_mask
            )

            # Guided Attention Loss
            ga_loss, ga_metric = self._compute_guided_attention_loss(
                outputs.get("attn"), inputs["enc_ids"], inputs["mel_len"], sigma=self.ga_sigma
            )

            # Combine losses
            total_loss = (
                self.loss_weights["mel_pre"] * mel_pre_loss +
                self.loss_weights["mel_post"] * mel_post_loss +
                self.loss_weights["stop"] * stop_loss +
                tf.cast(self.ga_weight_var, tf.float32) * ga_loss
            )

        # Handle mixed precision gradient scaling
        optimizer = self.optimizer
        if hasattr(optimizer, "get_scaled_loss"):
            scaled_loss = optimizer.get_scaled_loss(total_loss)
        else:
            scaled_loss = total_loss

        # Compute and process gradients
        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)

        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.train_loss.update_state(total_loss)
        self.mae_pre.update_state(mel_pre_loss)
        self.mae_post.update_state(mel_post_loss)
        self.bce_stop.update_state(stop_loss)
        self.ga_metric.update_state(ga_metric)

        # Stop token accuracy with masking
        stop_probabilities = tf.sigmoid(outputs["stop"])
        true_stop_labels = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)
        predicted_stop_labels = tf.squeeze(tf.cast(stop_probabilities > 0.5, tf.float32), axis=-1)
        self.stop_acc.update_state(true_stop_labels, predicted_stop_labels, sample_weight=stop_mask)

        # Within-epsilon dB accuracy
        self.within2db.update_state(
            self._compute_within_epsilon_db(targets["mel_post"], outputs["mel_post"], frame_mask, eps=2.0)
        )

        return {
            "loss": self.train_loss.result(),
            "l1_pre": self.mae_pre.result(),
            "l1_post": self.mae_post.result(),
            "bce_stop": self.bce_stop.result(),
            "stop_acc": self.stop_acc.result(),
            "within2db": self.within2db.result(),
            "gal": self.ga_metric.result(),
        }

    def test_step(self, data):
        """
        Custom validation/test step with masked losses and metrics.

        Similar to train_step but runs model in evaluation mode and computes
        metrics for monitoring validation performance.

        Args:
            data: Tuple of (inputs, targets) or (inputs, targets, sample_weights)

        Returns:
            Dictionary of validation metrics (Keras automatically adds 'val_' prefix)
        """
        inputs, targets = data[0], data[1] if len(data) >= 2 else (data[0], data[1])

        # Run model in eval mode but request attention for GAL monitoring
        mel_pre_core, mel_post_core, stop_logits_core, attn_core = self.core(
            inputs, training=False, return_attn=True
        )
        outputs = {
            "mel_pre": mel_pre_core,
            "mel_post": mel_post_core,
            "stop": stop_logits_core,
            "attn": attn_core
        }

        # Create masks
        frame_mask = self._create_frame_mask(targets)
        stop_mask = self._create_stop_mask(inputs, targets)

        # Compute losses
        mel_pre_loss = self._compute_masked_mae_per_band(targets["mel_pre"], outputs["mel_pre"], frame_mask)
        mel_post_loss = self._compute_masked_mae_per_band(targets["mel_post"], outputs["mel_post"], frame_mask)
        stop_loss = self._compute_weighted_bce_logits(targets["stop"], outputs["stop"], stop_mask)

        # Guided Attention Loss
        ga_loss, ga_metric = self._compute_guided_attention_loss(
            outputs.get("attn"), inputs["enc_ids"], inputs["mel_len"], sigma=self.ga_sigma
        )

        # Combine losses
        total_loss = (
            self.loss_weights["mel_pre"] * mel_pre_loss +
            self.loss_weights["mel_post"] * mel_post_loss +
            self.loss_weights["stop"] * stop_loss +
            tf.cast(self.ga_weight_var, tf.float32) * ga_loss
        )

        self.val_loss.update_state(total_loss)

        # Stop token accuracy with masking
        stop_probabilities = tf.sigmoid(outputs["stop"])
        true_stop_labels = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)
        predicted_stop_labels = tf.squeeze(tf.cast(stop_probabilities > 0.5, tf.float32), axis=-1)
        correct_predictions = tf.cast(tf.equal(predicted_stop_labels, true_stop_labels), tf.float32)
        val_stop_accuracy = tf.reduce_sum(correct_predictions * stop_mask) / (tf.reduce_sum(stop_mask) + 1e-8)

        # Within-epsilon dB accuracy
        val_within_2db = self._compute_within_epsilon_db(
            targets["mel_post"], outputs["mel_post"], frame_mask, eps=2.0
        )

        # Return metrics without 'val_' prefix (Keras adds it automatically)
        return {
            "loss": self.val_loss.result(),
            "l1_pre": mel_pre_loss,
            "l1_post": mel_post_loss,
            "bce_stop": stop_loss,
            "stop_acc": val_stop_accuracy,
            "within2db": val_within_2db,
            "gal": ga_metric,
        }

def create_optimizer_and_learner(config: TrainingConfig, data_split, model, strategy):
    """Create optimizer, learner, and compile the model."""
    print("🔄 Setting up optimizer and learner...")

    # Calculate steps per epoch for warmup scheduling
    steps_per_epoch = max(1, len(train_generator))

    with strategy.scope():
        # Create learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            config, data_split, steps_per_epoch
        )

        # Create optimizer with fallback for different TF versions
        optimizer = create_optimizer(lr_schedule, config)

        # Create learner model
        learner = TTSLearner(
            model,
            loss_weights={
                "mel_pre": config.mel_pre_loss_weight,
                "mel_post": config.mel_post_loss_weight,
                "stop": config.stop_loss_weight
            },
            stop_pos_weight=None,  # Dynamic weighting
            guided_attention_weight=config.guided_attention_weight,
            guided_attention_sigma=config.guided_attention_sigma,
        )

        # Compile with appropriate settings
        run_eagerly = bool(int(os.environ.get("RUN_EAGERLY", "0")))
        try:
            learner.compile(
                optimizer=optimizer,
                run_eagerly=run_eagerly,
                steps_per_execution=1  # Avoid XLA issues with MirroredStrategy
            )
        except TypeError:
            learner.compile(optimizer=optimizer, run_eagerly=run_eagerly)

    print("✅ Optimizer and learner setup complete")
    return learner


def create_learning_rate_schedule(config: TrainingConfig, data_split, steps_per_epoch):
    """Create Noam learning rate schedule."""
    from MyTTSModel import TransformerTTS
    from TTSConfig import get_model_preset

    model_config = get_model_preset(config.model_preset)
    warmup_steps = max(1, 8 * steps_per_epoch)  # 8 epochs of warmup

    return NoamLearningRateSchedule(
        model_dimension=model_config.d_model,
        warmup_steps=warmup_steps
    )


class NoamLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Noam learning rate schedule as used in "Attention is All You Need".

    Implements warm-up followed by inverse square root decay:
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, model_dimension: int, warmup_steps: int = 4000):
        super().__init__()
        self.model_dimension = tf.cast(model_dimension, tf.float32)
        self.warmup_steps = float(max(1, warmup_steps))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_factor = tf.maximum(tf.constant(self.warmup_steps, tf.float32), 1.0)

        # Noam schedule formula
        return (
            tf.math.rsqrt(self.model_dimension) *
            tf.math.minimum(
                tf.math.rsqrt(step),
                step * tf.math.pow(warmup_factor, -1.5)
            )
        )


def create_optimizer(lr_schedule, config: TrainingConfig):
    """Create AdamW optimizer with fallback for different TensorFlow versions."""
    optimizer_kwargs = {
        "learning_rate": lr_schedule,
        "weight_decay": config.weight_decay,
        "beta_1": 0.9,
        "beta_2": 0.98,
        "epsilon": 1e-9,
        "clipnorm": config.gradient_clip_norm
    }

    # Try AdamW (TensorFlow 2.11+)
    try:
        return tf.keras.optimizers.AdamW(**optimizer_kwargs)
    except AttributeError:
        pass

    # Try experimental AdamW (TensorFlow 2.10)
    try:
        from tensorflow.keras.optimizers.experimental import AdamW as AdamWExp
        return AdamWExp(**optimizer_kwargs)
    except ImportError:
        pass

    # Fallback to Adam
    print("⚠️ AdamW not available, using Adam optimizer")
    return tf.keras.optimizers.Adam(**optimizer_kwargs)


# Create optimizer and learner
learner = create_optimizer_and_learner(config, data_split, model, strategy)

def create_callbacks(config: TrainingConfig):
    """Create training callbacks for monitoring and checkpointing."""
    print("🔄 Setting up training callbacks...")

    # Enhanced TensorBoard logging with histograms and images
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/tts/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        update_freq='batch',
        profile_batch=0,  # Disable profiling for performance
        histogram_freq=1,  # Log histograms every epoch
        write_graph=True,  # Log model graph
        write_images=True,  # Log model weights as images
        embeddings_freq=1   # Log embeddings
    )

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )

    # Best model checkpoint
    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/tts_learner_best.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )

    # EMA and core model saving
    ema_callback = ExponentialMovingAverageCallback(
        decay=config.ema_decay,
        core_path=config.checkpoint_path
    )

    # Guided Attention weight ramping
    ga_ramp = GuidedAttentionRampCallback(
        start=0.0,
        target=config.guided_attention_weight,
        ramp_epochs=config.guided_attention_ramp_epochs
    )

    callbacks = [ga_ramp, ema_callback, best_checkpoint, tensorboard, early_stopping]
    print(f"✅ Created {len(callbacks)} callbacks")
    return callbacks


class ExponentialMovingAverageCallback(tf.keras.callbacks.Callback):
    """
    Callback to maintain Exponential Moving Average (EMA) of model weights.

    EMA provides better generalization and smoother inference by maintaining
    a running average of model weights during training.
    """

    def __init__(self, decay: float = 0.999, core_path: str = "checkpoints/tts_core_last.weights.h5"):
        super().__init__()
        self.decay = float(decay)
        self.core_path = core_path
        self.ema_path = core_path.replace('.weights.h5', '_ema_last.weights.h5')
        self._ema_pairs = None

    def on_train_begin(self, logs=None):
        """Initialize EMA shadow variables."""
        core_model = self.model.core
        ema_pairs = []
        for variable in core_model.trainable_variables:
            ema_variable = tf.Variable(
                tf.zeros_like(variable),
                trainable=False,
                name=f"ema_{variable.name}"
            )
            ema_variable.assign(variable)
            ema_pairs.append((ema_variable, variable))
        self._ema_pairs = ema_pairs

    def on_train_batch_end(self, batch, logs=None):
        """Update EMA weights after each batch."""
        if not self._ema_pairs:
            return

        decay = self.decay
        for ema_var, var in self._ema_pairs:
            ema_var.assign(decay * ema_var + (1.0 - decay) * tf.cast(var, ema_var.dtype))

    def on_epoch_end(self, epoch, logs=None):
        """Save model weights at epoch end."""
        core_model = self.model.core

        # Save regular weights
        core_model.save_weights(self.core_path)
        message = f"\n💾 Saved core weights → {self.core_path}"

        # Save EMA weights
        if self._ema_pairs:
            original_values = [tf.identity(var) for _, var in self._ema_pairs]
            for ema_var, var in self._ema_pairs:
                var.assign(tf.cast(ema_var, var.dtype))

            core_model.save_weights(self.ema_path)
            message += f" | EMA weights → {self.ema_path}"

            # Restore original weights
            for (_, var), original_val in zip(self._ema_pairs, original_values):
                var.assign(original_val)

        print(message)


class GuidedAttentionRampCallback(tf.keras.callbacks.Callback):
    """
    Callback to gradually ramp up Guided Attention Loss weight during training.

    Helps stabilize early training by starting with zero GAL weight and gradually
    increasing it to encourage proper attention alignment.
    """

    def __init__(self, start: float = 0.0, target: float = 0.2, ramp_epochs: int = 3):
        super().__init__()
        self.start = float(start)
        self.target = float(target)
        self.ramp_epochs = int(ramp_epochs)

    def on_train_begin(self, logs=None):
        """Set initial GAL weight."""
        try:
            self.model.ga_weight_var.assign(tf.cast(self.start, tf.float32))
        except Exception as e:
            print(f"⚠️ Could not set initial GA weight: {e}")

    def on_epoch_begin(self, epoch, logs=None):
        """Update GAL weight based on training progress."""
        if self.ramp_epochs <= 0:
            weight = self.target
        else:
            progress = min(1.0, max(0.0, (epoch + 1) / float(self.ramp_epochs)))
            weight = self.start + (self.target - self.start) * progress

        try:
            self.model.ga_weight_var.assign(tf.cast(weight, tf.float32))
            if epoch < self.ramp_epochs:
                print(f"🎯 GAL weight: {weight:.4f} (ramping to {self.target:.4f})")
        except Exception as e:
            print(f"⚠️ Could not update GA weight: {e}")


class SampleGenerationCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate sample audio during training for progress monitoring.

    Generates mel-spectrograms and audio samples at regular intervals to visually
    and audibly track training progress. Saves samples to disk for later comparison.
    """

    def __init__(self, tokenizer, text_samples, generation_interval=200, max_samples_to_keep=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_samples = text_samples
        self.generation_interval = generation_interval
        self.max_samples_to_keep = max_samples_to_keep
        self.samples_dir = "training_samples"
        self.step_count = 0
        os.makedirs(self.samples_dir, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        """Generate samples at specified intervals."""
        self.step_count += 1

        if self.step_count % self.generation_interval == 0:
            try:
                self._generate_samples()
            except Exception as e:
                print(f"⚠️ Sample generation failed: {e}")

    def _generate_samples(self):
        """Generate and save sample predictions with enhanced validation."""
        print(f"\n🎵 Generating samples at step {self.step_count}...")

        # Validate inputs
        if not self.text_samples:
            print("   ⚠️ No text samples provided")
            return

        if not hasattr(self.model, 'core') or self.model.core is None:
            print("   ⚠️ Model not properly initialized")
            return

        # Clean up old samples
        self._cleanup_old_samples()

        for i, text in enumerate(self.text_samples):
            # Validate text input
            if not isinstance(text, str) or len(text.strip()) == 0:
                print(f"   ⚠️ Invalid text sample {i}: must be non-empty string")
                continue

            try:
                # Tokenize text with validation
                tokens = self.tokenizer.encode(text, add_special_tokens=True, src_lang="eng_Latn")
                if len(tokens) == 0:
                    print(f"   ⚠️ Tokenization failed for sample {i}")
                    continue

                tokens = tokens[:256]  # Truncate if too long
                input_ids = tf.constant([tokens], dtype=tf.int32)

                # Validate input shape
                if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
                    print(f"   ⚠️ Invalid input shape for sample {i}: {input_ids.shape}")
                    continue

                # Generate mel-spectrogram with proper mixed precision handling
                # Temporarily switch to float32 policy for inference
                original_policy = tf.keras.mixed_precision.global_policy()
                try:
                    tf.keras.mixed_precision.set_global_policy('float32')

                    mel_pred, stop_probs = self.model.core.greedy_generate_fast(
                        input_ids,
                        max_steps=600,
                        min_steps=50,
                        stop_threshold=0.8,
                        verbose=False
                    )

                    # Validate outputs
                    if mel_pred is None or stop_probs is None:
                        print(f"   ⚠️ Model returned None outputs for sample {i}")
                        continue

                    # Ensure outputs are float32
                    mel_pred = tf.cast(mel_pred, tf.float32)
                    stop_probs = tf.cast(stop_probs, tf.float32)

                    # Validate output shapes
                    if mel_pred.shape[0] != 1 or mel_pred.shape[2] != audio_cfg.n_mels:
                        print(f"   ⚠️ Invalid mel output shape for sample {i}: {mel_pred.shape}")
                        continue
                    if stop_probs.shape[0] != 1:
                        print(f"   ⚠️ Invalid stop probs shape for sample {i}: {stop_probs.shape}")
                        continue

                finally:
                    # Restore original policy
                    tf.keras.mixed_precision.set_global_policy(original_policy)

                # Log to TensorBoard
                self._log_sample_to_tensorboard(text, mel_pred, stop_probs, i)

                # Save sample data
                sample_data = {
                    'text': text,
                    'mel_shape': mel_pred.shape,
                    'stop_probs_shape': stop_probs.shape,
                    'step': self.step_count
                }

                # Save numpy arrays
                np.savez_compressed(
                    f"{self.samples_dir}/sample_{i}_step_{self.step_count}.npz",
                    mel=mel_pred.numpy(),
                    stop_probs=stop_probs.numpy(),
                    metadata=str(sample_data)
                )

                print(f"   ✅ Sample {i}: '{text[:30]}...' → {mel_pred.shape[1]} frames")

            except Exception as e:
                print(f"   ❌ Sample {i} failed: {e}")

        # Generate comparison plot
        try:
            self._generate_comparison_plot()
        except Exception as e:
            print(f"   ⚠️ Comparison plot failed: {e}")

    def _cleanup_old_samples(self):
        """Keep only the most recent samples."""
        sample_files = sorted([
            f for f in os.listdir(self.samples_dir)
            if f.startswith("sample_") and f.endswith(".npz")
        ], key=lambda x: os.path.getctime(os.path.join(self.samples_dir, x)))

        if len(sample_files) > self.max_samples_to_keep * len(self.text_samples):
            files_to_remove = sample_files[:len(sample_files) - self.max_samples_to_keep * len(self.text_samples)]
            for f in files_to_remove:
                try:
                    os.remove(os.path.join(self.samples_dir, f))
                except:
                    pass

    def _generate_comparison_plot(self):
        """Generate a comparison plot of recent samples."""
        try:
            import matplotlib.pyplot as plt

            # Find recent samples
            sample_files = [
                f for f in os.listdir(self.samples_dir)
                if f.startswith("sample_0_") and f.endswith(".npz")
            ]

            if len(sample_files) < 2:
                return

            # Sort by step number
            sample_files.sort(key=lambda x: int(x.split('_step_')[1].split('.')[0]))

            # Load last few samples
            recent_samples = sample_files[-3:]  # Show last 3 samples

            plt.figure(figsize=(15, 4))

            for i, fname in enumerate(recent_samples):
                step_num = int(fname.split('_step_')[1].split('.')[0])
                data = np.load(os.path.join(self.samples_dir, fname))

                mel_db = self._mel_to_db(data['mel'][0])

                plt.subplot(1, len(recent_samples), i+1)
                plt.imshow(mel_db.T, origin='lower', aspect='auto', cmap='magma', vmin=-100, vmax=0)
                plt.title(f'Step {step_num}')
                plt.xlabel('Frames')
                plt.ylabel('Mel bins')

            plt.tight_layout()
            plt.savefig(f"{self.samples_dir}/progress_step_{self.step_count}.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   📊 Progress plot saved to {self.samples_dir}/progress_step_{self.step_count}.png")

        except ImportError:
            print("   ⚠️ Matplotlib not available for plotting")
        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")

    @staticmethod
    def _mel_to_db(mel_norm):
        """Convert normalized mel [-1,1] to dB [-100,0]."""
        mel_norm = np.clip(mel_norm, -1.0, 1.0)
        mel_01 = (mel_norm + 1.0) * 0.5  # [-1,1] -> [0,1]
        return mel_01 * 100.0 - 100.0    # -> [-100, 0] dB

    def _log_sample_to_tensorboard(self, text, mel_pred, stop_probs, sample_idx):
        """Log generated sample to TensorBoard for monitoring training progress."""
        try:
            import tensorflow as tf

            # Create summary writer if not exists
            if not hasattr(self, '_summary_writer'):
                log_dir = f"logs/tts/samples_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                self._summary_writer = tf.summary.create_file_writer(log_dir)

            with self._summary_writer.as_default():
                # Log mel-spectrogram as image
                mel_db = self._mel_to_db(mel_pred.numpy()[0])  # Convert to dB for better visualization
                mel_image = tf.expand_dims(mel_db, axis=0)  # Add batch dim
                mel_image = tf.expand_dims(mel_image, axis=-1)  # Add channel dim
                mel_image = tf.image.resize(mel_image, [128, 256])  # Resize for display

                tf.summary.image(
                    name=f"sample_{sample_idx}_mel",
                    data=mel_image,
                    step=self.step_count,
                    description=f"Generated mel-spectrogram for: '{text[:50]}...'"
                )

                # Log stop probabilities as scalar
                avg_stop_prob = tf.reduce_mean(stop_probs).numpy()
                tf.summary.scalar(
                    name=f"sample_{sample_idx}_avg_stop_prob",
                    data=avg_stop_prob,
                    step=self.step_count,
                    description="Average stop probability for generated sample"
                )

                # Log sequence length
                seq_length = mel_pred.shape[1]
                tf.summary.scalar(
                    name=f"sample_{sample_idx}_seq_length",
                    data=seq_length,
                    step=self.step_count,
                    description="Generated sequence length in frames"
                )

                # Log text as text summary
                tf.summary.text(
                    name=f"sample_{sample_idx}_text",
                    data=text,
                    step=self.step_count
                )

        except Exception as e:
            print(f"⚠️ Failed to log sample {sample_idx} to TensorBoard: {e}")


class TensorBoardAudioLogger(tf.keras.callbacks.Callback):
    """
    Callback to log generated audio samples to TensorBoard during training.

    Generates audio waveforms from mel-spectrograms and logs them to TensorBoard
    for real-time monitoring of training progress.
    """

    def __init__(self, tokenizer, sample_texts, log_interval=500, max_audio_samples=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_texts = sample_texts[:max_audio_samples]  # Limit number of samples
        self.log_interval = log_interval
        self.max_audio_samples = max_audio_samples
        self.step_count = 0
        self.writer = None

    def on_train_begin(self, logs=None):
        """Initialize TensorBoard writer."""
        try:
            import tensorflow as tf
            log_dir = f"logs/tts/audio_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.writer = tf.summary.create_file_writer(log_dir)
            print(f"🎵 TensorBoard audio logging enabled: {log_dir}")
        except Exception as e:
            print(f"⚠️ TensorBoard audio logging failed to initialize: {e}")
            self.writer = None

    def on_train_batch_end(self, batch, logs=None):
        """Generate and log audio samples at specified intervals."""
        if self.writer is None:
            return

        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            try:
                self._generate_and_log_audio()
            except Exception as e:
                print(f"⚠️ Audio logging failed: {e}")

    def _generate_and_log_audio(self):
        """Generate audio samples and log to TensorBoard."""
        if self.writer is None:
            return

        print(f"🎵 Logging audio samples to TensorBoard at step {self.step_count}...")

        with self.writer.as_default():
            for i, text in enumerate(self.sample_texts):
                try:
                    # Generate mel-spectrogram
                    audio_waveform = self._text_to_audio(text)

                    if audio_waveform is not None:
                        # Ensure audio is float32 for TensorBoard logging
                        audio_waveform = tf.cast(audio_waveform, tf.float32)

                        # Log audio to TensorBoard
                        tf.summary.audio(
                            name=f"sample_{i}_{text[:20].replace(' ', '_')}",
                            data=audio_waveform,
                            sample_rate=audio_cfg.target_sample_rate,
                            step=self.step_count,
                            encoding='wav'
                        )

                        # Also log mel-spectrogram as image
                        mel_spec = self._get_last_mel()
                        if mel_spec is not None:
                            # Convert to image format (add batch and channel dims)
                            mel_image = tf.expand_dims(mel_spec, axis=0)  # Add batch dim [1, T, M]
                            mel_image = tf.expand_dims(mel_image, axis=-1)  # Add channel dim [1, T, M, 1]
                            mel_image = tf.image.resize(mel_image, [128, 256])  # Resize for display
                            tf.summary.image(
                                name=f"mel_{i}_{text[:20].replace(' ', '_')}",
                                data=mel_image,
                                step=self.step_count
                            )

                        print(f"   ✅ Logged audio sample {i}: '{text[:30]}...'")

                except Exception as e:
                    print(f"   ❌ Failed to log sample {i}: {e}")

    def _text_to_audio(self, text):
        """Convert text to audio waveform with enhanced validation."""
        try:
            # Validate inputs
            if not isinstance(text, str) or len(text.strip()) == 0:
                print("   ⚠️ Invalid text input for audio generation")
                return None

            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print("   ⚠️ Tokenizer not available")
                return None

            # Tokenize with validation
            tokens = self.tokenizer.encode(text, add_special_tokens=True, src_lang="eng_Latn")
            if len(tokens) == 0:
                print("   ⚠️ Tokenization produced empty sequence")
                return None

            tokens = tokens[:256]  # Truncate
            input_ids = tf.constant([tokens], dtype=tf.int32)

            # Validate input shape
            if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
                print(f"   ⚠️ Invalid input shape: {input_ids.shape}")
                return None

            # Generate with proper mixed precision handling
            original_policy = tf.keras.mixed_precision.global_policy()
            try:
                tf.keras.mixed_precision.set_global_policy('float32')

                mel_pred, _ = self.model.core.greedy_generate_fast(
                    input_ids,
                    max_steps=600,
                    min_steps=50,
                    stop_threshold=0.8,
                    verbose=False
                )

                # Validate mel prediction
                if mel_pred is None:
                    print("   ⚠️ Model returned None mel prediction")
                    return None

                # Ensure float32
                mel_pred = tf.cast(mel_pred, tf.float32)

                # Validate mel shape
                if mel_pred.shape[0] != 1 or mel_pred.shape[2] != audio_cfg.n_mels:
                    print(f"   ⚠️ Invalid mel shape: {mel_pred.shape}")
                    return None

            finally:
                # Restore original policy
                tf.keras.mixed_precision.set_global_policy(original_policy)

            # Convert to audio with validation
            audio = self._mel_to_waveform(mel_pred.numpy()[0])
            if audio is None:
                print("   ⚠️ Waveform conversion failed")
                return None

            # Validate audio output
            if audio.shape[0] != 1:
                print(f"   ⚠️ Invalid audio shape: {audio.shape}")
                return None

            return audio

        except Exception as e:
            print(f"   ⚠️ Audio generation failed: {e}")
            return None

    def _get_last_mel(self):
        """Get the last generated mel-spectrogram with validation."""
        # This is a simplified version - in practice you'd store the last generated mel
        try:
            # Validate sample texts
            if not self.sample_texts or len(self.sample_texts) == 0:
                print("   ⚠️ No sample texts available")
                return None

            text = self.sample_texts[0]

            # Validate text
            if not isinstance(text, str) or len(text.strip()) == 0:
                print("   ⚠️ Invalid sample text")
                return None

            # Validate tokenizer
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print("   ⚠️ Tokenizer not available")
                return None

            tokens = self.tokenizer.encode(text, add_special_tokens=True, src_lang="eng_Latn")
            if len(tokens) == 0:
                print("   ⚠️ Tokenization failed")
                return None

            tokens = tokens[:256]
            input_ids = tf.constant([tokens], dtype=tf.int32)

            # Validate input shape
            if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
                print(f"   ⚠️ Invalid input shape: {input_ids.shape}")
                return None

            # Generate with proper mixed precision handling
            original_policy = tf.keras.mixed_precision.global_policy()
            try:
                tf.keras.mixed_precision.set_global_policy('float32')

                mel_pred, _ = self.model.core.greedy_generate_fast(
                    input_ids,
                    max_steps=600,
                    min_steps=50,
                    stop_threshold=0.8,
                    verbose=False
                )

                # Validate mel prediction
                if mel_pred is None:
                    print("   ⚠️ Model returned None mel prediction")
                    return None

                mel_pred = tf.cast(mel_pred, tf.float32)

                # Validate shape
                if mel_pred.shape[0] != 1 or mel_pred.shape[2] != audio_cfg.n_mels:
                    print(f"   ⚠️ Invalid mel shape: {mel_pred.shape}")
                    return None

            finally:
                # Restore original policy
                tf.keras.mixed_precision.set_global_policy(original_policy)

            return mel_pred.numpy()[0]  # Remove batch dimension

        except Exception as e:
            print(f"   ⚠️ Failed to get last mel: {e}")
            return None

    def _mel_to_waveform(self, mel_norm):
        """Convert normalized mel-spectrogram to audio waveform with validation."""
        try:
            # Validate input
            if mel_norm is None:
                print("   ⚠️ Input mel is None")
                return None

            mel_norm = np.asarray(mel_norm)
            if mel_norm.ndim != 2:
                print(f"   ⚠️ Invalid mel dimensions: {mel_norm.shape}")
                return None

            if mel_norm.shape[1] != audio_cfg.n_mels:
                print(f"   ⚠️ Invalid number of mel bins: {mel_norm.shape[1]}, expected {audio_cfg.n_mels}")
                return None

            # Convert mel to linear spectrogram
            mel_power = self._denorm_mel(mel_norm)
            if mel_power is None:
                return None

            linear_power = self._mel_to_linear_power(mel_power)
            if linear_power is None:
                return None

            # Convert to magnitude
            mag = tf.sqrt(tf.maximum(linear_power, 1e-10))

            # Griffin-Lim reconstruction
            waveform = self._griffin_lim(mag.numpy())
            if waveform is None:
                return None

            # Validate output
            if waveform.shape[0] != 1:
                print(f"   ⚠️ Invalid waveform batch dimension: {waveform.shape}")
                return None

            return waveform

        except Exception as e:
            print(f"   ⚠️ Waveform conversion failed: {e}")
            return None

    @staticmethod
    def _denorm_mel(mel_norm):
        """Convert normalized mel [-1,1] to power scale with validation."""
        try:
            if mel_norm is None:
                return None

            mel_norm = tf.convert_to_tensor(mel_norm, dtype=tf.float32)

            # Validate range (should be approximately [-1, 1])
            mel_min = tf.reduce_min(mel_norm)
            mel_max = tf.reduce_max(mel_norm)

            if mel_min < -1.5 or mel_max > 1.5:
                print(f"   ⚠️ Mel values outside expected range: [{mel_min:.3f}, {mel_max:.3f}]")

            mel_01 = (mel_norm + 1.0) * 0.5  # [-1,1] -> [0,1]
            mel_01 = tf.clip_by_value(mel_01, 0.0, 1.0)  # Ensure valid range

            mel_db = mel_01 * 100.0 - 100.0   # -> [-100, 0] dB
            return tf.pow(10.0, mel_db / 10.0)  # -> power

        except Exception as e:
            print(f"   ⚠️ Mel denormalization failed: {e}")
            return None

    def _mel_to_linear_power(self, mel_power):
        """Convert mel power to linear power spectrogram with validation."""
        try:
            if mel_power is None:
                return None

            mel_power = tf.convert_to_tensor(mel_power, dtype=tf.float32)

            # Validate mel power shape
            if mel_power.ndim != 2:
                print(f"   ⚠️ Invalid mel power dimensions: {mel_power.shape}")
                return None

            if mel_power.shape[1] != audio_cfg.n_mels:
                print(f"   ⚠️ Invalid number of mel bins: {mel_power.shape[1]}, expected {audio_cfg.n_mels}")
                return None

            # Get mel filterbank (cached)
            if not hasattr(self, '_mel_matrix'):
                self._mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins=audio_cfg.n_mels,
                    num_spectrogram_bins=audio_cfg.n_fft // 2 + 1,
                    sample_rate=audio_cfg.target_sample_rate,
                    lower_edge_hertz=audio_cfg.fmin,
                    upper_edge_hertz=audio_cfg.fmax,
                    dtype=tf.float32
                )

            # Apply inverse mel filterbank
            linear_power = tf.matmul(mel_power, tf.linalg.pinv(self._mel_matrix))

            # Validate output shape
            expected_bins = audio_cfg.n_fft // 2 + 1
            if linear_power.shape[1] != expected_bins:
                print(f"   ⚠️ Invalid linear power bins: {linear_power.shape[1]}, expected {expected_bins}")
                return None

            return linear_power

        except Exception as e:
            print(f"   ⚠️ Mel to linear conversion failed: {e}")
            return None

    @staticmethod
    def _griffin_lim(mag, n_iter=30):
        """Simple Griffin-Lim algorithm for phase reconstruction with mixed precision fix."""
        # Ensure mag is float32 to avoid mixed precision issues
        mag = tf.cast(mag, tf.float32)

        # Random initial phase (build complex tensor without Python 1j literals)
        theta = tf.random.uniform(tf.shape(mag), 0.0, 2 * np.pi, dtype=tf.float32)
        phase = tf.complex(tf.cos(theta), tf.sin(theta))
        S = tf.cast(mag, tf.complex64) * phase

        for _ in range(n_iter):
            # ISTFT
            wav = tf.signal.inverse_stft(
                S,
                frame_length=audio_cfg.win_length,
                frame_step=audio_cfg.hop_length,
                window_fn=tf.signal.hann_window
            )

            # STFT
            S = tf.signal.stft(
                wav,
                frame_length=audio_cfg.win_length,
                frame_step=audio_cfg.hop_length,
                window_fn=tf.signal.hann_window
            )

            # Update phase - compute complex unit phase from angles without 1j
            angles = tf.math.angle(S)
            phase = tf.complex(tf.cos(angles), tf.sin(angles))
            S = tf.cast(mag, tf.complex64) * phase

        # Final ISTFT
        wav = tf.signal.inverse_stft(
            S,
            frame_length=audio_cfg.win_length,
            frame_step=audio_cfg.hop_length,
            window_fn=tf.signal.hann_window
        )

        # Normalize and ensure float32 output
        wav = tf.cast(wav, tf.float32)
        wav = wav / (tf.reduce_max(tf.abs(wav)) + 1e-6)
        return tf.expand_dims(wav, 0)  # Add batch dimension for TensorBoard


class TrainingProgressLogger(tf.keras.callbacks.Callback):
    """
    Callback to log detailed training progress and metrics.

    Provides real-time feedback on training progress, including loss values,
    learning rate, and time estimates.
    """

    def __init__(self, log_interval=50):
        super().__init__()
        self.log_interval = log_interval
        self.start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        """Initialize training start time."""
        self.start_time = tf.timestamp()
        print("🏁 Training started...")

    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time."""
        self.epoch_start_time = tf.timestamp()
        print(f"\n📅 Epoch {epoch + 1}/{self.model._train_counter if hasattr(self.model, '_train_counter') else 'N'}")

    def on_train_batch_end(self, batch, logs=None):
        """Log progress at specified intervals."""
        if batch % self.log_interval == 0 and logs:
            current_time = tf.timestamp()
            elapsed = current_time - self.start_time

            # Format metrics
            loss = logs.get('loss', 0)
            l1_pre = logs.get('l1_pre', 0)
            l1_post = logs.get('l1_post', 0)
            stop_acc = logs.get('stop_acc', 0)
            within_2db = logs.get('within2db', 0)
            gal = logs.get('gal', 0)

            print(f"🔄 Step {batch:4d} | Loss: {loss:.4f} | L1_pre: {l1_pre:.4f} | L1_post: {l1_post:.4f} | "
                  f"Stop_acc: {stop_acc:.3f} | Within_2dB: {within_2db:.3f} | GAL: {gal:.4f} | "
                  f"Time: {elapsed/3600:.1f}h")

    def on_epoch_end(self, epoch, logs=None):
        """Log epoch completion with validation metrics."""
        if logs:
            train_loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            val_l1_pre = logs.get('val_l1_pre', 0)
            val_l1_post = logs.get('val_l1_post', 0)
            val_stop_acc = logs.get('val_stop_acc', 0)
            val_within_2db = logs.get('val_within2db', 0)

            epoch_time = tf.timestamp() - self.epoch_start_time
            print(f"✅ Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val L1_pre: {val_l1_pre:.4f} | Val L1_post: {val_l1_post:.4f} | "
                  f"Val Stop_acc: {val_stop_acc:.3f} | Val Within_2dB: {val_within_2db:.3f} | "
                  f"Duration: {epoch_time:.1f}s")


# Create callbacks
callbacks = create_callbacks(config)

# Fix mixed precision for sample generation
sample_generation_callback = SampleGenerationCallback(
    tokenizer=tokenizer,
    text_samples=["Hello world", "This is a test of the text to speech system"],
    generation_interval=200,  # Generate samples every 200 steps
    max_samples_to_keep=5
)
callbacks.append(sample_generation_callback)

# Add training progress logger
progress_logger = TrainingProgressLogger(log_interval=50)
callbacks.append(progress_logger)

# Enable TensorBoard audio logging with fixed mixed precision
tensorboard_audio_callback = TensorBoardAudioLogger(
    tokenizer=tokenizer,
    sample_texts=["Hello world", "This is a test"],
    log_interval=500,  # Log audio every 500 steps
    max_audio_samples=3
)
callbacks.append(tensorboard_audio_callback)

# Start training
print("🚀 Starting training...")
validation_data = val_generator if len(val_generator) > 0 else None
learner.fit(
    train_generator,
    validation_data=validation_data,
    epochs=config.epochs,
    callbacks=callbacks
)

print("✅ Training completed!")
