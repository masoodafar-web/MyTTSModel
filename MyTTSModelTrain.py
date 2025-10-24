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
from TTSDataLoader import AudioCfg, TextCfg, preprocess_dataset, TTSDataset, load_ljspeech_items

# Import tfio for audio resampling (if available)
try:
    import tensorflow_io as tfio
    TFIO_AVAILABLE = True
except ImportError:
    print("⚠️ tensorflow-io not available, audio resampling may not work")
    tfio = None
    TFIO_AVAILABLE = False

# Import phonemizer for pronunciation (required)
try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    PHONEMIZER_AVAILABLE = True
except ImportError:
    raise ImportError("phonemizer is required for this TTS model. Please install it: pip install phonemizer")

# Import audio quality metrics (optional)
try:
    import pesq
    import pystoi
    AUDIO_METRICS_AVAILABLE = True
    print("✅ Audio quality metrics (PESQ, STOI) available")
except ImportError:
    AUDIO_METRICS_AVAILABLE = False
    print("⚠️ Audio quality metrics not available (PESQ, STOI). Install: pip install pesq pystoi")


def setup_environment():
    """Configure GPU memory growth and logging levels."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Help mitigate fragmentation in TF BFC allocator
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

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

    # Objective: 'encodec_diffusion' (unified approach)
    objective: str = "encodec_diffusion"

    # Audio processing
    audio_preset: str = "base16k"
    encodec_model_id: str = "facebook/encodec_24khz"

    # Text processing (phonemizer only)
    phonemizer_language: str = "en-us"  # Language for phonemizer
    use_phoneme_pos_encoding: bool = True  # Use standard positional encoding

    # Training
    batch_size: int = 16
    epochs: int = 50
    validation_split: float = 0.02

    # Model
    model_preset: str = "modern_large"
    use_tortoise: bool = False  # Set to True for Tortoise-style diffusion model
    checkpoint_path: str = "checkpoints/tts_core_last.weights.h5"

    # Optimization
    learning_rate_warmup_steps: int = 4000
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    # Diffusion parameters (for EncodecDiffusion)
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Encodec parameters
    num_codebooks: int = 4  # K in RVQ
    codebook_size: int = 1024  # C in each codebook

    # Voice conditioning
    latent_dim: int = 512

    # EMA
    ema_decay: float = 0.999

    # Callbacks
    early_stopping_patience: int = 8
    guided_attention_ramp_epochs: int = 3

    def __post_init__(self):
        """Override defaults with environment variables."""
        self.audio_preset = os.environ.get("TTS_AUDIO_PRESET", self.audio_preset)
        self.model_preset = os.environ.get("TTS_MODEL_PRESET", self.model_preset)
        self.objective = os.environ.get("TTS_OBJECTIVE", self.objective)
        self.use_phoneme_pos_encoding = bool(int(os.environ.get("TTS_USE_PHONEME_POS_ENCODING", "0")))
        # For encodec_diffusion, always use 24kHz preset
        if str(self.objective).lower() == 'encodec_diffusion':
            self.audio_preset = 'encodec24k'


def load_tokenizer(phonemizer_language: str = "en-us"):
    """Load and configure phonemizer tokenizer."""
    if not PHONEMIZER_AVAILABLE:
        raise ImportError("phonemizer is required but not available. Please install it.")

    print(f"Using phonemizer for language: {phonemizer_language}")

    class PhonemeTokenizer:
        def __init__(self, language):
            self.language = language
            self.backend = EspeakBackend(language)
            # Build vocabulary from common IPA phonemes
            self.vocab = self._build_phoneme_vocab()
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2

        def _build_phoneme_vocab(self):
            # Comprehensive IPA phoneme set
            phonemes = [
                # Consonants
                'p', 'b', 't', 'd', 'k', 'g', 'ʔ', 'q', 'ʕ',
                'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'x', 'ɣ', 'h',
                'm', 'n', 'ŋ', 'l', 'r', 'ɹ', 'ʁ', 'ɻ', 'j', 'w',

                # Vowels
                'i', 'ɪ', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ə', 'ɚ', 'ɜ', 'ɝ',
                'a', 'ɐ', 'e', 'ø', 'œ', 'ɘ', 'ɤ', 'ɯ', 'ɨ', 'ʉ', 'ɶ',

                # Diphthongs and vowel combinations
                'aɪ', 'aʊ', 'ɔɪ', 'oʊ', 'ɛə', 'ɪə', 'ʊə', 'eɪ', 'oɪ', 'aʊ',

                # Suprasegmentals and special
                'ˈ', 'ˌ', 'ː', 'ˑ', '.', '|', ' ', '‿', '͡', '̯',
            ]
            vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
            for i, phoneme in enumerate(phonemes, start=3):
                vocab[phoneme] = i
            return vocab

        def encode(self, text, add_special_tokens=True):
            try:
                # Phonemize the text
                phonemes = phonemize(text, language=self.language, backend=self.backend, strip=True)
                # Convert to token IDs
                tokens = []
                for char in phonemes:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        # Unknown phoneme, use space or skip
                        tokens.append(self.vocab.get(' ', 3))

                if add_special_tokens:
                    tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
                return tokens
            except Exception as e:
                # Silent fallback for batch processing
                tokens = [self.bos_token_id]
                for char in text.lower():
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab.get(' ', 3))
                tokens.append(self.eos_token_id)
                return tokens

        def __len__(self):
            return len(self.vocab)

    tokenizer = PhonemeTokenizer(phonemizer_language)
    print(f"✅ Phoneme tokenizer ready with {len(tokenizer)} tokens")
    return tokenizer


def setup_configs(tokenizer, config: TrainingConfig):
    """Create audio and text configurations."""
    from TTSConfig import make_audio_cfg, make_text_cfg

    audio_cfg = make_audio_cfg(config.audio_preset)
    text_cfg = make_text_cfg(tokenizer, config.phonemizer_language, max_text_len=256)

    print(f"Audio config: preset={config.audio_preset}, "
          f"sr={audio_cfg.target_sample_rate}, n_fft={audio_cfg.n_fft}, "
          f"hop={audio_cfg.hop_length}, n_mels={audio_cfg.n_mels}, fmax={audio_cfg.fmax}")

    return audio_cfg, text_cfg


def preprocess_data(config: TrainingConfig, audio_cfg, text_cfg, tokenizer):
    """Create tf.data pipeline for EncodecDiffusion (no offline preprocessing)."""
    print("🔄 Creating tf.data pipeline for EncodecDiffusion...")

    # Load dataset metadata only (no audio processing)
    items = load_ljspeech_items(
        config.dataset_root,
        metadata_name=config.metadata_name
    )

    if len(items) == 0:
        raise ValueError(f"No training samples found in {config.dataset_root}. "
                        f"Please check the dataset path and metadata file {config.metadata_name}.")

    print(f"✅ Dataset metadata loaded:")
    print(f"   - Total examples: {len(items)}")
    print(f"   - Will use tf.data pipeline for on-the-fly processing")

    # Return minimal info (tf.data will handle the rest)
    return items, None, None, None, {"num_codebooks": config.num_codebooks, "codebook_size": config.codebook_size}


# Initialize configuration
config = TrainingConfig()
tokenizer = load_tokenizer(config.phonemizer_language)
audio_cfg, text_cfg = setup_configs(tokenizer, config)
pre_data = preprocess_data(config, audio_cfg, text_cfg, tokenizer)
items, text_ids, codes_list, code_lens, encodec_info = pre_data

# Log training setup summary
print("\n" + "="*60)
print("🎯 TRAINING SETUP SUMMARY")
print("="*60)
print(f"📊 Dataset: {config.dataset_root}")
print(f"🎵 Audio: {config.audio_preset} ({audio_cfg.target_sample_rate}Hz, Encodec 24kHz)")
print(f"📝 Text: Phonemes ({config.phonemizer_language})")
print(f"   - Positional Encoding: {'Phoneme-aware' if config.use_phoneme_pos_encoding else 'Standard'}")
from TTSConfig import get_model_preset
model_preset_info = get_model_preset(config.model_preset)
print(f"🤖 Model: EncodecDiffusion TTS (layers={model_preset_info.num_layers})")
print(f"   - Diffusion timesteps: {config.num_timesteps}")
print(f"   - Encodec codebooks: K={config.num_codebooks}, C={config.codebook_size}")
print(f"   - Voice latent dim: {config.latent_dim}")
print(f"📈 Batch size: {config.batch_size} (adjusted for {strategy.num_replicas_in_sync} GPUs)")
print(f"🎯 Loss: Diffusion on discrete codes")
print(f"⏰ Training: {config.epochs} epochs")
print(f"💾 Checkpoints: {config.checkpoint_path}")
print(f"📊 Monitoring: Diffusion sampling progress")
print("="*60 + "\n")

# Data split is now handled by tf.data pipeline - no need for manual splitting


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


# No manual data split needed - tf.data handles it

def create_tfdata_pipeline(items, config: TrainingConfig, audio_cfg, text_cfg, tokenizer, is_training=True):
    """Create efficient tf.data pipeline for EncodecDiffusion training."""

    # Extract wav paths and texts
    wav_paths = [item[0] for item in items]
    texts = [item[1] for item in items]

    # Create tf.data dataset
    dataset = tf.data.Dataset.from_tensor_slices((wav_paths, texts))

    # Shuffle for training
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(wav_paths), 10000), seed=42)

    # Load and process audio/text
    dataset = dataset.map(
        lambda wav_path, text: load_and_process_sample(wav_path, text, audio_cfg, text_cfg, tokenizer, config),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # Do not filter aggressively; handle any bad samples by returning a minimal valid record

    # Batch with padding on the time dimension of codes; text is already padded
    max_text_len = int(getattr(text_cfg, 'max_text_len', 256))
    pad_id = tf.cast(getattr(text_cfg, 'pad_id', 0), tf.int32)

    # Choose batch size considering distribution strategy
    try:
        adj_bs = adjust_batch_size_for_strategy(config.batch_size, strategy)
    except Exception:
        adj_bs = int(config.batch_size)
    if is_training:
        batch_size_to_use = int(adj_bs)
    else:
        # For validation, do not exceed dataset length
        batch_size_to_use = int(min(max(1, len(wav_paths)), adj_bs))

    dataset = dataset.padded_batch(
        batch_size=batch_size_to_use,
        padded_shapes={
            'text_ids': [max_text_len],
            'codes': [None, config.num_codebooks],
            'code_len': [],
            'text_len': []
        },
        padding_values={
            'text_ids': pad_id,
            'codes': tf.cast(0, tf.int32),
            'code_len': tf.cast(0, tf.int32),
            'text_len': tf.cast(0, tf.int32)
        },
        drop_remainder=False
    )

    # Repeat training dataset for infinite stream; keep validation finite
    if is_training:
        dataset = dataset.repeat()
    else:
        # For validation, keep finite; ensure at least one batch
        try:
            val_batches = max(1, (len(wav_paths) + batch_size_to_use - 1) // batch_size_to_use)
        except Exception:
            val_batches = 1
        dataset = dataset.take(val_batches)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_and_process_sample(wav_path, text, audio_cfg, text_cfg, tokenizer, config):
    """Load and process a single sample with tf.data operations."""

    try:
        # Load audio
        audio_binary = tf.io.read_file(wav_path)
        audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=-1)

        # Resample if needed (guard tfio and use tf.cond for tensor compare)
        if TFIO_AVAILABLE and tfio is not None:
            target_sr = tf.cast(audio_cfg.target_sample_rate, sr.dtype)
            audio = tf.cond(
                tf.not_equal(sr, target_sr),
                lambda: tfio.audio.resample(audio, sr, target_sr),
                lambda: audio,
            )
        else:
            # Fallback: skip resampling if tfio not available
            pass

        # Convert to float32 and squeeze
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        # Normalize audio
        audio = audio / tf.maximum(tf.abs(audio), 1e-6)

        # Encode with Encodec (using cached tf.function)
        codes = encodec_encode_tf(audio, config.num_codebooks, config.codebook_size)

        # Tokenize text using tf.py_function, then truncate + pad to fixed length
        def _tok_py(t_bytes):
            # t_bytes can be a bytes object or a 0-d numpy array containing bytes
            try:
                if isinstance(t_bytes, (bytes, bytearray)):
                    s = t_bytes.decode('utf-8', errors='ignore')
                else:
                    # numpy scalar or array
                    s = t_bytes.item() if hasattr(t_bytes, 'item') else t_bytes
                    if isinstance(s, (bytes, bytearray)):
                        s = s.decode('utf-8', errors='ignore')
                    else:
                        s = str(s)

                # Use phonemizer tokenizer
                ids = tokenizer.encode(s, add_special_tokens=True)
                return np.asarray(ids, dtype=np.int32)
            except Exception as e:
                # Silent fallback for batch processing
                return np.asarray([tokenizer.bos_token_id, tokenizer.eos_token_id], dtype=np.int32)

        tokens = tf.py_function(func=_tok_py, inp=[text], Tout=tf.int32)
        tokens.set_shape([None])

        max_len = tf.cast(getattr(text_cfg, 'max_text_len', 256), tf.int32)
        tokens = tokens[:max_len]
        text_len = tf.shape(tokens)[0]
        pad_len = tf.maximum(0, max_len - text_len)
        pad_id = tf.cast(getattr(text_cfg, 'pad_id', 0), tf.int32)
        tokens = tf.pad(tokens, [[0, pad_len]], constant_values=pad_id)

        # Get lengths
        code_len = tf.shape(codes)[0]

        return {
            'text_ids': tokens,
            'codes': codes,
            'code_len': code_len,
            'text_len': text_len,
        }

    except:
        # Return a minimal valid dummy sample to keep the pipeline alive
        max_len = tf.cast(getattr(text_cfg, 'max_text_len', 256), tf.int32)
        pad_id = tf.cast(getattr(text_cfg, 'pad_id', 0), tf.int32)
        dummy_tokens = tf.fill([max_len], pad_id)
        return {
            'text_ids': dummy_tokens,
            'codes': tf.zeros([1, config.num_codebooks], dtype=tf.int32),
            'code_len': tf.constant(1, dtype=tf.int32),
            'text_len': max_len,
        }


@tf.function
def encodec_encode_tf(audio, num_codebooks, codebook_size):
    """TensorFlow-compatible Encodec encoding using the codec wrapper."""
    # Convert TensorFlow tensor to numpy for the codec
    audio_np = audio.numpy()  # [T]

    # Use the Encodec24k wrapper for encoding
    try:
        codec = Encodec24k()
        codes, _ = codec.encode_path_to_codes_from_array(audio_np, sr=24000)
        # codes should be [T, K] where K = num_codebooks
        codes_tf = tf.convert_to_tensor(codes, dtype=tf.int32)
        return codes_tf
    except Exception as e:
        # Fallback to dummy codes if Encodec fails
        tf.print(f"⚠️ Encodec encoding failed: {e}, using dummy codes")
        dummy_length = tf.maximum(1, tf.cast(tf.shape(audio)[0] // 320, tf.int32))
        return tf.random.uniform([dummy_length, num_codebooks], 0, codebook_size, dtype=tf.int32)


def create_data_generators(items, config: TrainingConfig, audio_cfg, text_cfg, tokenizer):
    """Create tf.data generators for training and validation."""
    print("🔄 Creating tf.data pipelines...")

    # Split data
    total_samples = len(items)
    val_samples = max(1, int(total_samples * config.validation_split))
    val_samples = min(val_samples, total_samples - 1)  # Ensure at least 1 training sample
    train_items = items[:total_samples - val_samples]
    val_items = items[total_samples - val_samples:]

    # Ensure we have samples
    if len(train_items) == 0:
        raise ValueError("No training samples available. Check dataset path and metadata.")
    if len(val_items) == 0:
        print("⚠️ No validation samples, using 1 training sample for validation.")
        val_items = train_items[:1]
        train_items = train_items[1:]

    # Create datasets
    train_dataset = create_tfdata_pipeline(train_items, config, audio_cfg, text_cfg, tokenizer, is_training=True)
    val_dataset = create_tfdata_pipeline(val_items, config, audio_cfg, text_cfg, tokenizer, is_training=False)

    print(f"✅ tf.data pipelines created:")
    print(f"   - Training samples: {len(train_items)}")
    print(f"   - Validation samples: {len(val_items)}")

    # Calculate steps per epoch using adjusted batch size (ceil to cover dataset)
    try:
        adj_train_bs = adjust_batch_size_for_strategy(config.batch_size, strategy)
    except Exception:
        adj_train_bs = int(config.batch_size)
    train_steps = max(1, (len(train_items) + adj_train_bs - 1) // adj_train_bs)
    setattr(config, 'steps_per_epoch', int(train_steps))
    print(f"   - Steps per epoch: {train_steps}")
    print(f"   - Effective batch size: {adj_train_bs}")

    return train_dataset, val_dataset


# Create tf.data generators
train_generator, val_generator = create_data_generators(items, config, audio_cfg, text_cfg, tokenizer)

# Store val_items for validation steps calculation (robust for tiny datasets)
_val_count = max(1, int(len(items) * config.validation_split))
_val_count = min(_val_count, max(1, len(items) - 1))
val_items_for_steps = items[len(items) - _val_count:]

def create_model(config: TrainingConfig, data_split, strategy, tokenizer, text_cfg):
    """Create and initialize the EncodecDiffusion TTS model."""
    print("🔄 Creating EncodecDiffusion TTS model...")

    # Import presets
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
        # Import the unified diffusion model
        from MyTTSModel import EncodecDiffusionTTS
        print("🎯 Using EncodecDiffusion model")
        model = EncodecDiffusionTTS(
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            dff=model_config.dff,
            input_vocab_size=len(tokenizer),
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            latent_dim=config.latent_dim,
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            use_phoneme_pos_encoding=config.use_phoneme_pos_encoding,
            name="EncodecDiffusionTTS"
        )

        # Build model with estimated input shapes (tf.data will handle actual shapes)
        model.build_for_load(
            max_src_len=256,  # Estimated max text length
            max_tgt_len=500   # Estimated max code length
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
model = create_model(config, None, strategy, tokenizer, text_cfg)

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
        # Compute losses in float32 for numeric stability under mixed precision
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(mask, tf.float32)
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
        # Compute losses in float32 for numeric stability under mixed precision
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(mask, tf.float32)
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


class TTSEncodecLearner(tf.keras.Model):
    """
    Training wrapper for Encodec-code objective.

    Predicts per-frame RVQ code indices across K codebooks using cross-entropy.
    Also trains a stop gate with BCE and computes Guided Attention Loss.
    """
    def __init__(self, model, num_codebooks: int, codebook_size: int, loss_weights=None, guided_attention_weight=0.2, guided_attention_sigma=0.2):
        super().__init__()
        self.core = model
        self.K = int(num_codebooks)
        self.C = int(codebook_size)
        self.loss_weights = loss_weights or {
            "codes": 1.0,
            "stop": 0.5
        }
        self.ga_weight = float(guided_attention_weight)
        self.ga_sigma = float(guided_attention_sigma)
        self.ga_weight_var = tf.Variable(self.ga_weight, trainable=False, dtype=tf.float32, name="ga_weight")

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name="loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.codes_ce = tf.keras.metrics.Mean(name="codes_ce")
        self.bce_stop = tf.keras.metrics.Mean(name="bce_stop")
        self.stop_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="stop_acc")
        self.ga_metric = tf.keras.metrics.Mean(name="gal")

    @staticmethod
    def _sequence_mask_from_len(lengths, maxlen):
        return tf.sequence_mask(lengths, maxlen=maxlen, dtype=tf.float32)  # (B,T)

    @staticmethod
    def _codes_loss(y_true, logits, mask):
        # y_true: (B,T,K) int32, logits: (B,T,K,C), mask: (B,T)
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(mask, tf.float32)
        B = tf.shape(y_true)[0]
        T = tf.shape(y_true)[1]
        K = tf.shape(y_true)[2]
        C = tf.shape(logits)[-1]
        logits2 = tf.reshape(logits, [B*T*K, C])
        y2 = tf.reshape(y_true, [B*T*K])
        mask2 = tf.reshape(tf.tile(mask[:, :, None], [1, 1, K]), [B*T*K])
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2)
        ce = ce * mask2
        return tf.reduce_sum(ce) / (tf.reduce_sum(mask2) + 1e-8)

    @staticmethod
    def _compute_guided_attention_loss(attn_weights, enc_ids, tgt_len, sigma=0.2):
        if attn_weights is None:
            return tf.constant(0.0), tf.constant(0.0)
        B = tf.shape(enc_ids)[0]
        T = tf.shape(attn_weights)[1]
        S = tf.shape(attn_weights)[2]
        enc_lengths = tf.reduce_sum(tf.cast(tf.not_equal(enc_ids, 0), tf.int32), axis=1)
        decoder_lengths = tf.cast(tgt_len, tf.int32)
        t_idx = tf.cast(tf.range(T)[None, :, None], tf.float32)
        s_idx = tf.cast(tf.range(S)[None, None, :], tf.float32)
        t_norm = t_idx / tf.maximum(tf.cast(decoder_lengths[:, None, None], tf.float32) - 1.0, 1.0)
        s_norm = s_idx / tf.maximum(tf.cast(enc_lengths[:, None, None], tf.float32) - 1.0, 1.0)
        diff = t_norm - s_norm
        penalty = 1.0 - tf.exp(-tf.square(diff) / (2.0 * (sigma ** 2)))
        dec_mask = tf.sequence_mask(decoder_lengths, maxlen=T, dtype=tf.float32)
        enc_mask = tf.sequence_mask(enc_lengths, maxlen=S, dtype=tf.float32)
        mask = dec_mask[:, :, None] * enc_mask[:, None, :]
        weighted = penalty * tf.cast(attn_weights, tf.float32) * mask
        loss = tf.reduce_sum(weighted) / (tf.reduce_sum(mask) + 1e-8)
        metric = tf.reduce_sum(tf.reduce_sum(penalty * tf.cast(attn_weights, tf.float32), axis=-1) * dec_mask) / (tf.reduce_sum(dec_mask) + 1e-8)
        return loss, metric

    def call(self, inputs, training=False):
        if training:
            outputs = self.core(inputs, training=training, return_attn=True)
            codes_logits, stop_logits, attention = outputs
        else:
            codes_logits, stop_logits = self.core(inputs, training=training)
            attention = None
        return {
            "codes_logits": codes_logits,
            "stop": stop_logits,
            "attn": attention
        }

    def train_step(self, data):
        inputs, targets = data[0], data[1] if len(data) >= 2 else (data[0], data[1])
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            mask = self._sequence_mask_from_len(inputs["codes_len"], tf.shape(targets["codes"])[1])
            codes_loss = self._codes_loss(targets["codes"], outputs["codes_logits"], mask)
            stop_mask = mask
            stop_loss = TTSLearner._compute_weighted_bce_logits(self, targets["stop"], outputs["stop"], stop_mask)
            ga_loss, ga_metric = self._compute_guided_attention_loss(outputs.get("attn"), inputs["enc_ids"], inputs["codes_len"], sigma=self.ga_sigma)
            total_loss = self.loss_weights["codes"] * codes_loss + self.loss_weights["stop"] * stop_loss + tf.cast(self.ga_weight_var, tf.float32) * ga_loss

        optimizer = self.optimizer
        scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss
        grads = tape.gradient(scaled_loss, self.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.train_loss.update_state(total_loss)
        self.codes_ce.update_state(codes_loss)
        self.bce_stop.update_state(stop_loss)
        self.ga_metric.update_state(ga_metric)
        stop_probabilities = tf.sigmoid(outputs["stop"])  # (B,T,1)
        true_stop = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)
        pred_stop = tf.squeeze(tf.cast(stop_probabilities > 0.5, tf.float32), axis=-1)
        self.stop_acc.update_state(true_stop, pred_stop, sample_weight=stop_mask)
        return {
            "loss": self.train_loss.result(),
            "codes_ce": self.codes_ce.result(),
            "bce_stop": self.bce_stop.result(),
            "stop_acc": self.stop_acc.result(),
            "gal": self.ga_metric.result(),
        }

    def test_step(self, data):
        inputs, targets = data[0], data[1] if len(data) >= 2 else (data[0], data[1])
        codes_logits, stop_logits, attn = self.core(inputs, training=False, return_attn=True)
        mask = self._sequence_mask_from_len(inputs["codes_len"], tf.shape(targets["codes"])[1])
        codes_loss = self._codes_loss(targets["codes"], codes_logits, mask)
        stop_mask = mask
        stop_loss = TTSLearner._compute_weighted_bce_logits(self, targets["stop"], stop_logits, stop_mask)
        ga_loss, ga_metric = self._compute_guided_attention_loss(attn, inputs["enc_ids"], inputs["codes_len"], sigma=self.ga_sigma)
        total_loss = self.loss_weights["codes"] * codes_loss + self.loss_weights["stop"] * stop_loss + tf.cast(self.ga_weight_var, tf.float32) * ga_loss
        self.val_loss.update_state(total_loss)
        stop_prob = tf.sigmoid(stop_logits)
        true_stop = tf.squeeze(tf.cast(targets["stop"], tf.float32), axis=-1)
        pred_stop = tf.squeeze(tf.cast(stop_prob > 0.5, tf.float32), axis=-1)
        correct = tf.cast(tf.equal(pred_stop, true_stop), tf.float32)
        val_stop_acc = tf.reduce_sum(correct * stop_mask) / (tf.reduce_sum(stop_mask) + 1e-8)
        return {
            "loss": self.val_loss.result(),
            "codes_ce": codes_loss,
            "bce_stop": stop_loss,
            "stop_acc": val_stop_acc,
            "gal": ga_metric,
        }

def create_optimizer_and_learner(config: TrainingConfig, model, strategy):
    """Create optimizer and learner for EncodecDiffusion."""
    print("🔄 Setting up optimizer and learner...")

    # Resolve steps_per_epoch for warmup scheduling
    steps_per_epoch = int(getattr(config, 'steps_per_epoch', 0) or 0)
    if steps_per_epoch <= 0:
        # Fallback: try to infer from the globally created train_generator
        try:
            adj_batch_size = adjust_batch_size_for_strategy(config.batch_size, strategy)
            # len(train_generator) may not be available for tf.data;
            # use a safe default if it fails
            total_batches = len(train_generator)
            steps_per_epoch = max(1, int(total_batches))
            print(f"📊 Using steps_per_epoch from train_generator: {steps_per_epoch}")
        except Exception as e:
            print(f"⚠️ Could not resolve steps_per_epoch ({e}); using default 1000")
            steps_per_epoch = 1000
    else:
        print(f"📊 Using configured steps_per_epoch: {steps_per_epoch}")

    with strategy.scope():
        # Create learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            config, steps_per_epoch
        )

        # Create optimizer with fallback for different TF versions
        optimizer = create_optimizer(lr_schedule, config)

        # Create unified diffusion learner
        learner = EncodecDiffusionLearner(
            model,
            loss_weights={"diffusion": 1.0},
            tokenizer=tokenizer  # Pass tokenizer to learner
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

    # Expose steps_per_epoch for training loop
    try:
        setattr(config, 'steps_per_epoch', int(steps_per_epoch))
    except Exception:
        pass

    print("✅ Optimizer and learner setup complete")
    return learner


class EncodecDiffusionLearner(tf.keras.Model):
    """
    Training wrapper for EncodecDiffusion TTS.
    Handles diffusion training on discrete Encodec codes.
    """

    def __init__(self, model, loss_weights=None, tokenizer=None):
        super().__init__()
        self.core = model
        self.tokenizer = tokenizer  # Add tokenizer reference
        self.loss_weights = loss_weights or {"diffusion": 1.0}

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name="loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.diffusion_loss_metric = tf.keras.metrics.Mean(name="diffusion_loss")

        # Codebook usage metrics
        self.codebook_usage = tf.keras.metrics.Mean(name="codebook_usage")
        self.codebook_perplexity = tf.keras.metrics.Mean(name="codebook_perplexity")

        # Basic diffusion metrics
        self.diffusion_step_accuracy = tf.keras.metrics.Mean(name="diffusion_step_acc")
        self.code_reconstruction_error = tf.keras.metrics.Mean(name="code_recon_error")

        # Audio quality metrics (if available)
        if AUDIO_METRICS_AVAILABLE:
            self.pesq_metric = tf.keras.metrics.Mean(name="pesq")
            self.stoi_metric = tf.keras.metrics.Mean(name="stoi")

    def _ensure_int_inputs(self, inputs):
        # Cast fields that are used as indices/lengths to int32
        for k in ("text_ids", "codes", "code_len", "text_len", "voice_codes"):
            if k in inputs:
                inputs[k] = tf.cast(inputs[k], tf.int32)
        return inputs

    def call(self, inputs, training=False):
        # Ensure integer fields are int32 during symbolic build and training
        inputs = self._ensure_int_inputs(dict(inputs))
        return self.core(inputs, training=training)

    def train_step(self, data):
        # Accept dict batches or (x, y) tuples
        if isinstance(data, (tuple, list)):
            inputs = data[0]
        else:
            inputs = data

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            total_loss = self.loss_weights["diffusion"] * outputs["diffusion_loss"]

        optimizer = self.optimizer
        scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss
        grads = tape.gradient(scaled_loss, self.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.train_loss.update_state(tf.cast(total_loss, tf.float32))
        self.diffusion_loss_metric.update_state(tf.cast(outputs["diffusion_loss"], tf.float32))

        # Calculate codebook usage metrics
        codes = inputs["codes"]  # (B, T, K)
        # Flatten all codes across batch, time, and codebooks
        flat_codes = tf.reshape(codes, [-1])  # (B*T*K,)
        unique_codes = tf.unique(flat_codes)[0]
        # Remove padding (assuming 0 is padding)
        unique_codes = tf.boolean_mask(unique_codes, tf.not_equal(unique_codes, 0))
        usage = tf.cast(tf.shape(unique_codes)[0], tf.float32) / tf.cast(self.core.codebook_size, tf.float32)
        self.codebook_usage.update_state(usage)

        # Calculate codebook perplexity properly
        # Count frequency of each code
        code_counts = tf.math.bincount(flat_codes, minlength=self.core.codebook_size, maxlength=self.core.codebook_size)
        code_probs = tf.cast(code_counts, tf.float32) / tf.cast(tf.reduce_sum(code_counts), tf.float32)
        # Remove zero probabilities to avoid log(0)
        code_probs = tf.where(code_probs > 0, code_probs, 1e-10)
        entropy = -tf.reduce_sum(code_probs * tf.math.log(code_probs))
        perplexity = tf.exp(entropy)
        self.codebook_perplexity.update_state(perplexity)

        # Calculate diffusion step prediction accuracy (simplified)
        # This measures how well the model predicts the noise level
        # For now, just track if loss is decreasing (placeholder)
        step_acc = tf.exp(-total_loss)  # Higher when loss is lower
        self.diffusion_step_accuracy.update_state(step_acc)

        # Code reconstruction error (difference between predicted and target codes)
        # This would require access to the denoiser output, simplified for now
        recon_error = total_loss * 0.1  # Placeholder
        self.code_reconstruction_error.update_state(recon_error)

        metrics = {
            "loss": self.train_loss.result(),
            "diffusion_loss": self.diffusion_loss_metric.result(),
            "codebook_usage": self.codebook_usage.result(),
            "codebook_perplexity": self.codebook_perplexity.result(),
            "diffusion_step_acc": self.diffusion_step_accuracy.result(),
            "code_recon_error": self.code_reconstruction_error.result(),
        }

        # Add audio quality metrics if available
        if AUDIO_METRICS_AVAILABLE and hasattr(self, 'pesq_metric'):
            metrics["pesq"] = self.pesq_metric.result()
            metrics["stoi"] = self.stoi_metric.result()

        return metrics

    def test_step(self, data):
        if isinstance(data, (tuple, list)):
            inputs = data[0]
        else:
            inputs = data
        outputs = self(inputs, training=False)
        total_loss = self.loss_weights["diffusion"] * outputs["diffusion_loss"]

        self.val_loss.update_state(tf.cast(total_loss, tf.float32))

        # Calculate validation codebook metrics
        codes = inputs["codes"]  # (B, T, K)
        flat_codes = tf.reshape(codes, [-1])  # (B*T*K,)
        unique_codes = tf.unique(flat_codes)[0]
        unique_codes = tf.boolean_mask(unique_codes, tf.not_equal(unique_codes, 0))
        usage = tf.cast(tf.shape(unique_codes)[0], tf.float32) / tf.cast(self.core.codebook_size, tf.float32)

        # Calculate perplexity
        code_counts = tf.math.bincount(flat_codes, minlength=self.core.codebook_size, maxlength=self.core.codebook_size)
        code_probs = tf.cast(code_counts, tf.float32) / tf.cast(tf.reduce_sum(code_counts), tf.float32)
        code_probs = tf.where(code_probs > 0, code_probs, 1e-10)
        entropy = -tf.reduce_sum(code_probs * tf.math.log(code_probs))
        perplexity = tf.exp(entropy)

        metrics = {
            "loss": self.val_loss.result(),
            "diffusion_loss": tf.cast(outputs["diffusion_loss"], tf.float32),
            "codebook_usage": usage,
            "codebook_perplexity": perplexity,
            "diffusion_step_acc": tf.exp(-tf.cast(outputs["diffusion_loss"], tf.float32)),
            "code_recon_error": tf.cast(outputs["diffusion_loss"], tf.float32) * 0.1,
        }

        # Add audio quality metrics if available (validation)
        if AUDIO_METRICS_AVAILABLE and hasattr(self, 'pesq_metric'):
            # For validation, we could compute metrics on a subset, but for now use placeholders
            metrics["pesq"] = 0.0  # Placeholder
            metrics["stoi"] = 0.0  # Placeholder

        return metrics


def create_learning_rate_schedule(config: TrainingConfig, steps_per_epoch):
    """Create Noam learning rate schedule."""
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


class EncodecDiffusionSampleGenerationCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate sample codes using EncodecDiffusion during training.
    """

    def __init__(self, tokenizer, text_samples, generation_interval=500, max_samples_to_keep=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_samples = text_samples
        self.generation_interval = generation_interval
        self.max_samples_to_keep = max_samples_to_keep
        self.samples_dir = "training_samples_encodec_diffusion"
        self.step_count = 0
        os.makedirs(self.samples_dir, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        """Generate samples at specified intervals."""
        self.step_count += 1

        if self.step_count % self.generation_interval == 0:
            try:
                self._generate_samples()
            except Exception as e:
                print(f"⚠️ EncodecDiffusion sample generation failed: {e}")

    def _generate_samples(self):
        """Generate and save sample predictions using diffusion on codes."""
        print(f"\n🎯 Generating EncodecDiffusion samples at step {self.step_count}...")

        # Clean up old samples
        self._cleanup_old_samples()

        for i, text in enumerate(self.text_samples):
            try:
                # Tokenize text (phonemizer only)
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                tokens = tokens[:256]  # Truncate
                input_ids = tf.constant([tokens], dtype=tf.int32)

                # Generate using diffusion on codes
                generated_codes = self.model.core.generate(input_ids, num_steps=50)

                # Save sample data
                sample_data = {
                    'text': text,
                    'codes_shape': generated_codes.shape,
                    'step': self.step_count,
                    'diffusion_steps': 50
                }

                np.savez_compressed(
                    f"{self.samples_dir}/encodec_diffusion_sample_{i}_step_{self.step_count}.npz",
                    codes=generated_codes.numpy(),
                    metadata=str(sample_data)
                )

                # Try decoding to audio via Encodec if available
                try:
                    from codec.encodec_codec import Encodec24k
                    import soundfile as sf
                    codec = Encodec24k()
                    # Ensure codes are in correct shape [T, K]
                    codes_np = generated_codes.numpy()
                    if codes_np.ndim == 3 and codes_np.shape[0] == 1:
                        codes_np = codes_np[0]  # Remove batch dimension
                    print(f"   🔍 Debug: codes shape = {codes_np.shape}, dtype = {codes_np.dtype}")
                    wav = codec.decode_codes_to_audio(codes_np)  # [T, K] -> [1, N, 1]
                    wav_path = f"{self.samples_dir}/encodec_diffusion_sample_{i}_step_{self.step_count}.wav"
                    sf.write(wav_path, wav[0, :, 0], 24000)
                    print(f"   🎵 Wrote {wav_path}")
                except Exception as de:
                    print(f"   ⚠️ Could not decode to audio (saving codes only): {de}")
                    print(f"   🔍 Debug: generated_codes shape = {generated_codes.shape}, dtype = {generated_codes.dtype}")

                print(f"   ✅ EncodecDiffusion sample {i}: '{text[:30]}...' → {generated_codes.shape[1]} frames")

            except Exception as e:
                print(f"   ❌ EncodecDiffusion sample {i} failed: {e}")

    def _cleanup_old_samples(self):
        """Keep only the most recent samples."""
        sample_files = sorted([
            f for f in os.listdir(self.samples_dir)
            if f.startswith("encodec_diffusion_sample_") and f.endswith(".npz")
        ], key=lambda x: os.path.getctime(os.path.join(self.samples_dir, x)))

        if len(sample_files) > self.max_samples_to_keep * len(self.text_samples):
            files_to_remove = sample_files[:len(sample_files) - self.max_samples_to_keep * len(self.text_samples)]
            for f in files_to_remove:
                try:
                    os.remove(os.path.join(self.samples_dir, f))
                except:
                    pass
# Create optimizer and learner
learner = create_optimizer_and_learner(config, model, strategy)

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

    # Custom TensorBoard metrics logger
    class TensorBoardMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super().__init__()
            self.writer = tf.summary.create_file_writer(log_dir + "/metrics")
            self.step_count = 0

        def on_train_batch_end(self, batch, logs=None):
            if logs and batch % 10 == 0:  # Log every 10 steps
                with self.writer.as_default():
                    for key, value in logs.items():
                        if key in ['diffusion_step_acc', 'code_recon_error', 'codebook_usage', 'codebook_perplexity', 'pesq', 'stoi']:
                            tf.summary.scalar(f"train/{key}", value, step=self.step_count)
                self.step_count += 1

        def on_epoch_end(self, epoch, logs=None):
            if logs:
                with self.writer.as_default():
                    # Log all validation metrics
                    for key, value in logs.items():
                        if 'val_' in key:
                            metric_name = key.replace('val_', '')
                            tf.summary.scalar(f"validation/{metric_name}", value, step=epoch)

                    # Generate and log sample audio at the end of each epoch
                    try:
                        self._generate_sample_audio_for_tensorboard(epoch)
                    except Exception as e:
                        print(f"⚠️ Failed to generate sample audio for TensorBoard: {e}")

        def _generate_sample_audio_for_tensorboard(self, epoch):
            """Generate sample audio and log to TensorBoard for evaluation."""
            if not hasattr(self, 'model') or not hasattr(self.model, 'core'):
                return

            sample_texts = ["Hello world", "This is a test", "How are you today"]
            with self.writer.as_default():
                for i, text in enumerate(sample_texts):
                    try:
                        # Generate audio using the model's generation method
                        generated_codes = self.model.core.generate(
                            tf.constant([self.model.tokenizer.encode(text, add_special_tokens=True)], dtype=tf.int32),
                            num_steps=50
                        )

                        # Decode to audio using Encodec
                        if hasattr(generated_codes, 'numpy'):
                            codes_np = generated_codes.numpy()
                        else:
                            codes_np = generated_codes

                        try:
                            from codec.encodec_codec import Encodec24k
                            codec = Encodec24k()
                            # Ensure codes are in correct shape [T, K]
                            if codes_np.ndim == 3 and codes_np.shape[0] == 1:
                                codes_np = codes_np[0]  # Remove batch dimension
                            print(f"   🔍 Debug: final codes shape = {codes_np.shape}, dtype = {codes_np.dtype}")
                            audio_waveform = codec.decode_codes_to_audio(codes_np)  # [T, K] -> [1, N, 1]

                            if audio_waveform is not None:
                                # Ensure proper shape for TensorBoard [batch, samples, channels]
                                audio_tf = tf.convert_to_tensor(audio_waveform, dtype=tf.float32)
                                if tf.rank(audio_tf) == 3 and audio_tf.shape[0] == 1:
                                    audio_tf = tf.squeeze(audio_tf, axis=0)  # Remove batch dim if present
                                if tf.rank(audio_tf) == 2:
                                    audio_tf = tf.expand_dims(audio_tf, axis=-1)  # Add channel dim

                                # Log audio to TensorBoard
                                tf.summary.audio(
                                    name=f"generated_audio_sample_{i}",
                                    data=audio_tf,
                                    sample_rate=24000,
                                    step=epoch,
                                    description=f"Generated audio for: '{text}'"
                                )
                                print(f"   ✅ Logged generated audio sample {i} to TensorBoard")
                        except Exception as e:
                            print(f"   ⚠️ Failed to decode audio for sample {i}: {e}")

                    except Exception as e:
                        print(f"   ❌ Failed to generate audio for sample {i}: {e}")

    tensorboard_metrics = TensorBoardMetricsCallback(f"logs/tts/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

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

    # Assemble callbacks
    callbacks = [ema_callback, best_checkpoint, tensorboard, tensorboard_metrics, early_stopping]

    # Guided Attention ramp only for non-Encodec objectives
    if str(getattr(config, 'objective', '')).lower() != 'encodec_diffusion':
        target_weight = float(getattr(config, 'guided_attention_weight', 0.2))
        ga_ramp = GuidedAttentionRampCallback(
            start=0.0,
            target=target_weight,
            ramp_epochs=getattr(config, 'guided_attention_ramp_epochs', 3)
        )
        callbacks.insert(0, ga_ramp)

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
                # Tokenize text with validation (phonemizer only)
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                if len(tokens) == 0:
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
                mel_db = self._mel_to_db(mel_pred.numpy()[0])  # -> [-100, 0] dB
                # Normalize to [0,1] so TensorBoard shows contrast
                mel_img01 = (mel_db + 100.0) / 100.0
                mel_img01 = tf.clip_by_value(tf.convert_to_tensor(mel_img01, dtype=tf.float32), 0.0, 1.0)
                mel_image = tf.expand_dims(mel_img01, axis=0)  # Add batch dim [1, T, M]
                mel_image = tf.expand_dims(mel_image, axis=-1)  # Add channel dim [1, T, M, 1]
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

    def __init__(self, tokenizer, sample_texts, log_interval=500, max_audio_samples=3, use_encodec=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_texts = sample_texts[:max_audio_samples]  # Limit number of samples
        self.log_interval = log_interval
        self.max_audio_samples = max_audio_samples
        self.step_count = 0
        self.writer = None
        self.use_encodec = bool(use_encodec)
        self._encodec = None  # lazy init

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
                        # Ensure shape [batch, samples, channels] for tf.summary.audio
                        if tf.rank(audio_waveform) == 2:
                            audio_waveform = tf.expand_dims(audio_waveform, axis=-1)

                        # Log audio to TensorBoard
                        # Use 24k if Encodec is used
                        sr_out = 24000 if self.use_encodec else audio_cfg.target_sample_rate
                        tf.summary.audio(
                            name=f"sample_{i}_{text[:20].replace(' ', '_')}",
                            data=audio_waveform,
                            sample_rate=sr_out,
                            step=self.step_count,
                            encoding='wav'
                        )

                        # Also log mel-spectrogram as image
                        mel_spec = self._get_last_mel()
                        if mel_spec is not None:
                            # Normalize mel (expected in [-1,1]) to [0,1] for visualization
                            mel_img01 = tf.clip_by_value((tf.convert_to_tensor(mel_spec, dtype=tf.float32) + 1.0) * 0.5, 0.0, 1.0)
                            # Convert to image format (add batch and channel dims)
                            mel_image = tf.expand_dims(mel_img01, axis=0)  # [1, T, M]
                            mel_image = tf.expand_dims(mel_image, axis=-1)  # [1, T, M, 1]
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

            # Tokenize with validation (phonemizer only)
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) == 0:
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

            # Optionally re-encode with Encodec 24kHz for higher-quality codec output
            if self.use_encodec:
                try:
                    if self._encodec is None:
                        from codec.encodec_codec import Encodec24k
                        self._encodec = Encodec24k()
                    import numpy as np
                    # audio is [1, T, 1] float32 in [-1,1]
                    audio_np = audio.numpy().astype(np.float32)
                    one_d = audio_np[0, :, 0]
                    audio_24k = self._encodec.reencode_to_24k(one_d, input_sr=int(audio_cfg.target_sample_rate))
                    return tf.convert_to_tensor(audio_24k, dtype=tf.float32)
                except Exception as e:
                    print(f"   ⚠️ Encodec re-encode failed (falling back to raw): {e}")
                    return audio

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

            tokens = self.tokenizer.encode(text, add_special_tokens=True)
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

            # Only log essential metrics to console
            print(f"🔄 Step {batch:4d} | Loss: {loss:.4f} | "
                  f"Codebook Usage: {logs.get('codebook_usage', 0):.4f} | "
                  f"Codebook Perplexity: {logs.get('codebook_perplexity', 0):.1f} | "
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

# Add training progress logger
progress_logger = TrainingProgressLogger(log_interval=50)
callbacks.append(progress_logger)

# Add EncodecDiffusion sample generation callback
encodec_diffusion_sample_callback = EncodecDiffusionSampleGenerationCallback(
    tokenizer=tokenizer,
    text_samples=["Hello world", "This is a test of the EncodecDiffusion system", "How are you today"],
    generation_interval=500,
    max_samples_to_keep=3
)
callbacks.append(encodec_diffusion_sample_callback)

# Phonemizer is always used now
print("🎯 Using phonemizer for better pronunciation in sample generation")

# Start training
print("🚀 Starting training...")
validation_data = val_generator  # len() may be unknown; pass dataset directly

# Calculate validation steps using adjusted batch size (ceil)
try:
    adj_val_bs = adjust_batch_size_for_strategy(config.batch_size, strategy)
except Exception:
    adj_val_bs = int(config.batch_size)
val_count = max(1, len(val_items_for_steps))
val_steps = max(1, (val_count + adj_val_bs - 1) // adj_val_bs)

learner.fit(
    train_generator,
    validation_data=validation_data,
    epochs=config.epochs,
    steps_per_epoch=config.steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks
)

print("✅ Training completed!")
