# MyTTSModel - Transformer-based Text-to-Speech

A complete implementation of a Transformer-based Text-to-Speech (TTS) system built with TensorFlow/Keras. This model converts text to mel-spectrograms using an encoder-decoder architecture with attention mechanisms.

## 🎯 Features

- **Transformer Architecture**: Encoder-decoder with multi-head attention
- **Multilingual Support**: Uses NLLB tokenizer for text encoding
- **PostNet Refinement**: Convolutional post-processing for improved mel quality
- **Advanced Training**:
  - Mixed precision (float16) training
  - Multi-GPU support via MirroredStrategy
  - Exponential Moving Average (EMA) of weights
  - Guided Attention Loss (GAL) with ramp-up
  - Dynamic stop token weighting
- **Flexible Data Loading**: Both online (tf.data) and offline (NumPy) approaches
- **Efficient Inference**: Autoregressive generation with learnable go-frame
- **Comprehensive Documentation**: Full docstrings in Persian and English

## 📋 Requirements

```bash
pip install tensorflow>=2.12.0
pip install transformers
pip install librosa
pip install soundfile
pip install numpy
```

## 🏗️ Architecture

### Model Components

1. **Encoder**: Processes tokenized text with:
   - Token embeddings with positional encoding
   - 6-12 layers of self-attention + feed-forward networks
   - Pre-LayerNorm for training stability

2. **Decoder**: Generates mel-spectrograms with:
   - Mel prenet (bottleneck layers with dropout)
   - Causal self-attention + cross-attention to encoder
   - 6-12 layers with residual connections

3. **Prediction Heads**:
   - `mel_head`: Projects to mel-spectrogram dimensions (pre-PostNet)
   - `stop_head`: Predicts end-of-sequence tokens

4. **PostNet**: Refines mel predictions with:
   - 5 convolutional layers
   - Batch normalization + tanh activation
   - Residual connection to pre-PostNet output

### Key Hyperparameters

```python
NUM_LAYERS = 12        # Encoder/Decoder layers
D_MODEL = 512          # Model dimension
NUM_HEADS = 8          # Attention heads
DFF = 2024             # Feed-forward dimension
N_MELS = 80            # Mel-spectrogram bands
SAMPLE_RATE = 16000    # Audio sample rate
HOP_LENGTH = 256       # STFT hop length
```

## 📊 Data Preparation

### Dataset Structure

The model expects LJSpeech-style format:

```
dataset_train/
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata_train.csv  # Format: id|transcript|normalized_text
```

### Preprocessing

Two approaches are available:

#### 1. Offline Preprocessing (Recommended)

```python
from TTSDataLoader import preprocess_dataset, AudioCfg, TextCfg, TTSDataset
from transformers import AutoTokenizer

# Initialize tokenizer
tok = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", 
    use_fast=False, 
    src_lang="eng_Latn"  # or "pes_Arab" for Persian
)

# Configure audio and text
audio_cfg = AudioCfg(
    sample_rate=16000,
    target_sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    fmax=8000.0
)

text_cfg = TextCfg(
    pad_id=tok.pad_token_id,
    bos_id=tok.bos_token_id,
    eos_id=tok.eos_token_id,
    max_text_len=256,
    lang_code="eng_Latn"
)

# Preprocess entire dataset
items, text_ids, mels, mel_lens = preprocess_dataset(
    root_dir="./dataset_train",
    audio_cfg=audio_cfg,
    text_cfg=text_cfg,
    tok=tok,
    num_workers=8,  # Parallel processing
    cache_dir="checkpoints/mel_cache"  # Cache mel-spectrograms
)

# Create data generator
train_gen = TTSDataset(
    text_ids_list=text_ids,
    mels_list=mels,
    batch_size=4,
    pad_id=text_cfg.pad_id,
    n_mels=80,
    max_src_len=256,
    max_mel_len=2000,
    shuffle=True
)
```

#### 2. Online tf.data Pipeline

```python
from TTSDataLoader import build_dataset, load_ljspeech_items, PipelineCfg

items = load_ljspeech_items("./dataset_train")
pipeline_cfg = PipelineCfg(batch_size=4, shuffle_buffer=1000)

dataset = build_dataset(
    items=items,
    hf_tokenizer=tok,
    audio_cfg=audio_cfg,
    text_cfg=text_cfg,
    pipeline_cfg=pipeline_cfg
)
```

## 🚀 Training

### Basic Training Script

```python
from MyTTSModel import TransformerTTS
from MyTTSModelTrain import TTSLearner, CustomSchedule, EMAAndSaveCore, GAWeightRamp
import tensorflow as tf

# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Build model
    model_core = TransformerTTS(
        num_layers=12,
        d_model=512,
        num_heads=8,
        dff=2024,
        input_vocab_size=len(tok),
        n_mels=80,
        dropout_rate=0.1,
        pad_id=tok.pad_token_id,
        use_prenet=True,
        prenet_drop=0.5
    )
    
    # Create learner with custom training loop
    learner = TTSLearner(
        model_core,
        loss_weights={"mel_pre": 0.5, "mel_post": 1.0, "stop": 0.5},
        stop_pos_weight=None,  # Dynamic weighting
        ga_weight=0.2,         # Guided Attention Loss weight
        ga_sigma=0.2
    )
    
    # Optimizer with Noam learning rate schedule
    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(d_model=512, warmup_steps=8*steps_per_epoch),
        beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
    )
    
    learner.compile(optimizer=optimizer)

# Callbacks
callbacks = [
    GAWeightRamp(start=0.0, target=0.2, ramp_epochs=3),  # Ramp up GAL
    EMAAndSaveCore(decay=0.999),                          # EMA weights
    tf.keras.callbacks.ModelCheckpoint(                    # Best model
        "checkpoints/tts_learner_best.weights.h5",
        save_best_only=True, monitor="val_loss"
    ),
    tf.keras.callbacks.TensorBoard(log_dir="logs/tts"),   # TensorBoard
    tf.keras.callbacks.EarlyStopping(                      # Early stopping
        monitor="val_loss", patience=8, restore_best_weights=True
    )
]

# Train
learner.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)
```

### Training Metrics

The model tracks several metrics:
- **loss**: Total weighted loss
- **l1_pre/l1_post**: L1 loss for pre/post-PostNet mel (mean per mel-band)
- **bce_stop**: Binary cross-entropy for stop tokens
- **stop_acc**: Stop token prediction accuracy
- **within2db**: Percentage of mel bins within 2dB error
- **gal**: Guided Attention Loss penalty (alignment quality)

## 🎤 Inference

### Generate Speech from Text

```python
from MyTTSModel import TransformerTTS
import numpy as np
import librosa
import soundfile as sf

# Load model
model_core = TransformerTTS(
    num_layers=12,
    d_model=512,
    num_heads=8,
    dff=2024,
    input_vocab_size=len(tok),
    n_mels=80,
    pad_id=tok.pad_token_id
)

model_core.build_for_load(max_src_len=256, max_tgt_len=2000)
model_core.load_weights("checkpoints/tts_core_ema_last.weights.h5")

# Tokenize input text
text = "Hello, this is a test of the text to speech system."
enc_ids = tok.encode(text, add_special_tokens=True, src_lang="eng_Latn")
enc_ids = np.array([enc_ids], dtype=np.int32)  # Add batch dimension

# Generate mel-spectrogram
mel_post, stop_probs = model_core.greedy_generate_fast(
    enc_ids,
    max_steps=600,
    min_steps=40,
    stop_threshold=0.6,
    verbose=True
)

# Convert mel to audio (Griffin-Lim)
mel_np = mel_post[0].numpy()  # Remove batch dimension

# Denormalize mel from [-1, 1] to dB
mel_db = (mel_np + 1.0) / 2.0 * 100.0 - 100.0

# Convert dB to power
mel_power = librosa.db_to_power(mel_db)

# Mel to linear spectrogram (pseudo-inverse)
n_fft = 1024
n_mels = 80
mel_basis = librosa.filters.mel(
    sr=16000,
    n_fft=n_fft,
    n_mels=n_mels,
    fmin=0.0,
    fmax=8000.0
)
linear_spec = np.dot(np.linalg.pinv(mel_basis), mel_power.T)

# Griffin-Lim reconstruction
audio = librosa.griffinlim(
    linear_spec,
    n_iter=60,
    hop_length=256,
    win_length=1024
)

# Save audio
sf.write("output.wav", audio, 16000)
```

## 📁 Project Structure

```
MyTTSModel/
├── MyTTSModel.py           # Core model architecture
├── MyTTSModelTrain.py      # Training script with custom losses
├── TTSDataLoader.py        # Data loading and preprocessing
├── MyModelInfrence.ipynb   # Inference notebook
├── doc/
│   └── index.html         # Comprehensive architecture documentation
├── checkpoints/           # Model weights and cache
│   ├── mel_cache/        # Cached mel-spectrograms
│   ├── tts_core_last.weights.h5
│   └── tts_core_ema_last.weights.h5
└── logs/                  # TensorBoard logs
```

## 📖 Documentation

### Code Documentation

All Python modules have comprehensive docstrings:
- **MyTTSModel.py**: Model architecture components
- **MyTTSModelTrain.py**: Training utilities and custom losses
- **TTSDataLoader.py**: Data loading approaches

### Architecture Documentation

See `doc/index.html` for detailed architectural documentation with diagrams covering:
- Data flow and preprocessing pipeline
- Model architecture with component interactions
- Training flow with loss computations
- Inference process
- Configuration parameters

Open in browser:
```bash
open doc/index.html  # macOS
xdg-open doc/index.html  # Linux
start doc/index.html  # Windows
```

## 🔧 Advanced Configuration

### Multilingual Training (Persian Example)

```python
# Persian tokenizer
tok = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    use_fast=False,
    src_lang="pes_Arab"  # Persian language code
)

text_cfg = TextCfg(
    pad_id=tok.pad_token_id,
    bos_id=tok.bos_token_id,
    eos_id=tok.eos_token_id,
    max_text_len=256,
    lang_code="pes_Arab"
)
```

### Adjusting Model Size

For smaller/faster models:
```python
model_core = TransformerTTS(
    num_layers=6,      # Fewer layers
    d_model=256,       # Smaller dimension
    num_heads=4,       # Fewer heads
    dff=1024,          # Smaller FFN
    # ... other params
)
```

### Custom Loss Weights

```python
learner = TTSLearner(
    model_core,
    loss_weights={
        "mel_pre": 0.3,    # Pre-PostNet weight
        "mel_post": 1.0,   # Post-PostNet weight (primary)
        "stop": 0.5        # Stop token weight
    },
    ga_weight=0.15,        # Lower GAL for more flexibility
    ga_sigma=0.25          # Wider attention diagonal
)
```

## 🐛 Troubleshooting

### Out of Memory

1. Reduce batch size
2. Reduce max sequence lengths
3. Use gradient accumulation
4. Disable mixed precision

### Poor Alignment

1. Increase `ga_weight` (e.g., 0.3)
2. Decrease `ga_sigma` for stricter diagonal
3. Ensure proper data preprocessing
4. Check that text and audio are correctly paired

### Slow Training

1. Enable mixed precision
2. Use mel-spectrogram caching
3. Increase `num_workers` for data loading
4. Use multi-GPU training

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{myttsmodel2024,
  title={MyTTSModel: Transformer-based Text-to-Speech},
  author={masoodafar-web},
  year={2024},
  url={https://github.com/masoodafar-web/MyTTSModel}
}
```

## 📄 License

[Add your license information here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Tacotron 2](https://arxiv.org/abs/1712.05884)
- [Guided Attention Loss](https://arxiv.org/abs/1710.08969)
- [NLLB Tokenizer](https://huggingface.co/facebook/nllb-200-distilled-600M)

## 📧 Contact

For questions or issues, please open an issue on GitHub.
