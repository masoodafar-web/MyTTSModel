"""
Centralized configuration for MyTTS model, training, and inference.

Provides:
- Model presets: tiny, normal, large
- Greedy decoding defaults with adaptive helpers

Usage:
    from TTSConfig import get_model_preset, GreedyDefaults, compute_greedy_params
    mcfg = get_model_preset("normal")
    params = compute_greedy_params(token_len, GreedyDefaults())
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ModelConfig:
    num_layers: int
    d_model: int
    num_heads: int
    dff: int
    dropout_rate: float = 0.1
    droppath_rate: float = 0.05
    use_prenet: bool = True
    prenet_drop: float = 0.5
    cross_win: Optional[float] = 0.2
    max_length: int = 4096


# Presets tuned for this repo; "normal" matches training defaults
PRESETS: Dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        num_layers=4, d_model=256, num_heads=4, dff=1024,
        dropout_rate=0.1, droppath_rate=0.05, use_prenet=True, prenet_drop=0.5, cross_win=0.2,
    ),
    "normal": ModelConfig(
        num_layers=8, d_model=512, num_heads=8, dff=2048,
        dropout_rate=0.1, droppath_rate=0.05, use_prenet=True, prenet_drop=0.5, cross_win=0.2,
    ),
    "large": ModelConfig(
        num_layers=12, d_model=768, num_heads=12, dff=3072,
        dropout_rate=0.1, droppath_rate=0.1, use_prenet=True, prenet_drop=0.5, cross_win=0.2,
    ),
}


def get_model_preset(name: str) -> ModelConfig:
    key = (name or "normal").lower()
    return PRESETS.get(key, PRESETS["normal"])


@dataclass
class GreedyDefaults:
    # Hard caps
    max_steps: int = 1200
    min_frames: int = 64
    # Adaptive heuristic
    frames_per_token: float = 6.0
    # Stop gate behavior
    stop_threshold: float = 0.9
    window: int = 6
    patience: int = 2
    check_stop_every: int = 5
    # Post-processing
    use_postnet: bool = False
    verbose: bool = True


def compute_greedy_params(token_len: int, d: GreedyDefaults) -> Dict[str, object]:
    """Derive stable greedy-generate parameters from text length.

    Args:
        token_len: Number of tokens in input text
        d: GreedyDefaults

    Returns: dict suitable for TransformerTTS.greedy_generate_fast(**params)
    """
    L = int(max(0, token_len or 0))
    min_steps = max(int(d.min_frames), int(round(d.frames_per_token * max(1, L))))
    # Keep within sane limits
    min_steps = min(min_steps, int(d.max_steps * 0.9))
    return {
        "max_steps": int(d.max_steps),
        "min_steps": int(min_steps),
        "stop_threshold": float(d.stop_threshold),
        "window": int(d.window),
        "patience": int(d.patience),
        "check_stop_every": int(d.check_stop_every),
        "use_postnet": bool(d.use_postnet),
        "verbose": bool(d.verbose),
    }


# ---------------- Audio / Text configs ----------------
@dataclass
class AudioPreset:
    sample_rate: int = 16000
    target_sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    trim_silence: bool = False
    trim_threshold_db: float = 60.0


AUDIO_PRESETS: Dict[str, AudioPreset] = {
    "base16k": AudioPreset(),
    "ljspeech22k": AudioPreset(sample_rate=22050, target_sample_rate=22050, fmax=8000.0),
}


def get_audio_preset(name: str = "base16k") -> AudioPreset:
    key = (name or "base16k").lower()
    return AUDIO_PRESETS.get(key, AUDIO_PRESETS["base16k"]) 


def make_audio_cfg(name: str = "base16k"):
    """Return an instance of TTSDataLoader.AudioCfg from preset name."""
    from TTSDataLoader import AudioCfg as _AudioCfg  # local import to avoid circular deps
    ap = get_audio_preset(name)
    return _AudioCfg(
        sample_rate=ap.sample_rate,
        target_sample_rate=ap.target_sample_rate,
        n_fft=ap.n_fft,
        hop_length=ap.hop_length,
        win_length=ap.win_length,
        n_mels=ap.n_mels,
        fmin=ap.fmin,
        fmax=ap.fmax,
        trim_silence=ap.trim_silence,
        trim_threshold_db=ap.trim_threshold_db,
    )


def make_text_cfg(tokenizer, lang_code: str = "eng_Latn", max_text_len: int = 256):
    """Construct TTSDataLoader.TextCfg using pad/bos/eos from tokenizer."""
    from TTSDataLoader import TextCfg as _TextCfg
    return _TextCfg(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        max_text_len=int(max_text_len),
        lang_code=str(lang_code),
    )
