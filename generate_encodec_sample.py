"""
Generate audio from trained EncodecDiffusion TTS.

Usage:
  python3 generate_encodec_sample.py \
    --text "Hello from the EncodecDiffusion TTS" \
    --ckpt checkpoints/tts_core_last_ema_last.weights.h5 \
    --preset normal \
    --lang eng_Latn \
    --steps 50 \
    --out out.wav

Requires:
  - TensorFlow 2.x
  - transformers + torch + soundfile + librosa (for Encodec decode)
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf


def setup_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_core_model(preset: str, vocab_size: int, num_codebooks: int, codebook_size: int,
                    steps: int, ckpt_path: str, pad_token_id: int = 0):
    from TTSConfig import get_model_preset
    from MyTTSModel import EncodecDiffusionTTS

    cfg = get_model_preset(preset)
    model = EncodecDiffusionTTS(
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        dff=cfg.dff,
        input_vocab_size=vocab_size,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        latent_dim=512,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        pad_token_id=pad_token_id,
    )
    # Build and load weights
    model.build_for_load(max_src_len=256, max_tgt_len=500)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        print(f"✅ Loaded weights: {ckpt_path}")
    else:
        print(f"⚠️ Checkpoint not found, using random weights: {ckpt_path}")
    return model


def tokenize(text: str, lang: str):
    """Tokenize text using PhonemeTokenizer to match training."""
    # Import the tokenizer from training script to ensure consistency
    from MyTTSModelTrain import load_tokenizer

    # Use phonemizer language mapping (same as training)
    phonemizer_lang = "en-us"  # Default for English
    if "eng" in lang.lower():
        phonemizer_lang = "en-us"
    elif "spa" in lang.lower():
        phonemizer_lang = "es"
    elif "fra" in lang.lower():
        phonemizer_lang = "fr-fr"
    # Add more language mappings as needed

    tok = load_tokenizer(phonemizer_lang)
    ids = tok.encode(text, add_special_tokens=True)
    return tok, np.array(ids, dtype=np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Input text")
    p.add_argument("--out", default="out.wav", help="Output WAV path (24kHz)")
    p.add_argument("--ckpt", default="checkpoints/tts_core_last_ema_last.weights.h5")
    p.add_argument("--preset", default="normal")
    p.add_argument("--lang", default="eng_Latn")
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    setup_tf()

    # Tokenize
    tok, ids = tokenize(args.text, args.lang)
    text_ids = tf.constant(ids[None, :], dtype=tf.int32)

    # Model hyperparams (must match training)
    num_codebooks = 4
    codebook_size = 1024

    # Load model
    model = load_core_model(
        args.preset,
        vocab_size=len(tok),
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        steps=args.steps,
        ckpt_path=args.ckpt,
        pad_token_id=getattr(tok, 'pad_token_id', 0)
    )

    # Generate codes
    print(f"🎯 Generating codes ({args.steps} steps)...")
    gen_codes = model.generate(text_ids, voice_codes=None, num_steps=args.steps)
    codes_np = gen_codes.numpy()[0]  # [T, K]
    print(f"✅ Codes: shape={codes_np.shape}, dtype={codes_np.dtype}")

    # Decode to audio via Encodec
    try:
        from codec.encodec_codec import Encodec24k
        import soundfile as sf
        codec = Encodec24k()
        audio = codec.decode_codes_to_audio(codes_np)  # [1, N, 1]
        wav = audio[0, :, 0]
        sf.write(args.out, wav, 24000)
        print(f"✅ Wrote {args.out} (24kHz)")
    except Exception as e:
        print(f"⚠️ Encodec decode unavailable: {e}")
        np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", codes=codes_np)
        print(f"💾 Saved codes to {os.path.splitext(args.out)[0] + '.npz'}")


if __name__ == "__main__":
    sys.exit(main())
