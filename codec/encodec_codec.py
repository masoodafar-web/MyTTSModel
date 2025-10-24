"""
Lightweight wrapper for Hugging Face Encodec 24kHz model (facebook/encodec_24khz).

This module provides optional integration for compressing/decompressing audio
using Encodec. It is torch/transformers-based and is entirely optional. If the
dependencies are not available, the wrapper will raise ImportError on use.

Usage:
    codec = Encodec24k(device="cpu")
    wav_24k = codec.reencode_to_24k(waveform_16k, input_sr=16000)  # -> np.ndarray [1, N, 1]

Notes:
    - This does NOT convert mel-spectrograms to audio. You should first obtain
      a waveform by any method (e.g., Griffin-Lim), then optionally re-encode
      with Encodec.
    - Designed for inference-time sample logging and export.
"""

from typing import Optional


class Encodec24k:
    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        try:
            from transformers import EncodecModel, AutoProcessor  # noqa: F401
            import torch  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Encodec dependencies not available. Please install transformers and torch."
            ) from e

        from transformers import EncodecModel, AutoProcessor
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model.eval()

    @property
    def info(self):
        cfg = self.model.config
        # Attribute names differ across versions; try common ones
        num_cq = (
            getattr(cfg, 'num_codebooks', None)
            or getattr(cfg, 'codebook_count', None)
            or getattr(cfg, 'rvq_groups', None)
            or getattr(cfg, 'nb_codebooks', None)
        )
        codebook_size = (
            getattr(cfg, 'codebook_size', None)
            or getattr(cfg, 'codebook_size_codebook', None)
            or getattr(cfg, 'codebook_bins', None)
        )
        return {
            'num_codebooks': int(num_cq) if num_cq is not None else None,
            'codebook_size': int(codebook_size) if codebook_size is not None else None,
        }

    def encode_path_to_codes(self, wav_path: str) -> "tuple[np.ndarray, dict]":
        """
        Encode a WAV file into Encodec RVQ code indices.

        Returns:
            (codes, meta) where codes has shape [T, K] (time, num_codebooks), dtype int32
            meta includes num_codebooks and codebook_size
        """
        import numpy as np
        import soundfile as sf

        y, sr = sf.read(wav_path)
        codes, cfg = self._encode_audio_array(y, sr)
        codes_np = codes.detach().cpu().numpy().astype(np.int32)
        return codes_np, cfg

    def _encode_audio_array(self, audio_array: "np.ndarray", sr: int) -> "tuple[torch.Tensor, dict]":
        """Shared encoding logic for audio arrays."""
        import numpy as np
        import librosa
        import torch

        y = audio_array
        if y.ndim == 2:
            y = y.mean(axis=1)
        if int(sr) != 24000:
            try:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=24000, res_type="soxr_hq")
            except Exception:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=24000, res_type="kaiser_best")

        inputs = self.processor(raw_audio=y, sampling_rate=24000, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        with torch.no_grad():
            enc = self.model.encode(input_values, padding_mask=padding_mask)
            codes = enc.audio_codes  # expected [B, K, T] or [B, T, K]
        codes = codes[0]
        cfg = self.info

        # Normalize shape to [T, K]
        if codes.ndim != 2:
            codes = codes.view(codes.shape[0], -1)
        T0, T1 = codes.shape[0], codes.shape[1]
        K = cfg['num_codebooks'] or min(T0, T1)
        if T0 == K:
            codes = codes.transpose(0, 1)  # [K, T] -> [T, K]
        elif T1 == K:
            pass  # already [T, K]
        else:
            # Heuristic: assume [K, T]
            if T0 < T1:
                codes = codes.transpose(0, 1)

        return codes, cfg

    def encode_path_to_codes_from_array(self, audio_array: "np.ndarray", sr: int) -> "tuple[np.ndarray, dict]":
        """
        Encode a numpy audio array into Encodec RVQ code indices.

        Args:
            audio_array: np.ndarray [T] mono float32 in [-1, 1]
            sr: sampling rate of audio_array

        Returns:
            (codes, meta) where codes has shape [T, K] (time, num_codebooks), dtype int32
            meta includes num_codebooks and codebook_size
        """
        codes, cfg = self._encode_audio_array(audio_array, sr)
        # Ensure codes are int32 for TensorFlow compatibility
        codes_np = codes.detach().cpu().numpy().astype(np.int32)
        return codes_np, cfg

    def decode_codes_to_audio(self, codes: "np.ndarray") -> "np.ndarray":
        """
        Decode Encodec RVQ codes back to waveform (24kHz).

        Args:
            codes: np.ndarray [T, K] or [B, T, K] (int)

        Returns:
            np.ndarray [B, N, 1] float32 in [-1,1]
        """
        import numpy as np
        import torch

        # Prefer inferring K from the provided codes to avoid relying on config
        codes_np = np.asarray(codes)
        # Normalize shape to [B, T, K]
        if codes_np.ndim == 1:
            # Ambiguous; treat as [T] with K=1
            codes_np = codes_np[:, None]
        if codes_np.ndim == 2:
            # [T, K] → [1, T, K]
            codes_np = codes_np[None, ...]
        elif codes_np.ndim != 3:
            raise ValueError(f"Expected codes with 2 or 3 dims, got shape {codes_np.shape}")

        # Ensure we have the correct shape [B, T, K]
        B, T, K = codes_np.shape

        if self.verbose:
            print(f"🔍 Debug decode: input shape {codes_np.shape}, B={B}, T={T}, K={K}")

        # to torch: [B, K, T]
        # Build dense tensor on CPU first to avoid any odd sparse pathways, then move to device
        codes_t = torch.from_numpy(codes_np.astype(np.int64, copy=False)).contiguous()
        if self.verbose:
            print(f"🔍 Debug decode: torch tensor shape {codes_t.shape}")

        # Explicitly convert to [B, K, T] format expected by Encodec
        # codes_t is currently [B, T, K], we need [B, K, T]
        codes_t = codes_t.permute(0, 2, 1).contiguous()

        if self.verbose:
            print(f"🔍 Debug decode: final torch shape {codes_t.shape}")
        codes_t = codes_t.to(self.device)
        scales = torch.ones((B, K), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            decoded = self.model.decode(codes_t, audio_scales=scales)
            audio = decoded.audio_values  # [B, channels, time]
        audio_np = audio.detach().cpu().numpy()
        if audio_np.ndim == 3:
            audio_np = audio_np[:, 0, :]  # mono
        maxv = np.max(np.abs(audio_np), axis=1, keepdims=True) + 1e-6
        audio_np = (audio_np / maxv).astype(np.float32)
        return audio_np[:, :, None]

    def reencode_to_24k(self, waveform: "np.ndarray", input_sr: int) -> "np.ndarray":
        """
        Resample a mono waveform to 24kHz and round-trip through Encodec encode+decode.

        Args:
            waveform: np.ndarray of shape [T] or [1, T] mono float32 in [-1, 1]
            input_sr: input sampling rate

        Returns:
            np.ndarray shaped [1, N, 1] suitable for tf.summary.audio
        """
        import numpy as np
        import torch
        import librosa

        wav = waveform
        if wav.ndim == 2:
            wav = wav[0]
        wav = np.asarray(wav, dtype=np.float32)

        # Resample to 24k
        if int(input_sr) != 24000:
            try:
                wav_24k = librosa.resample(wav, orig_sr=int(input_sr), target_sr=24000, res_type="soxr_hq")
            except Exception:
                # Fallback to basic resampling
                wav_24k = librosa.resample(wav, orig_sr=int(input_sr), target_sr=24000, res_type="linear")
        else:
            wav_24k = wav

        # HF processor expects shape [batch, channels, time]
        inputs = self.processor(
            raw_audio=wav_24k,
            sampling_rate=24000,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)  # [1, channels, time]
        padding_mask = inputs.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        with torch.no_grad():
            encoded = self.model.encode(input_values, padding_mask=padding_mask)
            codes = encoded.audio_codes
            decoded = self.model.decode(codes, audio_scales=encoded.audio_scales)
            audio_24k = decoded.audio_values.squeeze(0)  # [channels, time]

        audio_np = audio_24k.detach().cpu().numpy()
        if audio_np.ndim == 2:
            # Keep mono
            audio_np = audio_np[0]
        # Normalize to [-1,1] safely
        audio_np = audio_np.astype(np.float32)
        maxv = np.max(np.abs(audio_np)) + 1e-6
        audio_np = audio_np / maxv
        # [1, T, 1] for tf.summary.audio
        return audio_np[None, :, None]
