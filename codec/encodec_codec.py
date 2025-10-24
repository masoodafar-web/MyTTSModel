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
    def __init__(self, device: Optional[str] = None):
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
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model.eval()

    @property
    def info(self):
        cfg = self.model.config
        # Attribute names differ across versions; try common ones
        num_cq = getattr(cfg, 'num_codebooks', None) or getattr(cfg, 'codebook_count', None)
        codebook_size = getattr(cfg, 'codebook_size', None) or getattr(cfg, 'codebook_size_codebook', None)
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
        import librosa
        import torch

        y, sr = sf.read(wav_path)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if int(sr) != 24000:
            try:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=24000, res_type="soxr_hq")
            except Exception:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=24000, res_type="kaiser_best")

        inputs = self.processor(audio=y, sampling_rate=24000, return_tensors="pt")
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

        cfg = self.info
        K = int(cfg['num_codebooks'])
        codes_np = np.asarray(codes)
        if codes_np.ndim == 2:
            codes_np = codes_np[None, ...]  # [B=1, T, K]
        B, T, K_in = codes_np.shape
        assert K_in == K, f"codes K={K_in} must match model K={K}"

        # to torch: [B, K, T]
        codes_t = torch.from_numpy(codes_np.astype(np.int64)).to(self.device)
        codes_t = codes_t.permute(0, 2, 1).contiguous()
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
            wav_24k = librosa.resample(wav, orig_sr=int(input_sr), target_sr=24000, res_type="soxr_hq")
        else:
            wav_24k = wav

        # HF processor expects shape [batch, channels, time]
        inputs = self.processor(
            audio=wav_24k,
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
