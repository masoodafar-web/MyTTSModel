# TTSDataLoader.py
import os, math, random, hashlib
import concurrent.futures
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# برای مسیر آفلاین:
import librosa
import soundfile as sf

# =========================================================
# Config
# =========================================================
@dataclass
class AudioCfg:
    sample_rate: int = 22050
    target_sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = 8000.0
    trim_silence: bool = False
    trim_threshold_db: float = 60.0  # هر چه کمتر، حساس‌تر به سکوت

@dataclass
class TextCfg:
    pad_id: int = 1   # NLLB pad=1
    bos_id: int = 0   # NLLB bos=0
    eos_id: int = 2   # NLLB eos=2
    max_text_len: int = 400
    lang_code: str = "eng_Latn"  # برای فارسی: "pes_Arab"

@dataclass
class PipelineCfg:
    batch_size: int = 16
    shuffle_buffer: int = 1000
    num_workers: int = tf.data.AUTOTUNE
    bucket_boundaries: Tuple[int, ...] = tuple(range(200, 2000, 200))  # بر اساس mel_len
    bucket_batch_sizes: Optional[Tuple[int, ...]] = None
    drop_remainder: bool = True

# =========================================================
# --- مسیر tf.data (اختیاری) ---
#   شامل کشِ mel/window برای رفع گلوگاه و سازگاری با Graph
# =========================================================
# Utils: dB, Trim, Resample (TF)
def _power_to_db(S, amin=1e-10):
    return 10.0 * tf.math.log(tf.maximum(S, amin)) / tf.math.log(tf.constant(10.0, S.dtype))

def _resample_1d_linear(x, sr_in, sr_out):
    T = tf.shape(x)[0]
    new_len = tf.cast(
        tf.round(tf.cast(T, tf.float32) * tf.cast(sr_out, tf.float32) / tf.cast(sr_in, tf.float32)),
        tf.int32
    )
    x2 = tf.reshape(x, [1, T, 1])                              # [N=1, T, C=1]
    x2 = tf.image.resize(x2, size=[new_len, 1], method="bilinear", antialias=True)
    return tf.reshape(x2, [new_len])

def _maybe_trim(audio, sr, cfg: AudioCfg):
    if not cfg.trim_silence:
        return audio
    energy = tf.math.square(audio)
    frames = tf.signal.frame(energy, 2048, 512, pad_end=True)  # (num_frames, 2048)
    frame_energy = tf.reduce_mean(frames, axis=-1) + 1e-10
    energy_db = _power_to_db(frame_energy)
    mask = energy_db > -cfg.trim_threshold_db
    idx = tf.where(mask)
    def _no_trim(): return audio
    def _do_trim():
        start = tf.reduce_min(idx)
        end   = tf.reduce_max(idx)
        start_samp = start * 512
        end_samp   = (end + 1) * 512
        return audio[start_samp:end_samp]
    return tf.cond(tf.size(idx) > 0, _do_trim, _no_trim)

# کش ثابت‌ها برای مسیر TF
_MEL_CACHE = {}
_WIN_CACHE = {}
def _get_mel_and_window(cfg: AudioCfg):
    mel_key = (cfg.target_sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    mel_mat = _MEL_CACHE.get(mel_key)
    if mel_mat is None:
        sr_const = float(cfg.target_sample_rate)
        lower    = float(cfg.fmin)
        upper    = float(cfg.fmax) if cfg.fmax is not None else (sr_const / 2.0)
        mel_mat = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=cfg.n_mels,
            num_spectrogram_bins=cfg.n_fft // 2 + 1,
            sample_rate=sr_const,
            lower_edge_hertz=lower,
            upper_edge_hertz=upper,
            dtype=tf.float32,
        )
        _MEL_CACHE[mel_key] = mel_mat

    win_key = (cfg.win_length,)
    hann_win = _WIN_CACHE.get(win_key)
    if hann_win is None:
        hann_win = tf.signal.hann_window(cfg.win_length, dtype=tf.float32)
        _WIN_CACHE[win_key] = hann_win

    return mel_mat, hann_win

def wav_to_logmel_tf(wav_path: tf.Tensor, cfg: AudioCfg) -> tf.Tensor:
    mel_mat, hann_win = _get_mel_and_window(cfg)

    audio_bytes = tf.io.read_file(wav_path)
    wav, sr = tf.audio.decode_wav(audio_bytes)                 # (T, C)
    wav = tf.reduce_mean(wav, axis=-1)                         # mono (T,)

    sr     = tf.cast(sr, tf.int32)
    tgt_sr = tf.cast(cfg.target_sample_rate, tf.int32)
    wav = tf.cond(tf.not_equal(sr, tgt_sr),
                  lambda: _resample_1d_linear(wav, sr, tgt_sr),
                  lambda: wav)
    wav = tf.cast(wav, tf.float32)

    wav = _maybe_trim(wav, sr=tgt_sr, cfg=cfg)

    # STFT دستی با پنجره کش‌شده
    frames = tf.signal.frame(wav, cfg.win_length, cfg.hop_length, pad_end=True)  # (F, win)
    frames = tf.cast(frames, tf.float32) * hann_win[tf.newaxis, :]
    spec = tf.signal.rfft(frames, [cfg.n_fft])                                   # (F, n_fft/2+1)
    power_spec = tf.math.real(spec * tf.math.conj(spec))                         # |FFT|^2

    mel = tf.matmul(power_spec, mel_mat)                                         # (F, n_mels)

    logmel = 10.0 * tf.math.log(tf.maximum(mel, 1e-10)) / tf.math.log(tf.constant(10.0, tf.float32))
    logmel = tf.clip_by_value(logmel, -100.0, 0.0)
    logmel = (logmel + 100.0) / 100.0
    logmel = (logmel * 2.0) - 1.0
    return logmel

# ====== Text (HF NLLB wrapper) برای مسیر tf.data ======
class HFTokenizerWrapper:
    def __init__(self, hf_tokenizer: AutoTokenizer, text_cfg: TextCfg):
        self.tok = hf_tokenizer
        # Avoid deprecated attributes (lang_code_to_id / fairseq_tokens_to_ids)
        # by passing src_lang at encode-time instead of setting tok.src_lang.
        self.src_lang = text_cfg.lang_code
        self.pad_id = text_cfg.pad_id
        self.eos_id = text_cfg.eos_id
        self.max_len = text_cfg.max_text_len

    def encode_ids(self, text: str) -> List[int]:
        # Pass src_lang explicitly to avoid deprecated internals
        ids = self.tok.encode(text, add_special_tokens=True, src_lang=self.src_lang)
        if self.max_len is not None:
            ids = ids[: self.max_len]
        return ids

def default_text_normalize(txt: tf.Tensor) -> tf.Tensor:
    txt = tf.strings.regex_replace(txt, r"\s+", " ")
    return tf.strings.strip(txt)

def encode_text_dynamic(txt: tf.Tensor, py_tokenize_ids: Callable[[str], List[int]]) -> tf.Tensor:
    ids = tf.py_function(
        func=lambda s: tf.constant(py_tokenize_ids(s.numpy().decode("utf-8")), dtype=tf.int32),
        inp=[txt],
        Tout=tf.int32
    )
    return tf.reshape(ids, [-1])

# ====== Parsing & Dataset (tf.data) ======
def make_example(wav_path: tf.Tensor, text: tf.Tensor, audio_cfg: AudioCfg, tok_wrap: HFTokenizerWrapper):
    text = default_text_normalize(text)
    text_ids = encode_text_dynamic(text, tok_wrap.encode_ids)      # (L,)
    mel = wav_to_logmel_tf(wav_path, audio_cfg)                    # (T, n_mels)
    return {
        "text_ids": text_ids,
        "text_len": tf.shape(text_ids)[0],
        "mel": mel,
        "mel_len": tf.shape(mel)[0],
        "wav_path": wav_path,
    }

def pad_batch(batch, text_pad_id: int):
    text = batch["text_ids"]
    mel  = batch["mel"]
    if isinstance(text, tf.RaggedTensor):
        text = text.to_tensor(default_value=tf.cast(text_pad_id, tf.int32))
    if isinstance(mel, tf.RaggedTensor):
        mel = mel.to_tensor(default_value=tf.cast(0.0, tf.float32))
    return {
        "text_ids": text,
        "text_len": batch["text_len"],
        "mel": mel,
        "mel_len": batch["mel_len"],
        "wav_path": batch["wav_path"],
    }

def make_bucketer(cfg: PipelineCfg):
    boundaries = list(cfg.bucket_boundaries)
    if not boundaries:
        return None, None
    batch_sizes = list(cfg.bucket_batch_sizes) if cfg.bucket_batch_sizes is not None else [cfg.batch_size] * (len(boundaries) + 1)
    return boundaries, batch_sizes

def build_dataset(
    items: List[Tuple[str, str]],
    hf_tokenizer: AutoTokenizer,
    audio_cfg: AudioCfg = AudioCfg(),
    text_cfg: TextCfg = TextCfg(),
    pipeline_cfg: PipelineCfg = PipelineCfg(),
):
    """
    items: لیست (wav_path, text)  ← مسیر TF (در صورت نیاز)
    """
    tok_wrap = HFTokenizerWrapper(hf_tokenizer, text_cfg)

    wav_paths = tf.constant([p for p, _ in items])
    texts = tf.constant([t for _, t in items])

    ds = tf.data.Dataset.from_tensor_slices((wav_paths, texts))
    # بهتر: map → cache → shuffle
    ds = ds.map(
        lambda p, t: make_example(p, t, audio_cfg, tok_wrap),
        num_parallel_calls=pipeline_cfg.num_workers,
        deterministic=False,
    )
    ds = ds.cache()  # می‌تونی مسیر دیسک یا RAM tmpfs بدی
    ds = ds.shuffle(pipeline_cfg.shuffle_buffer, reshuffle_each_iteration=True)

    boundaries, batch_sizes = make_bucketer(pipeline_cfg)
    if boundaries is not None:
        def _element_len(elem):
            return elem["mel_len"]
        ds = ds.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=_element_len,
                bucket_boundaries=boundaries,
                bucket_batch_sizes=batch_sizes,
                pad_to_bucket_boundary=False,
                drop_remainder=pipeline_cfg.drop_remainder,
            )
        )
    else:
        ds = ds.padded_batch(
            pipeline_cfg.batch_size,
            padded_shapes={
                "text_ids": [None],
                "text_len": [],
                "mel": [None, audio_cfg.n_mels],
                "mel_len": [],
                "wav_path": [],
            },
            padding_values={
                "text_ids": tf.cast(text_cfg.pad_id, tf.int32),
                "text_len": tf.cast(0, tf.int32),
                "mel": tf.cast(0.0, tf.float32),
                "mel_len": tf.cast(0, tf.int32),
                "wav_path": tf.constant("", dtype=tf.string),
            },
            drop_remainder=pipeline_cfg.drop_remainder,
        )

    ds = ds.map(lambda b: pad_batch(b, text_pad_id=text_cfg.pad_id),
                num_parallel_calls=pipeline_cfg.num_workers)

    # (اختیاری) کپی روی GPU:
    try:
        ds = ds.apply(tf.data.experimental.copy_to_device('/GPU:0')).prefetch(1)
    except Exception:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# --- نسخهٔ سریع بدون py_function: ورودی‌ها از قبل توکنایز شده‌اند ---
def make_example_fast(wav_path: tf.Tensor, text_ids: tf.RaggedTensor, audio_cfg: AudioCfg):
    mel = wav_to_logmel_tf(wav_path, audio_cfg)  # (T, n_mels)
    return {
        "text_ids": text_ids,                 # Ragged (L,)
        "text_len": tf.shape(text_ids)[0],
        "mel": mel,
        "mel_len": tf.shape(mel)[0],
        "wav_path": wav_path,
    }

def build_dataset_pre_tokenized(
    wav_paths: tf.Tensor,           # tf.constant([...], dtype=tf.string)
    text_ids_ragged: tf.RaggedTensor,  # tf.ragged.constant(list_of_id_lists, tf.int32)
    audio_cfg: AudioCfg,
    text_cfg: TextCfg,
    pipeline_cfg: PipelineCfg,
):
    ds = tf.data.Dataset.from_tensor_slices((wav_paths, text_ids_ragged))
    ds = ds.map(lambda p, ids: make_example_fast(p, ids, audio_cfg),
                num_parallel_calls=pipeline_cfg.num_workers,
                deterministic=False)
    ds = ds.cache()
    ds = ds.shuffle(pipeline_cfg.shuffle_buffer, reshuffle_each_iteration=True)

    boundaries = list(pipeline_cfg.bucket_boundaries)
    if boundaries:
        batch_sizes = list(pipeline_cfg.bucket_batch_sizes) if pipeline_cfg.bucket_batch_sizes is not None \
                      else [pipeline_cfg.batch_size] * (len(boundaries) + 1)
        ds = ds.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda e: e["mel_len"],
            bucket_boundaries=boundaries,
            bucket_batch_sizes=batch_sizes,
            pad_to_bucket_boundary=False,
            drop_remainder=pipeline_cfg.drop_remainder,
        ))
    else:
        ds = ds.padded_batch(
            pipeline_cfg.batch_size,
            padded_shapes={
                "text_ids": [None],
                "text_len": [],
                "mel": [None, audio_cfg.n_mels],
                "mel_len": [],
                "wav_path": [],
            },
            padding_values={
                "text_ids": tf.cast(text_cfg.pad_id, tf.int32),
                "text_len": tf.cast(0, tf.int32),
                "mel": tf.cast(0.0, tf.float32),
                "mel_len": tf.cast(0, tf.int32),
                "wav_path": tf.constant("", dtype=tf.string),
            },
            drop_remainder=pipeline_cfg.drop_remainder,
        )

    ds = ds.map(lambda b: pad_batch(b, text_pad_id=text_cfg.pad_id),
                num_parallel_calls=pipeline_cfg.num_workers)

    # بهینه‌سازی‌ها
    opts = tf.data.Options()
    opts.deterministic = False
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_threading.max_intra_op_parallelism = 0
    opts.experimental_threading.private_threadpool_size = 0
    ds = ds.with_options(opts)

    try:
        ds = ds.apply(tf.data.experimental.copy_to_device('/GPU:0')).prefetch(1)
    except Exception:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# =========================================================
# --- مسیر آفلاین / بدون tf.data ---
#   محاسبه‌ی کامل دیتاست (wav→logmel + tokenize) در RAM
#   و خوراک‌دهی با Keras Sequence
# =========================================================
def load_ljspeech_items(root_dir: str, metadata_name: str = "metadata.csv") -> List[Tuple[str, str]]:
    """
    ساختار استاندارد LJSpeech:
      root_dir/
        wavs/*.wav
        metadata.csv  (pipe-delimited: <id>|<transcript>|<normalized>)
    خروجی: [(wav_path, text), ...]  متن: ستون دوم (transcript)
    """
    meta_path = os.path.join(root_dir, metadata_name)
    items: List[Tuple[str, str]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                utt_id, transcript = parts[0], parts[1]
                wav_path = os.path.join(root_dir, "wavs", f"{utt_id}.wav")
                if os.path.exists(wav_path):
                    items.append((wav_path, transcript))
    return items

def _trim_silence_librosa(y: np.ndarray, sr: int, enable: bool, top_db: float):
    if not enable:
        return y
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt

def wav_to_logmel_np(path: str, cfg: AudioCfg) -> np.ndarray:
    y, sr = sf.read(path)  # y: (T,) or (T, C)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != cfg.target_sample_rate:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.target_sample_rate, res_type="soxr_hq")
        except Exception:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.target_sample_rate, res_type="kaiser_best")
    y = _trim_silence_librosa(y, cfg.target_sample_rate, cfg.trim_silence, cfg.trim_threshold_db)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.target_sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
        norm="slaney",
        htk=False,
        center=True,
        window="hann",
    )
    logmel = librosa.power_to_db(S, top_db=100.0)  # [-100, 0]
    logmel = np.clip(logmel, -100.0, 0.0)
    logmel = (logmel + 100.0) / 100.0 * 2.0 - 1.0   # → [-1, 1]
    return logmel.T.astype(np.float32)              # (frames, n_mels)

# ====== Simple on-disk cache for mel features ======
def _mel_cache_key(wav_path: str, cfg: AudioCfg) -> str:
    try:
        st = os.stat(wav_path)
        sig = f"{wav_path}|{st.st_size}|{st.st_mtime}|{cfg.target_sample_rate}|{cfg.n_fft}|{cfg.hop_length}|{cfg.win_length}|{cfg.n_mels}|{cfg.fmin}|{cfg.fmax}|{cfg.trim_silence}|{cfg.trim_threshold_db}"
    except FileNotFoundError:
        sig = f"{wav_path}|missing|{cfg}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
    return h

def _mel_cache_path(cache_dir: str, wav_path: str, cfg: AudioCfg) -> str:
    key = _mel_cache_key(wav_path, cfg)
    sub = os.path.join(cache_dir, "mels", key[:2], key[2:4])  # shard to avoid huge dirs
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{key}.npy")

def _load_or_compute_mel_cached(wav_path: str, cfg: AudioCfg, cache_dir: Optional[str]):
    if cache_dir:
        path = _mel_cache_path(cache_dir, wav_path, cfg)
        if os.path.exists(path):
            try:
                mel = np.load(path)
                return mel, mel.shape[0], True
            except Exception:
                pass  # fallthrough to recompute
    mel = wav_to_logmel_np(wav_path, cfg)
    if cache_dir:
        out_path = _mel_cache_path(cache_dir, wav_path, cfg)
        try:
            np.save(out_path, mel)
        except Exception:
            pass
    return mel, mel.shape[0], False

def tokenize_texts(items: List[Tuple[str, str]], tok: AutoTokenizer, text_cfg: TextCfg):
    ids_list = []
    for _, txt in items:
        # Pass src_lang explicitly instead of setting tok.src_lang
        ids = tok.encode(txt, add_special_tokens=True, src_lang=text_cfg.lang_code)[: text_cfg.max_text_len]
        ids_list.append(np.asarray(ids, dtype=np.int32))
    return ids_list

def _mel_worker(args):
    """Worker for parallel mel computation with optional cache.
    args=(wav_path, audio_cfg, cache_dir)
    """
    wav_path, audio_cfg, cache_dir = args
    mel, T, _ = _load_or_compute_mel_cached(wav_path, audio_cfg, cache_dir)
    return mel, T

def preprocess_dataset(
    root_dir: str,
    audio_cfg: AudioCfg,
    text_cfg: TextCfg,
    tok,
    metadata_name: str = "metadata.csv",
    num_workers: int = None,
    cache_dir: Optional[str] = None,
):
    """
    Offline preprocessing (tokenize + wav→mel) with optional parallel mel computation.
    num_workers: if None, uses max(1, os.cpu_count()-1). Set to 1 to disable parallelism.
    """
    items = load_ljspeech_items(root_dir, metadata_name=metadata_name)
    text_ids = tokenize_texts(items, tok, text_cfg)

    # Decide worker count
    if num_workers is None:
        try:
            hw_threads = os.cpu_count() or 1
        except Exception:
            hw_threads = 1
        num_workers = max(1, hw_threads - 1)

    wav_paths = [wav for (wav, _) in items]

    cache_hits = 0
    if num_workers <= 1 or len(wav_paths) <= 1:
        mels = []
        mel_lens = []
        for wav in wav_paths:
            mel, T, hit = _load_or_compute_mel_cached(wav, audio_cfg, cache_dir)
            cache_hits += int(bool(hit))
            mels.append(mel)
            mel_lens.append(T)
    else:
        # Parallel computation using process pool
        args_iter = [(wav, audio_cfg, cache_dir) for wav in wav_paths]
        mels = []
        mel_lens = []
        # choose a reasonable chunksize to reduce IPC overhead
        chunksize = max(1, len(wav_paths) // (num_workers * 8) if num_workers else 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            for mel, T in ex.map(_mel_worker, args_iter, chunksize=chunksize):
                mels.append(mel)
                mel_lens.append(T)

    if cache_dir:
        try:
            print(f"[preprocess] mel cache: hits={cache_hits}/{len(wav_paths)} dir={os.path.abspath(cache_dir)}")
        except Exception:
            pass

    return items, text_ids, mels, np.asarray(mel_lens, dtype=np.int32)

# ====== Sequence utilities ======
def _pad_1d(seqs, pad_val=0):
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L), pad_val, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out

def _pad_2d(seqs, feat_dim, pad_val=0.0):
    T = max(s.shape[0] for s in seqs)
    out = np.full((len(seqs), T, feat_dim), pad_val, dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0], :] = s
    return out

def _shift_right_mel_np(mel_batch):
    B, T, M = mel_batch.shape
    out = np.zeros((B, T, M), dtype=np.float32)
    out[:, 1:, :] = mel_batch[:, :-1, :]
    return out

class TTSDataset(tf.keras.utils.Sequence):
    """
    text_ids_list:  list[list[int]]    (طول‌های مختلف)
    mels_list:      list[np.ndarray]   هر mel با شکل (T, n_mels)
    """
    def __init__(self,
                 text_ids_list,
                 mels_list,
                 *,
                 batch_size: int,
                 pad_id: int,
                 n_mels: int,
                 max_src_len: int,
                 max_mel_len: int,
                 shuffle: bool = True):
        assert len(text_ids_list) == len(mels_list)
        self.text_ids = text_ids_list
        self.mels = mels_list
        self.bs = int(batch_size)
        self.pad_id = int(pad_id)
        self.n_mels = int(n_mels)
        self.max_src_len = int(max_src_len)
        self.max_mel_len = int(max_mel_len)
        self.shuffle = bool(shuffle)
        self.indices = np.arange(len(self.text_ids))
        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.indices) / self.bs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    @staticmethod
    def _shift_right_mel_np(mel_batch, pad_val=0.0):
        # mel_batch: (B, T, M)
        B, T, M = mel_batch.shape
        out = np.zeros_like(mel_batch, dtype=np.float32)
        out[:, 1:, :] = mel_batch[:, :-1, :]
        out[:, 0, :] = pad_val
        return out

    def __getitem__(self, idx):
        idxs = self.indices[idx*self.bs:(idx+1)*self.bs]

        B = len(idxs)
        enc = np.full((B, self.max_src_len), fill_value=self.pad_id, dtype=np.int32)
        mel = np.zeros((B, self.max_mel_len, self.n_mels), dtype=np.float32)
        mel_len = np.zeros((B,), dtype=np.int32)

        for i, k in enumerate(idxs):
            ids = self.text_ids[k]
            mel_np = self.mels[k]  # (T, n_mels)

            # --- متن: truncate + pad ثابت ---
            ids = ids[:self.max_src_len]
            enc[i, :len(ids)] = np.asarray(ids, dtype=np.int32)

            # --- mel: truncate + pad ثابت ---
            T = min(mel_np.shape[0], self.max_mel_len)
            mel[i, :T, :] = mel_np[:T, :]
            mel_len[i] = T  # طول واقعی بعد از clamp

        # teacher forcing
        dec_in = self._shift_right_mel_np(mel, pad_val=0.0)

        # stop targets: بعد از mel_len ⇒ 1
        stop = np.zeros((B, self.max_mel_len, 1), dtype=np.float32)
        for i in range(B):
            if mel_len[i] < self.max_mel_len:
                stop[i, mel_len[i]:, 0] = 1.0

        # خروجی‌های مدل (inputs, targets)
        inputs = {"enc_ids": enc, "dec_mel": dec_in, "mel_len": mel_len}
        targets = {"mel_pre": mel, "mel_post": mel, "stop": stop}
        return inputs, targets
# =========================================================
# (اختیاری) ساخت ورودی teacher forcing برای Keras.fit با tf.data
# =========================================================
def shift_right_mel(mel, pad_val=0.0):
    b = tf.shape(mel)[0]
    m = tf.shape(mel)[2]
    pad = tf.fill([b, 1, m], tf.cast(pad_val, mel.dtype))
    return tf.concat([pad, mel[:, :-1, :]], axis=1)

def add_teacher_forcing(example):
    dec_in = shift_right_mel(example["mel"], pad_val=0.0)
    stop_targets = 1.0 - tf.sequence_mask(
        example["mel_len"], maxlen=tf.shape(example["mel"])[1], dtype=tf.float32
    )[..., None]  # (B,T,1)  1=بعد از آخرین فریم
    return (
        {"enc_ids": example["text_ids"], "dec_mel": dec_in, "mel_len": example["mel_len"]},
        {"mel_pre": example["mel"], "mel_post": example["mel"], "stop": stop_targets}
    )

# =========================================================
# Example (اجرای مستقیم فایل) – دموی کوتاه
# =========================================================
if __name__ == "__main__":
    # GPU mem growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    # --- Demo: مسیر آفلاین (پیشنهادی وقتی RAM زیاد داری) ---
    root = "./LJSpeech-1.1"
    audio_cfg = AudioCfg(
        sample_rate=22050, target_sample_rate=22050,
        n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, fmin=0.0, fmax=8000.0, trim_silence=False
    )
    text_cfg = TextCfg(pad_id=1, bos_id=0, eos_id=2, max_text_len=256, lang_code="eng_Latn")

    print("Preprocessing dataset offline (tokenize + wav→mel)...")
    items, text_ids, mels, mel_lens = preprocess_dataset(root, audio_cfg, text_cfg)

    print(f"Total items: {len(items)}, example mel shape: {mels[0].shape}, text len: {len(text_ids[0])}")

    train_gen = TTSDataset(text_ids, mels, batch_size=4, pad_id=text_cfg.pad_id, shuffle=True, bucket=True)
    x, y = train_gen[0]
    print("One batch shapes (offline):", {k: v.shape for k, v in x.items()}, {k: v.shape for k, v in y.items()})

    # --- Demo: مسیر tf.data (در صورت نیاز) ---
    tok = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", use_fast=False, src_lang="eng_Latn"
    )
    items_tf = load_ljspeech_items(root)
    ds = build_dataset(items_tf, hf_tokenizer=tok, audio_cfg=audio_cfg, text_cfg=text_cfg, pipeline_cfg=PipelineCfg(batch_size=4))
    batch = next(iter(ds.take(1)))
    print("One batch keys (tf.data):", {k: v.shape for k, v in batch.items()})
