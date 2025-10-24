# =========================
# Diffusion Model Components (DDPM-style)
# =========================
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class SinusoidalPositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for sequences (additive on embeddings).
    """

    def __init__(self, dim, max_period=10000, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.max_period = max_period

    def call(self, x):
        # x: (B, S, d_model) - input tensor for positional encoding
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len, dtype=tf.float32)  # (S,)

        half_dim = self.dim // 2
        exponents = tf.range(half_dim, dtype=tf.float32) / half_dim
        freqs = tf.exp(-tf.math.log(float(self.max_period)) * exponents)
        args = positions[:, None] * freqs[None, :]  # (S, half_dim)

        pos_enc = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # (S, dim)
        pos_enc = tf.expand_dims(pos_enc, axis=0)  # (1, S, dim)
        pos_enc = tf.cast(pos_enc, x.dtype)  # Match input dtype

        return x + pos_enc

    def compute_output_shape(self, input_shape):
        return input_shape


class SinusoidalTimestepEncoding(layers.Layer):
    """
    Sinusoidal encoding for diffusion timesteps.
    Returns a vector per batch element.
    """

    def __init__(self, dim, max_period=10000, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.max_period = max_period

    def call(self, timesteps):
        # timesteps: (B,) or scalar
        timesteps = tf.reshape(timesteps, [-1])
        half_dim = self.dim // 2
        exponents = tf.range(half_dim, dtype=tf.float32) / tf.cast(half_dim, tf.float32)
        freqs = tf.exp(-tf.math.log(tf.cast(self.max_period, tf.float32)) * exponents)
        args = timesteps[:, None] * freqs[None, :]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


class DiffusionDenoiserUNet(layers.Layer):
    """
    U-Net based diffusion denoiser for Tortoise-style mel generation.
    Similar to the U-Net architecture used in original Tortoise TTS.
    """

    def __init__(self, n_mels, base_channels=128, channel_mults=[1, 2, 4, 8],
                 dropout_rate=0.1, name="diffusion_denoiser_unet"):
        super().__init__(name=name)
        self.n_mels = n_mels
        self.channel_mults = channel_mults
        self.base_channels = base_channels

        # Time embedding for diffusion timesteps
        self.time_embed = SinusoidalTimestepEncoding(base_channels * 4)
        self.time_proj = layers.Dense(base_channels * 4, activation="silu", name="time_proj")

        # Input projection
        self.input_proj = layers.Conv1D(base_channels, 3, padding="same", name="input_proj")

        # Encoder blocks (downsampling)
        self.encoder_blocks = []
        self.downsample_blocks = []
        channels = base_channels
        for i, mult in enumerate(channel_mults):
            # Residual blocks
            self.encoder_blocks.append([
                ResBlock(channels, dropout_rate, name=f"enc_res_{i}_0"),
                ResBlock(channels, dropout_rate, name=f"enc_res_{i}_1"),
            ])

            # Downsampling (except last)
            if i < len(channel_mults) - 1:
                self.downsample_blocks.append(
                    layers.Conv1D(channels * 2, 4, strides=2, padding="same", name=f"down_{i}")
                )
                channels *= 2

        # Middle block
        self.middle_block = [
            ResBlock(channels, dropout_rate, name="mid_res_0"),
            ResBlock(channels, dropout_rate, name="mid_res_1"),
        ]

        # Decoder blocks (upsampling)
        self.decoder_blocks = []
        self.upsample_blocks = []
        self.skip_connections = []
        for i, mult in enumerate(reversed(channel_mults)):
            # Skip connection projection
            skip_channels = base_channels * mult
            self.skip_connections.append(
                layers.Conv1D(channels, 1, name=f"skip_proj_{i}")
            )

            # Residual blocks
            self.decoder_blocks.append([
                ResBlock(channels, dropout_rate, name=f"dec_res_{i}_0"),
                ResBlock(channels, dropout_rate, name=f"dec_res_{i}_1"),
            ])

            # Upsampling (except last)
            if i < len(channel_mults) - 1:
                self.upsample_blocks.append(
                    layers.Conv1DTranspose(channels // 2, 4, strides=2, padding="same", name=f"up_{i}")
                )
                channels //= 2

        # Output projection
        self.output_proj = layers.Conv1D(n_mels, 3, padding="same", name="output_proj")

        # Conditioning projections
        self.text_conditioning = layers.Dense(base_channels, name="text_cond")
        self.voice_conditioning = layers.Dense(base_channels, name="voice_cond")

    def call(self, noisy_mel, timesteps, text_context, voice_latents, training=False):
        """
        U-Net forward pass for denoising.

        Args:
            noisy_mel: (B, T, n_mels) - Noisy mel input
            timesteps: (B,) - Diffusion timesteps
            text_context: (B, S, D) - Text encoder outputs
            voice_latents: (B, L, D) - Voice conditioning latents
            training: bool

        Returns:
            (B, T, n_mels) - Denoised mel prediction
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)  # (B, base_channels*4)
        t_emb = self.time_proj(t_emb)  # (B, base_channels*4)

        # Conditioning
        text_cond = self.text_conditioning(text_context)  # (B, S, base_channels)
        voice_cond = self.voice_conditioning(voice_latents)  # (B, L, base_channels)

        # Input projection
        x = self.input_proj(noisy_mel)  # (B, T, base_channels)
        x = tf.nn.silu(x)

        # Encoder path with skip connections
        skips = []
        for i, (res_blocks, downsample) in enumerate(zip(self.encoder_blocks, self.downsample_blocks + [None])):
            # Apply conditioning
            if i == 0:
                # Add text conditioning to first encoder block
                text_cond_avg = tf.reduce_mean(text_cond, axis=1, keepdims=True)  # (B, 1, base_channels)
                x = x + text_cond_avg

            for res_block in res_blocks:
                x = res_block(x, t_emb, training=training)

            skips.append(x)

            if downsample is not None:
                x = downsample(x)
                x = tf.nn.silu(x)

        # Middle
        for res_block in self.middle_block:
            x = res_block(x, t_emb, training=training)

        # Decoder path
        for i, (res_blocks, upsample) in enumerate(zip(self.decoder_blocks, self.upsample_blocks + [None])):
            if upsample is not None:
                x = upsample(x)
                x = tf.nn.silu(x)

            # Skip connection
            skip = self.skip_connections[i](skips[-(i+1)])
            x = tf.concat([x, skip], axis=-1)

            # Apply conditioning
            if i == len(self.decoder_blocks) - 1:
                # Add voice conditioning to last decoder block
                voice_cond_avg = tf.reduce_mean(voice_cond, axis=1, keepdims=True)  # (B, 1, base_channels)
                x = x + voice_cond_avg

            for res_block in res_blocks:
                x = res_block(x, t_emb, training=training)

        # Output
        return self.output_proj(x)


class ResBlock(layers.Layer):
    """
    Residual block for U-Net with time conditioning.
    """

    def __init__(self, channels, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.channels = channels

        self.norm1 = layers.GroupNormalization(groups=min(32, channels), name="norm1")
        self.conv1 = layers.Conv1D(channels, 3, padding="same", name="conv1")

        self.norm2 = layers.GroupNormalization(groups=min(32, channels), name="norm2")
        self.conv2 = layers.Conv1D(channels, 3, padding="same", name="conv2")

        self.time_proj = layers.Dense(channels, name="time_proj")

        self.dropout = layers.Dropout(dropout_rate, name="dropout")
        self.residual_conv = layers.Conv1D(channels, 1, name="residual_conv") if channels != channels else None

    def call(self, x, time_emb, training=False):
        """
        Args:
            x: (B, T, C) - Input features
            time_emb: (B, C) - Time embedding
            training: bool
        """
        h = self.norm1(x)
        h = tf.nn.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb_proj = self.time_proj(time_emb)  # (B, C)
        h = h + time_emb_proj[:, None, :]  # Broadcast to (B, T, C)

        h = self.norm2(h)
        h = tf.nn.silu(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)

        # Residual connection
        if self.residual_conv is not None:
            x = self.residual_conv(x)
        return x + h


class VoiceAutoencoder(tf.keras.Model):
    """
    Autoencoder for voice conditioning latents (similar to Tortoise's CLVP).
    """

    def __init__(self, latent_dim=512, n_mels=80, name="voice_autoencoder"):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(256, 3, strides=2, padding="same", activation="relu"),
            layers.Conv1D(256, 3, strides=2, padding="same", activation="relu"),
            layers.Conv1D(latent_dim, 3, strides=2, padding="same"),
            layers.GlobalAveragePooling1D(),
        ], name="encoder")

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.RepeatVector(10),  # Assuming ~10 time steps after pooling
            layers.Conv1DTranspose(256, 3, strides=2, padding="same", activation="relu"),
            layers.Conv1DTranspose(256, 3, strides=2, padding="same", activation="relu"),
            layers.Conv1DTranspose(n_mels, 3, strides=2, padding="same"),
        ], name="decoder")

    def call(self, mel, training=False):
        latent = self.encoder(mel, training=training)
        reconstructed = self.decoder(latent, training=training)
        return reconstructed, latent


class TortoiseDiffusionTTS(tf.keras.Model):
    """
    Tortoise-style TTS with diffusion model for mel generation.
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, n_mels, latent_dim=512,
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 name="TortoiseDiffusionTTS"):
        super().__init__(name=name)
        self.n_mels = n_mels
        self.num_timesteps = num_timesteps

        # Text encoder (simplified transformer encoder)
        self.text_encoder = tf.keras.Sequential([
            layers.Embedding(input_vocab_size, d_model),
            layers.LayerNormalization(epsilon=1e-5),
            *[tf.keras.Sequential([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads),
                layers.LayerNormalization(epsilon=1e-5),
                layers.Dense(dff, activation="gelu"),
                layers.Dense(d_model),
                layers.LayerNormalization(epsilon=1e-5),
            ]) for _ in range(num_layers)],
        ], name="text_encoder")

        # Voice autoencoder for conditioning
        self.voice_autoencoder = VoiceAutoencoder(latent_dim=latent_dim, n_mels=n_mels)

        # RVQ for mel quantization
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=4, num_embeddings=1024, embedding_dim=n_mels
        )

        # Diffusion denoiser (U-Net based like Tortoise)
        self.denoiser = DiffusionDenoiserUNet(
            n_mels=n_mels, base_channels=d_model // 4,  # Adjust based on d_model
            channel_mults=[1, 2, 4, 8], dropout_rate=0.1
        )

        # Diffusion schedule (DDPM)
        self.beta_schedule = tf.linspace(beta_start, beta_end, num_timesteps)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_cumprod = tf.math.cumprod(self.alpha_schedule)

    def q_sample(self, x_0, t):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        sqrt_alpha_cumprod = tf.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = tf.sqrt(1.0 - self.alpha_cumprod[t])

        noise = tf.random.normal(tf.shape(x_0), dtype=x_0.dtype)
        return sqrt_alpha_cumprod[:, None, None] * x_0 + sqrt_one_minus_alpha_cumprod[:, None, None] * noise, noise

    def p_sample(self, x_t, t, text_context, voice_latents, training=False):
        """
        Reverse diffusion step: p(x_{t-1} | x_t)
        """
        predicted_noise = self.denoiser(x_t, t, text_context, voice_latents, training=training)

        alpha_t = self.alpha_schedule[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        beta_t = self.beta_schedule[t]

        sqrt_one_minus_alpha_cumprod_t = tf.sqrt(1.0 - alpha_cumprod_t)
        sqrt_recip_alpha_t = 1.0 / tf.sqrt(alpha_t)

        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

        if t[0] > 0:
            noise = tf.random.normal(tf.shape(x_t), dtype=x_t.dtype)
            variance = beta_t
        else:
            noise = tf.zeros_like(x_t)
            variance = 0.0

        return mean + tf.sqrt(variance)[:, None, None] * noise

    def call(self, inputs, training=False):
        """
        Training forward pass with diffusion.
        """
        mel_target = inputs["mel"]  # (B, T, n_mels)
        text_ids = inputs["text_ids"]  # (B, S)
        voice_mel = inputs.get("voice_mel", mel_target)  # For conditioning

        # Encode text
        text_context = self.text_encoder(text_ids, training=training)

        # Voice conditioning
        _, voice_latents = self.voice_autoencoder(voice_mel, training=training)

        # RVQ quantization for target
        quantized_mel, rvq_indices, rvq_loss, rvq_perplexity = self.rvq(mel_target)

        # Diffusion training
        t = tf.random.uniform((tf.shape(mel_target)[0],), 0, self.num_timesteps, dtype=tf.int32)
        noisy_mel, noise = self.q_sample(quantized_mel, t)

        # Denoise
        predicted_noise = self.denoiser(noisy_mel, t, text_context, voice_latents, training=training)

        # Diffusion loss
        diffusion_loss = tf.reduce_mean((predicted_noise - noise)**2)

        return {
            "diffusion_loss": diffusion_loss,
            "rvq_loss": rvq_loss,
            "rvq_perplexity": rvq_perplexity,
            "total_loss": diffusion_loss + rvq_loss
        }

    def generate(self, text_ids, voice_mel, num_steps=50, guidance_scale=1.0):
        """
        Generate mel using diffusion sampling with optional classifier-free guidance.

        Args:
            text_ids: Input text token IDs (B, S)
            voice_mel: Reference voice mel for conditioning (B, T_ref, n_mels)
            num_steps: Number of diffusion steps to use (fewer = faster but lower quality)
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = more guidance)

        Returns:
            Generated mel-spectrogram (B, T, n_mels)
        """
        B = tf.shape(text_ids)[0]
        T = 200  # Target mel length (can be adjusted)

        # Encode text and voice
        text_context = self.text_encoder(text_ids, training=False)

        _, voice_latents = self.voice_autoencoder(voice_mel, training=False)

        # Start from noise
        x_t = tf.random.normal((B, T, self.n_mels), dtype=tf.float32)

        # Reverse diffusion with optional guidance
        for t in reversed(range(0, self.num_timesteps, self.num_timesteps // num_steps)):
            t_tensor = tf.fill((B,), t)

            if guidance_scale > 1.0:
                # Classifier-free guidance: predict with and without conditioning
                # For simplicity, we use unconditional prediction by zeroing text/voice
                text_context_uncond = tf.zeros_like(text_context)
                voice_latents_uncond = tf.zeros_like(voice_latents)

                # Conditional prediction
                x_t_cond = self.p_sample(x_t, t_tensor, text_context, voice_latents, training=False)

                # Unconditional prediction
                x_t_uncond = self.p_sample(x_t, t_tensor, text_context_uncond, voice_latents_uncond, training=False)

                # Apply guidance
                x_t = x_t_uncond + guidance_scale * (x_t_cond - x_t_uncond)
            else:
                # Standard sampling without guidance
                x_t = self.p_sample(x_t, t_tensor, text_context, voice_latents, training=False)

        return x_t

    def generate_with_encodec(self, text_ids, voice_mel, num_steps=50, guidance_scale=1.0):
        """
        Generate mel and convert to audio using Encodec for high-quality output.

        Returns:
            Generated audio waveform as tf.Tensor
        """
        mel = self.generate(text_ids, voice_mel, num_steps, guidance_scale)

        # Convert mel to audio using Encodec for better quality than Griffin-Lim
        try:
            from codec.encodec_codec import Encodec24k
            encodec = Encodec24k()

            # Convert normalized mel [-1,1] to audio
            mel_norm = mel.numpy()[0]  # Remove batch dim, (T, n_mels)

            # Denormalize mel to power scale
            mel_01 = (mel_norm + 1.0) * 0.5  # [-1,1] -> [0,1]
            mel_db = mel_01 * 100.0 - 100.0  # -> [-100, 0] dB
            mel_power = tf.pow(10.0, mel_db / 10.0)  # -> power

            # Convert to linear spectrogram
            n_fft = 1024  # Should match audio config
            n_mels = self.n_mels
            mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=n_mels,
                num_spectrogram_bins=n_fft // 2 + 1,
                sample_rate=24000,  # Encodec 24kHz
                lower_edge_hertz=0.0,
                upper_edge_hertz=12000.0
            )
            linear_power = tf.matmul(mel_power, tf.linalg.pinv(mel_matrix))

            # Convert to magnitude
            mag = tf.sqrt(tf.maximum(linear_power, 1e-10))

            # Griffin-Lim for initial phase reconstruction
            wav_gl = self._griffin_lim_simple(mag.numpy(), n_iter=16)

            # Re-encode with Encodec for quality enhancement
            audio_24k = encodec.reencode_to_24k(wav_gl, input_sr=24000)

            return tf.convert_to_tensor(audio_24k, dtype=tf.float32)

        except ImportError:
            print("⚠️ Encodec not available, falling back to Griffin-Lim")
            # Fallback to Griffin-Lim
            from MyTTSModelTrain import TensorBoardAudioLogger
            audio_logger = TensorBoardAudioLogger(None, [], use_encodec=False)
            audio = audio_logger._mel_to_waveform(mel.numpy()[0])
            return audio

    @staticmethod
    def _griffin_lim_simple(mag, n_iter=16):
        """Simple Griffin-Lim for phase reconstruction."""
        mag = tf.cast(mag, tf.float32)
        theta = tf.random.uniform(tf.shape(mag), 0.0, 2 * np.pi, dtype=tf.float32)
        phase = tf.complex(tf.cos(theta), tf.sin(theta))
        S = mag * phase

        for _ in range(n_iter):
            wav = tf.signal.inverse_stft(
                S, frame_length=1024, frame_step=256, window_fn=tf.signal.hann_window
            )
            S = tf.signal.stft(
                wav, frame_length=1024, frame_step=256, window_fn=tf.signal.hann_window
            )
            angles = tf.math.angle(S)
            phase = tf.complex(tf.cos(angles), tf.sin(angles))
            S = mag * phase

        wav_final = tf.signal.inverse_stft(
            S, frame_length=1024, frame_step=256, window_fn=tf.signal.hann_window
        )
        return tf.cast(wav_final, tf.float32)
# =========================
# Transformer TTS Model
# =========================
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class TransformerTTS(tf.keras.Model):
    """
    Transformer-based Text-to-Speech model similar to FastSpeech/Tacotron 2.

    This implementation includes:
    - Transformer encoder for text processing
    - Duration predictor for length regulation
    - Decoder with prenet and postnet
    - Stop token prediction
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, n_mels,
                 dropout_rate=0.1, pad_id=0, use_prenet=True, prenet_drop=0.5,
                 droppath_rate=0.0, cross_win=None, enable_postnet=True,
                 activation='gelu', pos_encoding_type='sinusoidal',
                 target_type='mel', num_codebooks=None, codebook_size=None, **kwargs):
        super().__init__(**kwargs)

        self.pad_id = pad_id
        self.n_mels = n_mels
        self.target_type = target_type
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # Output dimensions based on target type
        if target_type == 'codes':
            self.output_dim = num_codebooks * codebook_size
        else:
            self.output_dim = n_mels

        # Encoder
        self.encoder_embedding = layers.Embedding(input_vocab_size, d_model)
        if pos_encoding_type == 'sinusoidal':
            self.encoder_pos_encoding = SinusoidalPositionalEncoding(d_model)
        else:
            self.encoder_pos_encoding = LearnedPositionalEncoding(d_model)

        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                TransformerBlock(d_model, num_heads, dff, dropout_rate, activation)
            )

        # Duration predictor (for length regulation)
        self.duration_predictor = DurationPredictor(d_model, dropout_rate)

        # Decoder prenet
        if use_prenet:
            self.prenet = PreNet(self.output_dim, prenet_drop)
        else:
            self.prenet = None

        # Decoder
        self.decoder_embedding = LinearEmbedding(self.output_dim, d_model)
        if pos_encoding_type == 'sinusoidal':
            self.decoder_pos_encoding = SinusoidalPositionalEncoding(d_model)
        else:
            self.decoder_pos_encoding = LearnedPositionalEncoding(d_model)

        self.decoder_layers = []
        for i in range(num_layers):
            self.decoder_layers.append(
                TransformerBlock(d_model, num_heads, dff, dropout_rate, activation)
            )

        # Output projections
        self.mel_pre_projection = layers.Dense(self.output_dim)
        self.stop_projection = layers.Dense(1)

        # Postnet
        if enable_postnet:
            self.postnet = PostNet(self.output_dim, dropout_rate)
        else:
            self.postnet = None

        # Length regulator
        self.length_regulator = LengthRegulator()

    def build_for_load(self, max_src_len, max_tgt_len):
        """Build model by running a dummy forward pass on symbolic inputs.

        This ensures variables are created and `self.built` is set without
        baking in fixed batch dimensions from masks.
        """
        enc_ids_in = tf.keras.Input(shape=(max_src_len,), dtype=tf.int32, name='enc_ids')
        dec_mel_in = tf.keras.Input(shape=(max_tgt_len, self.output_dim), dtype=tf.float32, name='dec_mel')
        mel_len_in = tf.keras.Input(shape=(), dtype=tf.int32, name='mel_len')

        _ = self({
            'enc_ids': enc_ids_in,
            'dec_mel': dec_mel_in,
            'mel_len': mel_len_in,
        }, training=False)

        # Mark as built for Keras bookkeeping
        self.built = True
        print(f"✅ Model built with max_src_len={max_src_len}, max_tgt_len={max_tgt_len}")

    def call_encoder(self, src_ids, src_mask, training=False):
        """Encoder forward pass."""
        # Embedding + positional encoding
        x = self.encoder_embedding(src_ids)  # (B, S, d_model)
        x = tf.cast(x, tf.float32)  # Ensure consistent dtype
        x = x * tf.sqrt(tf.cast(self.encoder_embedding.output_dim, tf.float32))
        x = self.encoder_pos_encoding(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask, training=training)

        return x  # (B, S, d_model)

    def call_decoder(self, tgt_mel, tgt_mask, encoder_output, training=False):
        """Decoder forward pass."""
        # Prenet
        if self.prenet is not None:
            x = self.prenet(tgt_mel, training=training)  # (B, T, prenet_dim)
            x = self.decoder_embedding(x)  # (B, T, d_model)
        else:
            x = self.decoder_embedding(tgt_mel)  # (B, T, d_model)

        # Positional encoding
        x = self.decoder_pos_encoding(x)

        # Decoder layers (simplified - no cross-attention for this basic version)
        for layer in self.decoder_layers:
            x = layer(x, mask=tgt_mask, training=training)

        return x  # (B, T, d_model)

    def call(self, inputs, training=False, return_attn=False):
        """
        Full forward pass for training.

        Args:
            inputs: Dict with 'enc_ids', 'dec_mel', 'mel_len', etc.
            training: Training mode
            return_attn: Whether to return attention weights

        Returns:
            mel_pre, mel_post, stop_logits, [attention]
        """
        enc_ids = inputs['enc_ids']  # (B, S)
        dec_mel = inputs['dec_mel']  # (B, T, n_mels)
        mel_len = inputs.get('mel_len', None)  # (B,)

        # Ensure consistent dtypes for mixed precision
        enc_ids = tf.cast(enc_ids, tf.int32)
        dec_mel = tf.cast(dec_mel, tf.float32)
        if mel_len is not None:
            mel_len = tf.cast(mel_len, tf.int32)

        # Create masks
        batch_size = tf.shape(enc_ids)[0]
        src_len = tf.shape(enc_ids)[1]
        tgt_len = tf.shape(dec_mel)[1]

        # Source self-attention mask (B, S, S): allow attention only to valid tokens
        src_valid = enc_ids != self.pad_id                        # (B, S)
        src_mask = tf.logical_and(src_valid[:, None, :],          # (B, 1, S)
                                  src_valid[:, :, None])         # (B, S, 1)

        # Target mask (causal + padding)
        if mel_len is not None:
            tgt_padding_mask = tf.sequence_mask(mel_len, maxlen=tgt_len, dtype=tf.bool)  # (B, T)
        else:
            tgt_padding_mask = tf.ones((batch_size, tgt_len), dtype=tf.bool)
        causal_mask = tf.linalg.band_part(tf.ones((tgt_len, tgt_len), dtype=tf.bool), -1, 0)  # (T, T)
        tgt_mask = tf.logical_and(tgt_padding_mask[:, :, None], causal_mask[None, :, :])  # (B, T, T)

        # Encoder
        encoder_output = self.call_encoder(enc_ids, src_mask, training=training)

        # Decoder
        decoder_output = self.call_decoder(dec_mel, tgt_mask, encoder_output, training=training)

        # Output projections
        mel_pre = self.mel_pre_projection(decoder_output)  # (B, T, output_dim)

        # Stop token prediction
        stop_logits = self.stop_projection(decoder_output)  # (B, T, 1)

        # Postnet
        if self.postnet is not None:
            mel_residual = self.postnet(mel_pre, training=training)
            mel_post = mel_pre + mel_residual
        else:
            mel_post = mel_pre

        if return_attn:
            # Return dummy attention for compatibility
            attention = tf.zeros((batch_size, tgt_len, src_len), dtype=tf.float32)
            return mel_pre, mel_post, stop_logits, attention
        else:
            return mel_pre, mel_post, stop_logits

    def greedy_generate_fast(self, input_ids, max_steps=1000, min_steps=10, stop_threshold=0.5, verbose=False):
        """
        Fast greedy decoding for inference.

        Args:
            input_ids: (B, S) input token IDs
            max_steps: Maximum decoding steps
            min_steps: Minimum decoding steps
            stop_threshold: Stop token threshold
            verbose: Whether to print progress

        Returns:
            mel_output: (B, T, n_mels) generated mel
            stop_probs: (B, T) stop probabilities
        """
        batch_size = tf.shape(input_ids)[0]

        # Encoder
        src_valid = input_ids != self.pad_id
        src_mask = tf.logical_and(src_valid[:, None, :], src_valid[:, :, None])
        encoder_output = self.call_encoder(input_ids, src_mask, training=False)

        # Initialize decoder input (start with zeros)
        mel_output = []
        stop_probs = []

        # Start with a zero frame
        current_mel = tf.zeros((batch_size, 1, self.output_dim), dtype=tf.float32)

        for step in range(max_steps):
            # Decoder input (teacher forcing with generated so far)
            decoder_input = tf.concat(mel_output + [current_mel], axis=1) if mel_output else current_mel

            # Create target mask
            tgt_len = tf.shape(decoder_input)[1]
            causal_mask = tf.linalg.band_part(tf.ones((tgt_len, tgt_len), dtype=tf.bool), -1, 0)
            tgt_mask = tf.broadcast_to(causal_mask[None, :, :], (batch_size, tgt_len, tgt_len))

            # Decoder forward pass
            decoder_out = self.call_decoder(decoder_input, tgt_mask, encoder_output, training=False)

            # Get last frame predictions
            last_frame = decoder_out[:, -1:, :]  # (B, 1, d_model)
            mel_pred = self.mel_pre_projection(last_frame)  # (B, 1, output_dim)
            stop_logit = self.stop_projection(last_frame)  # (B, 1, 1)
            stop_prob = tf.sigmoid(stop_logit)  # (B, 1, 1)

            # Store predictions
            mel_output.append(mel_pred)
            stop_probs.append(tf.squeeze(stop_prob, axis=-1))

            # Check stop condition
            if step >= min_steps and tf.reduce_mean(stop_prob) > stop_threshold:
                if verbose:
                    print(f"Stopping at step {step} with stop prob {tf.reduce_mean(stop_prob).numpy():.3f}")
                break

            # Update current mel for next step
            current_mel = mel_pred

        # Concatenate outputs
        if mel_output:
            mel_output = tf.concat(mel_output, axis=1)  # (B, T, output_dim)
            stop_probs = tf.concat(stop_probs, axis=1)  # (B, T)
        else:
            mel_output = tf.zeros((batch_size, 0, self.output_dim), dtype=tf.float32)
            stop_probs = tf.zeros((batch_size, 0), dtype=tf.float32)

        return mel_output, stop_probs


class TransformerBlock(layers.Layer):
    """Transformer encoder/decoder block."""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, activation='gelu'):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation=activation),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        # Multi-head attention
        if mask is not None:
            # Convert mask to attention_mask format expected by MHA
            # MHA expects (B, num_heads, T, T) or (B, T, T)
            if len(mask.shape) == 3:
                # (B, T, T) - this should work directly with MHA
                attn_output = self.mha(x, x, attention_mask=mask, training=training)
            else:
                # Handle other formats
                attn_output = self.mha(x, x, attention_mask=mask, training=training)
        else:
            attn_output = self.mha(x, x, training=training)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def compute_output_shape(self, input_shape):
        return input_shape


class PreNet(layers.Layer):
    """Decoder prenet."""

    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.layers = [
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate)
        ]

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 256)


class PostNet(layers.Layer):
    """Mel postnet for residual refinement."""

    def __init__(self, mel_dim, dropout_rate=0.1):
        super().__init__()
        self.convs = []
        for i in range(5):
            in_channels = mel_dim if i == 0 else 512
            out_channels = mel_dim if i == 4 else 512
            self.convs.append(layers.Conv1D(out_channels, 5, padding='same', activation='tanh'))
            self.convs.append(layers.BatchNormalization())
            self.convs.append(layers.Dropout(dropout_rate))

    def call(self, x, training=False):
        # x: (B, T, mel_dim)
        for layer in self.convs:
            x = layer(x, training=training)
        return x


class DurationPredictor(layers.Layer):
    """Duration predictor for length regulation."""

    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.convs = [
            layers.Conv1D(256, 3, padding='same', activation='relu'),
            layers.Conv1D(256, 3, padding='same', activation='relu'),
            layers.Conv1D(256, 3, padding='same', activation='relu')
        ]
        self.linear = layers.Dense(1)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        for conv in self.convs:
            x = conv(x)
            x = self.dropout(x, training=training)
        x = self.linear(x)
        return x


class LengthRegulator(layers.Layer):
    """Length regulator to expand phonemes to frames."""

    def call(self, encoder_output, durations):
        """Expand encoder output based on predicted durations."""
        # Simple implementation - repeat each frame by duration
        batch_size = tf.shape(encoder_output)[0]
        expanded = []

        for b in range(batch_size):
            seq_expanded = []
            for t in range(tf.shape(encoder_output)[1]):
                duration = tf.maximum(tf.cast(durations[b, t], tf.int32), 1)
                repeated = tf.tile([encoder_output[b, t]], [duration, 1])
                seq_expanded.append(repeated)
            expanded.append(tf.concat(seq_expanded, axis=0))

        return tf.stack(expanded, axis=0)


class LearnedPositionalEncoding(layers.Layer):
    """Learned positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(max_len, d_model),
            initializer="uniform"
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_embedding[:seq_len]


class LinearEmbedding(layers.Layer):
    """Linear embedding for decoder input."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = layers.Dense(d_model)

    def call(self, x):
        return self.linear(x)


# =========================
# RVQ (Residual Vector Quantization)
# =========================
class VectorQuantizer(layers.Layer):
    """
    Single-layer Vector Quantizer for VQ-VAE.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, name=None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Embedding table: (num_embeddings, embedding_dim)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(num_embeddings, embedding_dim),
            initializer="uniform",
            trainable=True
        )

    def call(self, inputs):
        # inputs: (B, T, D) where D == embedding_dim
        input_shape = tf.shape(inputs)
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])  # (B*T, D)

        # Compute distances to embeddings
        distances = tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True) + \
                   tf.reduce_sum(self.embeddings**2, axis=1) - \
                   2 * tf.matmul(flat_inputs, self.embeddings, transpose_b=True)  # (B*T, num_emb)

        # Find nearest embedding indices
        encoding_indices = tf.argmin(distances, axis=1)  # (B*T,)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)  # (B*T, num_emb)

        # Quantized outputs
        quantized = tf.matmul(encodings, self.embeddings)  # (B*T, D)
        quantized = tf.reshape(quantized, input_shape)  # (B, T, D)

        # Straight-through estimator for gradients
        quantized_sg = inputs + tf.stop_gradient(quantized - inputs)

        # Loss: commitment loss + codebook loss
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
        q_latent_loss = tf.reduce_mean((quantized_sg - tf.stop_gradient(quantized))**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Perplexity for monitoring
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return quantized_sg, encoding_indices, loss, perplexity


class ResidualVectorQuantizer(layers.Layer):
    """
    Residual Vector Quantizer (RVQ) with multiple layers for better quantization.
    Used in Tortoise TTS for mel-spectrogram quantization.
    """

    def __init__(self, num_quantizers, num_embeddings, embedding_dim, commitment_cost=0.25, name=None):
        super().__init__(name=name)
        self.num_quantizers = num_quantizers
        self.quantizers = [
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, name=f"vq_{i}")
            for i in range(num_quantizers)
        ]

    def call(self, inputs):
        # inputs: (B, T, D)
        quantized = inputs
        total_loss = 0.0
        all_indices = []
        all_perplexities = []

        for quantizer in self.quantizers:
            quantized, indices, loss, perplexity = quantizer(quantized)
            total_loss += loss
            all_indices.append(indices)
            all_perplexities.append(perplexity)

        # Stack indices: (num_quantizers, B*T)
        indices_stack = tf.stack(all_indices, axis=0)
        avg_perplexity = tf.reduce_mean(tf.stack(all_perplexities))

        return quantized, indices_stack, total_loss, avg_perplexity

    def decode(self, indices_stack):
        """
        Decode from stacked indices back to quantized vectors.
        indices_stack: (num_quantizers, B*T)
        """
        B_T = tf.shape(indices_stack)[1]
        quantized = tf.zeros((B_T, self.quantizers[0].embedding_dim), dtype=tf.float32)

        for i, quantizer in enumerate(self.quantizers):
            indices = indices_stack[i]  # (B*T,)
            encodings = tf.one_hot(indices, quantizer.num_embeddings)  # (B*T, num_emb)
            quantized += tf.matmul(encodings, quantizer.embeddings)  # (B*T, D)

        return quantized
