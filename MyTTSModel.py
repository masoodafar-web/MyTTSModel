# =========================
# Diffusion Model Components (DDPM-style)
# =========================
import tensorflow as tf
from tensorflow.keras import layers

class SinusoidalPositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for diffusion timesteps.
    """

    def __init__(self, dim, max_period=10000, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.max_period = max_period

    def call(self, timesteps):
        # timesteps: (B,) or scalar
        half_dim = self.dim // 2
        exponents = tf.range(half_dim, dtype=tf.float32) / half_dim
        freqs = tf.exp(-tf.math.log(self.max_period) * exponents)
        args = timesteps[:, None] * freqs[None, :]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


class DiffusionDenoiser(layers.Layer):
    """
    Diffusion denoiser network for Tortoise-style mel generation.
    Uses a Transformer decoder conditioned on text and voice latents.
    """

    def __init__(self, num_layers, d_model, num_heads, dff, n_mels,
                 dropout_rate=0.1, max_length=4096, name="diffusion_denoiser"):
        super().__init__(name=name)
        self.n_mels = n_mels
        self.time_embed = SinusoidalPositionalEncoding(d_model)

        # Mel projection (noisy mel input)
        self.mel_proj = MelPositionalProjection(n_mels, d_model, max_length=max_length, name="mel_proj")

        # Time embedding projection
        self.time_proj = layers.Dense(d_model, name="time_proj")

        # Voice latent conditioning (from autoencoder)
        self.voice_proj = layers.Dense(d_model, name="voice_proj")

        # Cross-attention layers for text and voice conditioning
        self.cross_attn_text = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, name="cross_attn_text"
        )
        self.cross_attn_voice = BaseAttentionPreNorm(
            num_heads=num_heads, d_model=d_model, dropout=dropout_rate, name="cross_attn_voice"
        )

        # Decoder layers
        self.layers_ = []
        for i in range(num_layers):
            dp = 0.0 if dropout_rate == 0.0 else dropout_rate * (i / max(1, num_layers - 1))
            self.layers_.append(
                DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff,
                            dropout_rate=dropout_rate, droppath=dp, name=f"dec_layer_{i}")
            )

        self.dropout = layers.Dropout(dropout_rate)
        self.final_norm = layers.LayerNormalization(epsilon=1e-5, name="final_norm")
        self.output_proj = layers.Dense(n_mels, name="output_proj")

    def call(self, noisy_mel, timesteps, text_context, voice_latents, training=False):
        """
        Denoise mel-spectrogram conditioned on text and voice.

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
        t_emb = self.time_embed(timesteps)  # (B, D)
        t_emb = self.time_proj(t_emb)  # (B, D)

        # Mel projection
        x = self.mel_proj(noisy_mel)  # (B, T, D)

        # Add time embedding to mel
        x = x + t_emb[:, None, :]  # Broadcast to (B, T, D)

        # Voice conditioning projection
        voice_cond = self.voice_proj(voice_latents)  # (B, L, D)

        # Cross-attention with text
        x = self.cross_attn_text(x, context=text_context, training=training)

        # Cross-attention with voice
        x = self.cross_attn_voice(x, context=voice_cond, training=training)

        # Decoder layers (self-attention)
        T = tf.shape(x)[1]
        causal_mask = combine_causal_and_keypadding(tf.ones((tf.shape(x)[0], T), dtype=tf.bool))
        for layer in self.layers_:
            x = layer(x, context=x, dec_causal_key_mask=causal_mask, training=training)

        x = self.final_norm(x)
        x = self.dropout(x, training=training)
        return self.output_proj(x)  # (B, T, n_mels)


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

        # Text encoder (shared with original)
        self.text_encoder = Encoder(num_layers=num_layers, d_model=d_model,
                                   num_heads=num_heads, dff=dff,
                                   vocab_size=input_vocab_size, name="text_encoder")

        # Voice autoencoder for conditioning
        self.voice_autoencoder = VoiceAutoencoder(latent_dim=latent_dim, n_mels=n_mels)

        # RVQ for mel quantization
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=4, num_embeddings=1024, embedding_dim=n_mels
        )

        # Diffusion denoiser
        self.denoiser = DiffusionDenoiser(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, n_mels=n_mels
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
        text_valid = make_padding_bool(text_ids, self.text_encoder.pos_embedding.embedding.vocab_size - 1)
        text_key_mask = key_mask_from_valid(text_valid)
        text_context = self.text_encoder(text_ids, key_mask=text_key_mask, q_valid=text_valid, training=training)

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
        text_valid = make_padding_bool(text_ids, self.text_encoder.pos_embedding.embedding.vocab_size - 1)
        text_key_mask = key_mask_from_valid(text_valid)
        text_context = self.text_encoder(text_ids, key_mask=text_key_mask, q_valid=text_valid, training=False)

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
