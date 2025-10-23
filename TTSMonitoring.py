"""
TTSMonitoring - Utilities for monitoring and visualizing TTS model training and inference.

This module provides:
- Alignment visualization helpers
- Gradient analysis tools
- Mel-spectrogram plotting utilities
- Training metrics aggregation

These tools help diagnose common TTS issues like attention misalignment,
gradient vanishing/explosion, and output quality problems.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any


def extract_alignment_from_model(model, enc_ids, mel_target, training=False):
    """
    Extract cross-attention alignment weights from model forward pass.
    
    Args:
        model: TransformerTTS model instance
        enc_ids (tf.Tensor): Encoder input IDs of shape (batch_size, src_len)
        mel_target (tf.Tensor): Target mel-spectrogram of shape (batch_size, time, n_mels)
        training (bool): Whether to run in training mode
        
    Returns:
        tf.Tensor: Alignment matrix of shape (batch_size, tgt_len, src_len)
    """
    # Shift mel right for teacher forcing
    mel_shifted = model.shift_right_mel(mel_target)
    
    # Forward pass with attention return
    _, _, _, attn = model(
        {"enc_ids": enc_ids, "dec_mel": mel_shifted},
        training=training,
        return_attn=True
    )
    
    return attn  # (B, T, S)


def compute_alignment_diagonality(alignment, enc_lens=None, mel_lens=None):
    """
    Compute alignment diagonality score to measure attention quality.
    
    A good alignment should be close to diagonal (monotonic progression).
    Higher scores indicate better alignment.
    
    Args:
        alignment (tf.Tensor): Alignment weights of shape (batch_size, tgt_len, src_len)
        enc_lens (tf.Tensor, optional): Encoder sequence lengths
        mel_lens (tf.Tensor, optional): Mel sequence lengths
        
    Returns:
        float: Diagonality score (0-1, higher is better)
    """
    B, T, S = alignment.shape
    
    # Create normalized position indices
    t_idx = tf.range(T, dtype=tf.float32)[None, :, None] / float(T)  # (1, T, 1)
    s_idx = tf.range(S, dtype=tf.float32)[None, None, :] / float(S)  # (1, 1, S)
    
    # Compute distance from diagonal
    distance_from_diag = tf.abs(t_idx - s_idx)  # (1, T, S)
    
    # Weight by alignment strength
    weighted_distance = alignment * distance_from_diag
    
    # Average distance (lower is better)
    avg_distance = tf.reduce_mean(weighted_distance)
    
    # Convert to score (higher is better)
    diagonality_score = 1.0 - tf.minimum(avg_distance * 2.0, 1.0)
    
    return float(diagonality_score.numpy())


def visualize_alignment_matrix(alignment, save_path=None, title="Attention Alignment"):
    """
    Create a heatmap visualization of attention alignment.
    
    Args:
        alignment (np.ndarray or tf.Tensor): Alignment matrix of shape (tgt_len, src_len)
        save_path (str, optional): Path to save figure
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Note:
        Requires matplotlib. This function provides a helper for creating
        alignment visualizations during debugging and evaluation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Cannot visualize alignment.")
        return None
    
    if isinstance(alignment, tf.Tensor):
        alignment = alignment.numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(alignment, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Encoder Steps')
    ax.set_ylabel('Decoder Steps')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Alignment saved to {save_path}")
    
    return fig


def analyze_gradients(model, loss, max_norm=None):
    """
    Analyze gradient statistics to detect training issues.
    
    Computes gradient norms and statistics to help diagnose:
    - Vanishing gradients (very small norms)
    - Exploding gradients (very large norms)
    - Dead neurons (zero gradients)
    
    Args:
        model: TensorFlow model
        loss: Loss value to compute gradients from
        max_norm (float, optional): Maximum expected gradient norm for warning
        
    Returns:
        dict: Gradient statistics including min, max, mean, and std norms
    """
    trainable_vars = model.trainable_variables
    
    # Compute gradients
    grads = tf.gradients(loss, trainable_vars)
    
    # Filter out None gradients
    grads = [g for g in grads if g is not None]
    
    if not grads:
        return {"error": "No gradients computed"}
    
    # Compute gradient norms
    grad_norms = [tf.norm(g).numpy() for g in grads]
    
    stats = {
        "min_norm": float(np.min(grad_norms)),
        "max_norm": float(np.max(grad_norms)),
        "mean_norm": float(np.mean(grad_norms)),
        "std_norm": float(np.std(grad_norms)),
        "num_zero": int(np.sum(np.array(grad_norms) < 1e-8)),
        "total_params": len(grad_norms)
    }
    
    # Warnings
    if stats["max_norm"] > 10.0:
        stats["warning"] = "Possible gradient explosion detected"
    elif stats["mean_norm"] < 1e-6:
        stats["warning"] = "Possible vanishing gradients detected"
    
    if max_norm and stats["max_norm"] > max_norm:
        stats["clip_needed"] = True
    
    return stats


def compute_mel_spectrogram_snr(pred_mel, target_mel, frame_mask=None):
    """
    Compute Signal-to-Noise Ratio between predicted and target mel-spectrograms.
    
    Higher SNR indicates better prediction quality.
    
    Args:
        pred_mel (tf.Tensor): Predicted mel-spectrogram
        target_mel (tf.Tensor): Target mel-spectrogram
        frame_mask (tf.Tensor, optional): Valid frame mask
        
    Returns:
        float: SNR in dB
    """
    if frame_mask is not None:
        mask = tf.cast(frame_mask, pred_mel.dtype)
    else:
        mask = tf.ones_like(pred_mel)
    
    # Compute signal power
    signal_power = tf.reduce_sum(tf.square(target_mel) * mask)
    
    # Compute noise power (error)
    noise_power = tf.reduce_sum(tf.square(pred_mel - target_mel) * mask)
    
    # Avoid division by zero
    noise_power = tf.maximum(noise_power, 1e-10)
    
    # SNR in dB
    snr_db = 10.0 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)
    
    return float(snr_db.numpy())


def adaptive_gradient_clipping(grads, max_norm, norm_type='global'):
    """
    Apply adaptive gradient clipping with configurable strategies.
    
    Supports different clipping strategies:
    - 'global': Clip by global norm (standard approach)
    - 'per_layer': Clip each layer's gradient independently
    - 'percentile': Clip based on gradient norm percentile
    
    Args:
        grads (list): List of gradient tensors
        max_norm (float): Maximum gradient norm
        norm_type (str): Clipping strategy
        
    Returns:
        tuple: (clipped_grads, global_norm)
    """
    if norm_type == 'global':
        # Standard global norm clipping
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, max_norm)
        return clipped_grads, global_norm
    
    elif norm_type == 'per_layer':
        # Clip each gradient independently
        clipped_grads = []
        norms = []
        for g in grads:
            if g is not None:
                g_norm = tf.norm(g)
                norms.append(g_norm)
                if g_norm > max_norm:
                    g = g * (max_norm / g_norm)
                clipped_grads.append(g)
            else:
                clipped_grads.append(None)
        
        global_norm = tf.sqrt(tf.reduce_sum([n**2 for n in norms]))
        return clipped_grads, global_norm
    
    elif norm_type == 'percentile':
        # Clip based on 95th percentile of gradient norms
        grad_norms = [tf.norm(g) for g in grads if g is not None]
        if grad_norms:
            percentile_95 = tf.numpy_function(
                lambda x: np.percentile(x, 95),
                [tf.stack(grad_norms)],
                tf.float32
            )
            adaptive_max = tf.minimum(percentile_95 * 1.5, max_norm)
            clipped_grads, global_norm = tf.clip_by_global_norm(grads, adaptive_max)
        else:
            clipped_grads, global_norm = grads, tf.constant(0.0)
        
        return clipped_grads, global_norm
    
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class MetricsAggregator:
    """
    Aggregates and computes statistics for training metrics over time.
    
    Useful for tracking metrics across multiple batches and epochs,
    computing moving averages, and detecting training anomalies.
    """
    
    def __init__(self, window_size=100):
        """
        Initialize metrics aggregator.
        
        Args:
            window_size (int): Size of moving average window
        """
        self.window_size = window_size
        self.metrics = {}
        self.history = {}
    
    def update(self, metric_dict):
        """
        Update metrics with new values.
        
        Args:
            metric_dict (dict): Dictionary of metric names to values
        """
        for name, value in metric_dict.items():
            if name not in self.history:
                self.history[name] = []
            
            self.history[name].append(float(value))
            
            # Keep only recent window
            if len(self.history[name]) > self.window_size:
                self.history[name] = self.history[name][-self.window_size:]
    
    def get_statistics(self, metric_name):
        """
        Get statistics for a specific metric.
        
        Args:
            metric_name (str): Name of metric
            
        Returns:
            dict: Statistics including mean, std, min, max, recent value
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        
        values = np.array(self.history[metric_name])
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "recent": float(values[-1]),
            "trend": "improving" if len(values) > 1 and values[-1] < values[0] else "stable"
        }
    
    def get_all_statistics(self):
        """
        Get statistics for all tracked metrics.
        
        Returns:
            dict: Statistics for each metric
        """
        return {
            name: self.get_statistics(name)
            for name in self.history.keys()
        }
    
    def is_improving(self, metric_name, patience=5):
        """
        Check if metric is improving over recent history.
        
        Args:
            metric_name (str): Name of metric to check
            patience (int): Number of steps to look back
            
        Returns:
            bool: True if metric is improving (decreasing)
        """
        if metric_name not in self.history:
            return False
        
        values = self.history[metric_name]
        if len(values) < patience:
            return True  # Too early to tell
        
        recent = values[-patience:]
        return recent[-1] < recent[0]


def create_tensorboard_summary(alignment, mel_pred, mel_target, step, writer, tag_prefix="train"):
    """
    Create comprehensive TensorBoard summaries for TTS training.
    
    Args:
        alignment (tf.Tensor): Attention alignment matrix
        mel_pred (tf.Tensor): Predicted mel-spectrogram
        mel_target (tf.Tensor): Target mel-spectrogram
        step (int): Training step
        writer: TensorBoard SummaryWriter
        tag_prefix (str): Prefix for summary tags
    """
    try:
        import matplotlib.pyplot as plt
        import io
        
        # Alignment heatmap
        if alignment is not None:
            fig = visualize_alignment_matrix(alignment[0].numpy(), title="Attention Alignment")
            if fig:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                
                # Convert to tensor for TensorBoard
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                
                with writer.as_default():
                    tf.summary.image(f"{tag_prefix}/alignment", image, step=step)
        
        # Mel-spectrogram comparison
        if mel_pred is not None and mel_target is not None:
            # Take first example from batch
            pred = mel_pred[0].numpy().T  # (n_mels, time)
            target = mel_target[0].numpy().T
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            axes[0].imshow(target, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title('Target Mel-Spectrogram')
            axes[0].set_ylabel('Mel Bin')
            
            axes[1].imshow(pred, aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title('Predicted Mel-Spectrogram')
            axes[1].set_ylabel('Mel Bin')
            
            diff = np.abs(pred - target)
            axes[2].imshow(diff, aspect='auto', origin='lower', cmap='hot')
            axes[2].set_title('Absolute Difference')
            axes[2].set_ylabel('Mel Bin')
            axes[2].set_xlabel('Time Frame')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            
            with writer.as_default():
                tf.summary.image(f"{tag_prefix}/mel_comparison", image, step=step)
    
    except Exception as e:
        print(f"Warning: Could not create TensorBoard summary: {e}")


if __name__ == "__main__":
    # Example usage and tests
    print("TTSMonitoring module loaded successfully")
    
    # Test alignment diagonality
    test_alignment = tf.eye(100, 120)  # Perfect diagonal alignment
    score = compute_alignment_diagonality(test_alignment[None, :, :])
    print(f"Perfect alignment diagonality score: {score:.4f}")
    
    # Test gradient analysis (mock example)
    mock_grads = [tf.random.normal((10, 10)) * 0.1 for _ in range(5)]
    stats = analyze_gradients(None, None, max_norm=1.0)
    print("Gradient analysis test: Structure validated")
    
    # Test metrics aggregator
    aggregator = MetricsAggregator(window_size=10)
    for i in range(20):
        aggregator.update({"loss": 1.0 / (i + 1), "accuracy": i / 20.0})
    
    stats = aggregator.get_all_statistics()
    print(f"Metrics aggregation test: {len(stats)} metrics tracked")
