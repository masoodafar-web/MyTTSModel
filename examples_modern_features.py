"""
Example: Using Modern Architectural Features in MyTTSModel

This script demonstrates how to use the new architectural improvements:
- Rotary Position Embedding (RoPE)
- SwiGLU activation
- Temperature-based sampling
- Monitoring and debugging tools
"""

# ============================================================================
# Example 1: Creating a model with modern features
# ============================================================================

from MyTTSModel import TransformerTTS
from TTSConfig import get_model_preset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    use_fast=False,
    src_lang="eng_Latn"
)

# Option A: Use modern preset (recommended for new models)
print("=== Example 1: Modern preset ===")
preset = get_model_preset("modern_base")
model = TransformerTTS(
    num_layers=preset.num_layers,
    d_model=preset.d_model,
    num_heads=preset.num_heads,
    dff=preset.dff,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    dropout_rate=preset.dropout_rate,
    droppath_rate=preset.droppath_rate,
    activation=preset.activation,  # SwiGLU
    pos_encoding_type=preset.pos_encoding_type,  # RoPE
    pad_id=tokenizer.pad_token_id
)
print(f"✓ Model created with {preset.activation} activation and {preset.pos_encoding_type} position encoding")

# Option B: Custom configuration
print("\n=== Example 2: Custom configuration ===")
model_custom = TransformerTTS(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    dropout_rate=0.1,
    pad_id=tokenizer.pad_token_id,
    activation='swiglu',  # Use SwiGLU instead of GELU
    pos_encoding_type='rope',  # Use RoPE instead of sinusoidal
)
print("✓ Custom model created with SwiGLU and RoPE")


# ============================================================================
# Example 2: Backward compatibility (existing models work without changes)
# ============================================================================

print("\n=== Example 3: Backward compatibility ===")
model_classic = TransformerTTS(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    dropout_rate=0.1,
    pad_id=tokenizer.pad_token_id
    # No activation or pos_encoding_type specified → uses defaults (gelu, sinusoidal)
)
print("✓ Classic model created (backward compatible with existing checkpoints)")


# ============================================================================
# Example 3: Inference with temperature sampling
# ============================================================================

print("\n=== Example 4: Inference with temperature ===")

# Note: This is pseudocode - requires actual trained weights and proper input
"""
import tensorflow as tf
import numpy as np

# Tokenize input
text = "Hello, this is a test of the text to speech system."
enc_ids = tokenizer.encode(text, add_special_tokens=True, src_lang="eng_Latn")
enc_ids = np.array([enc_ids], dtype=np.int32)

# Load trained weights
model.build_for_load(max_src_len=256, max_tgt_len=2000)
model.load_weights("checkpoints/model_weights.h5")

# Generate with standard (greedy) decoding
mel_standard, stop_probs = model.greedy_generate_fast(
    enc_ids,
    max_steps=600,
    min_steps=120,
    temperature=1.0,  # Default: greedy/deterministic
    verbose=True
)

# Generate with higher temperature for more diversity
mel_diverse, stop_probs = model.greedy_generate_fast(
    enc_ids,
    max_steps=600,
    min_steps=120,
    temperature=1.2,  # More variation
    verbose=True
)

print(f"Standard mel shape: {mel_standard.shape}")
print(f"Diverse mel shape: {mel_diverse.shape}")
"""


# ============================================================================
# Example 4: Using monitoring tools
# ============================================================================

print("\n=== Example 5: Monitoring and debugging ===")

# Note: This is pseudocode showing the monitoring API
"""
from TTSMonitoring import (
    extract_alignment_from_model,
    compute_alignment_diagonality,
    visualize_alignment_matrix,
    analyze_gradients,
    adaptive_gradient_clipping,
    MetricsAggregator
)

# During training/evaluation
enc_ids = ...  # Input token IDs
mel_target = ...  # Target mel-spectrogram

# 1. Extract and analyze alignment
alignment = extract_alignment_from_model(model, enc_ids, mel_target)
diag_score = compute_alignment_diagonality(alignment)
print(f"Alignment quality score: {diag_score:.4f}")

if diag_score < 0.5:
    print("⚠️ Poor alignment detected - saving visualization")
    visualize_alignment_matrix(
        alignment[0].numpy(),
        save_path="debug/poor_alignment.png",
        title=f"Alignment (score={diag_score:.4f})"
    )

# 2. Analyze gradients
with tf.GradientTape() as tape:
    loss = compute_loss(model, batch)

grad_stats = analyze_gradients(model, loss, max_norm=1.0)
print(f"Gradient statistics: {grad_stats}")

if "warning" in grad_stats:
    print(f"⚠️ {grad_stats['warning']}")

# 3. Apply adaptive gradient clipping
grads = tape.gradient(loss, model.trainable_variables)
clipped_grads, global_norm = adaptive_gradient_clipping(
    grads,
    max_norm=1.0,
    norm_type='percentile'  # Adaptive based on gradient distribution
)
optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

# 4. Track metrics over time
aggregator = MetricsAggregator(window_size=100)

for step in range(1000):
    # Training step...
    aggregator.update({
        "loss": loss_value,
        "mel_l1": mel_error,
        "stop_acc": stop_accuracy
    })
    
    if step % 100 == 0:
        stats = aggregator.get_all_statistics()
        print(f"\\nStep {step} metrics:")
        for name, stat in stats.items():
            print(f"  {name}: mean={stat['mean']:.4f}, trend={stat['trend']}")
"""


# ============================================================================
# Example 5: Configuration presets comparison
# ============================================================================

print("\n=== Example 6: Available presets ===")
from TTSConfig import PRESETS

print("\nClassical presets (sinusoidal + GELU):")
for name in ['tiny', 'normal', 'large']:
    preset = PRESETS[name]
    print(f"  {name:12s}: {preset.num_layers} layers, {preset.d_model} dim, "
          f"{preset.activation} activation, {preset.pos_encoding_type} encoding")

print("\nModern presets (RoPE + SwiGLU):")
for name in ['modern_small', 'modern_base', 'modern_large']:
    preset = PRESETS[name]
    print(f"  {name:12s}: {preset.num_layers} layers, {preset.d_model} dim, "
          f"{preset.activation} activation, {preset.pos_encoding_type} encoding")


# ============================================================================
# Example 6: Training with new features
# ============================================================================

print("\n=== Example 7: Training loop pseudocode ===")
"""
import tensorflow as tf
from TTSMonitoring import adaptive_gradient_clipping, MetricsAggregator

# Setup
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
aggregator = MetricsAggregator(window_size=100)

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            mel_pre, mel_post, stop = model(batch, training=True)
            
            # Compute loss
            loss = compute_total_loss(mel_pre, mel_post, stop, batch)
        
        # Compute gradients
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Apply adaptive gradient clipping
        clipped_grads, global_norm = adaptive_gradient_clipping(
            grads,
            max_norm=1.0,
            norm_type='percentile'
        )
        
        # Update weights
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
        
        # Track metrics
        aggregator.update({
            "loss": loss.numpy(),
            "grad_norm": global_norm.numpy()
        })
        
        # Log progress
        if step % 100 == 0:
            stats = aggregator.get_all_statistics()
            print(f"Epoch {epoch}, Step {step}: loss={stats['loss']['mean']:.4f}")
"""

print("\n" + "="*70)
print("For complete examples, see:")
print("  - README.md: Full documentation")
print("  - ARCHITECTURE_IMPROVEMENTS_FA.md: Persian guide")
print("  - MyTTSModelTrain.py: Training implementation")
print("="*70)
