# Training Samples Directory

This directory contains sample generations from the TTS model during training to monitor progress.

## File Structure
- `sample_{i}_step_{step}.npz`: Compressed numpy arrays containing mel-spectrograms and stop probabilities
- `progress_step_{step}.png`: Visual comparison plots showing training progress

## Sample Data Format
Each `.npz` file contains:
- `mel`: Generated mel-spectrogram (shape: [1, time, n_mels])
- `stop_probs`: Stop token probabilities (shape: [1, time, 1])
- `metadata`: JSON string with generation parameters

## Usage
```python
import numpy as np
data = np.load('sample_0_step_200.npz')
mel = data['mel'][0]  # Remove batch dimension
stop_probs = data['stop_probs'][0]
```

## Monitoring Training Progress
- Check the spectrograms in the PNG files to see if the model is learning
- Look for more structured patterns as training progresses
- Audio quality should improve over time (can be synthesized using the inference notebook)