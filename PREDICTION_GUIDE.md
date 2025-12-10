# Prediction Guide - Using Trained Models

This guide explains how to use the trained models to predict emotions from audio files.

## 📁 Prediction Scripts

### 1. **predict_emotion.py** - Single File Prediction
Predict emotion from a single audio file.

**Usage:**
```bash
python predict_emotion.py <audio_file> [model_type]
```

**Examples:**
```bash
# Using default model (Traditional ML)
python predict_emotion.py data/ravdess/Actor_01/03-01-01-01-01-01-01.wav

# Using specific models
python predict_emotion.py myaudio.wav traditional
python predict_emotion.py myaudio.wav deep_learning
python predict_emotion.py myaudio.wav transformer
```

**Output:**
```
======================================================================
RESULT
======================================================================
Predicted Emotion: HAPPY
Confidence: 87.45%

All Probabilities:
    happy:  87.45% ██████████████████████████
      sad:   7.23% ██
  neutral:   3.12% █
    angry:   2.20%
======================================================================
```

---

### 2. **predict_batch.py** - Batch Prediction
Process multiple audio files at once.

**Usage:**
```bash
python predict_batch.py <audio_dir> [model_type] [--limit N]
```

**Examples:**
```bash
# Process all files in directory
python predict_batch.py data/ravdess/Actor_01

# Process first 10 files
python predict_batch.py data/ravdess/Actor_01 traditional --limit 10

# Use deep learning model
python predict_batch.py data/ravdess/Actor_01 deep_learning
```

**Output:**
```
Processing files...
----------------------------------------------------------------------
[  1/10] 03-01-01-01-01-01-01.wav -> neutral  (82.3%)
[  2/10] 03-01-02-01-01-01-01.wav -> sad      (91.5%)
[  3/10] 03-01-03-01-01-01-01.wav -> happy    (87.2%)
...

======================================================================
SUMMARY
======================================================================
Total files processed: 10
Processing time: 2.45 seconds
Average time per file: 0.245 seconds

Emotion Distribution:
  neutral:   4 (40.0%) ████████████████████
    happy:   3 (30.0%) ███████████████
      sad:   2 (20.0%) ██████████
    angry:   1 (10.0%) █████

Average Confidence: 85.32%
======================================================================

Results saved to: predictions_traditional_1702384567.csv
```

---

### 3. **predict_interactive.py** - Interactive Mode
User-friendly menu-driven interface for predictions.

**Usage:**
```bash
python predict_interactive.py
```

**Features:**
- Menu-driven interface
- Browse and select from sample files
- Switch between model types
- Real-time predictions with visual feedback

**Example Session:**
```
======================================================================
                    EMOTION PREDICTION TOOL
======================================================================

Select model type:
  1. Traditional ML (MLP) - Fast
  2. Deep Learning (LSTM) - Moderate
  3. Transformer (Wav2Vec2) - Best accuracy

Enter choice (1-3, default: 1): 1

Using: traditional model

----------------------------------------------------------------------
Options:
  1. Predict emotion from file path
  2. Select from sample files
  3. Change model type
  4. Exit

Enter choice (1-4): 2

Sample audio files from data/ravdess/Actor_01:
   1. 03-01-01-01-01-01-01.wav
   2. 03-01-02-01-01-01-01.wav
   ...

Select file (1-15, or 0 to cancel): 1

Predicting...

======================================================================
RESULT
======================================================================
File: 03-01-01-01-01-01-01.wav
Predicted Emotion: NEUTRAL
Confidence: 82.34%

All Probabilities:
  neutral:  82.34% ████████████████████████
      sad:   9.12% ██
    happy:   5.45% █
    angry:   3.09%
======================================================================
```

---

## 🎯 Model Types Comparison

| Model Type | Speed | Accuracy | GPU Required | Use Case |
|------------|-------|----------|--------------|----------|
| **traditional** | ⚡ Very Fast (0.1s) | 65-75% | No | Quick testing, CPU-only |
| **deep_learning** | ⚡ Fast (0.3s) | 65-75% | Optional | Balanced speed/accuracy |
| **transformer** | 🐌 Slow (1-3s) | 75-85% | Yes | Best accuracy, production |

---

## 📊 Understanding the Output

### Predicted Emotion
The model's top prediction from: `sad`, `neutral`, `happy`, `angry`

### Confidence Score
How confident the model is in its prediction (0-100%)
- **> 80%**: High confidence - Reliable prediction
- **60-80%**: Moderate confidence - Reasonable prediction
- **< 60%**: Low confidence - Uncertain prediction

### Probability Distribution
Shows confidence for all emotions. The top emotion is the prediction.

---

## 🎤 Supported Audio Formats

- **Format**: WAV files (.wav)
- **Sample Rate**: Any (automatically resampled to 16kHz for Transformer)
- **Channels**: Mono or Stereo (converted to mono automatically)
- **Duration**: Any length (shorter clips may be less accurate)

---

## 💡 Tips for Best Results

### 1. Audio Quality
- Use clear audio without background noise
- Avoid heavily compressed or distorted audio
- Ensure speech is audible and clear

### 2. Model Selection
- **Quick testing**: Use `traditional` model
- **Production use**: Use `transformer` model
- **Batch processing**: Use `deep_learning` for balance

### 3. Confidence Interpretation
- High confidence (>80%): Trust the prediction
- Medium confidence (60-80%): Consider context
- Low confidence (<60%): Manual verification recommended

---

## 🔧 Programmatic Usage

### Python Script Example

```python
from utils import extract_feature
import pickle

# Load model
with open('models/mlp_4emotions.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    int_to_emotion = data['int_to_emotion']

# Predict
audio_file = 'myaudio.wav'
features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
features_scaled = scaler.transform([features])
emotion_idx = model.predict(features_scaled)[0]
emotion = int_to_emotion[emotion_idx]

print(f"Emotion: {emotion}")
```

### Integration Example

```python
def analyze_audio(file_path):
    """Simple emotion analysis function"""
    from utils import extract_feature
    import pickle

    # Load model once (cache it in production)
    with open('models/mlp_4emotions.pkl', 'rb') as f:
        data = pickle.load(f)

    # Extract and predict
    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    features_scaled = data['scaler'].transform([features])
    emotion_idx = data['model'].predict(features_scaled)[0]
    emotion = data['int_to_emotion'][emotion_idx]

    # Get confidence
    if hasattr(data['model'], 'predict_proba'):
        proba = data['model'].predict_proba(features_scaled)[0]
        confidence = proba[emotion_idx]
        return emotion, confidence

    return emotion, None

# Use in your application
emotion, conf = analyze_audio('user_recording.wav')
print(f"User is feeling: {emotion} (confidence: {conf:.1%})")
```

---

## 🚨 Troubleshooting

### Error: Model file not found
**Solution**: Train the model first using the training scripts.
```bash
python train_traditional.py  # For traditional ML
python train_deep_learning.py  # For deep learning
python train_transformer.py --epochs 15  # For transformer
```

### Error: No module named 'tensorflow'
**Solution**: Install required dependencies.
```bash
pip install tensorflow
```

### Error: File format not supported
**Solution**: Ensure audio is in WAV format. Convert if needed:
```bash
ffmpeg -i input.mp3 output.wav
```

### Low accuracy on custom audio
**Possible causes**:
- Audio quality issues (noise, compression)
- Different recording conditions than training data
- Speaker characteristics very different from training data
- Emotion expression style different from datasets

**Solutions**:
- Use cleaner audio recordings
- Retrain model with your own data
- Use the transformer model for better generalization

---

## 📈 Batch Processing Best Practices

### For Large Datasets (1000+ files)

```python
# Use batch prediction script with logging
python predict_batch.py audio_directory traditional > results.log
```

### CSV Output Format

The batch script saves results in CSV format:
```csv
file,emotion,confidence
audio1.wav,happy,0.87
audio2.wav,sad,0.92
audio3.wav,neutral,0.78
...
```

You can load this in pandas for analysis:
```python
import pandas as pd

df = pd.read_csv('predictions_traditional_*.csv')
print(df['emotion'].value_counts())
print(f"Average confidence: {df['confidence'].mean():.2%}")
```

---

## 🎓 Next Steps

1. **Try the interactive tool** to get familiar with predictions
2. **Test on your own audio** to see model performance
3. **Compare different models** to find the best for your use case
4. **Integrate into your application** using the provided examples

For training new models or retraining with custom data, see `TRAINING_GUIDE.md`.
