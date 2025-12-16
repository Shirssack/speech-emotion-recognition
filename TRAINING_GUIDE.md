# Training Guide - All Model Types

This guide shows you how to train and use all three types of models available in this repository.

## Overview of Model Types

| Model Type | Script | Training Time | GPU Required | Accuracy | Best For |
|------------|--------|---------------|--------------|----------|----------|
| **Traditional ML** | `train_traditional.py` | 1-5 minutes | No | 60-70% | Quick experiments, CPU-only |
| **Deep Learning** | `train_deep_learning.py` | 10-30 minutes | Optional (faster with GPU) | 65-75% | Balanced speed/accuracy |
| **Transformer** | `train_transformer.py` | 1-3 hours | Yes (6GB+) | 75-85% | Best accuracy, production |

---

## 1. Traditional ML Models (SVM, MLP, Random Forest)

### Quick Start

```bash
python train_traditional.py
```

This trains an MLP (Multi-Layer Perceptron) classifier with default settings.

### Customization

Edit the configuration section in `train_traditional.py`:

```python
# Configuration
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'MLP'  # Options: MLP, SVM, RandomForest, GradientBoosting, KNN
```

### Available Models

- **MLP**: Multi-Layer Perceptron (default, recommended)
- **SVM**: Support Vector Machine (good for small datasets)
- **RandomForest**: Random Forest Classifier (robust, interpretable)
- **GradientBoosting**: Gradient Boosting Classifier (high accuracy)
- **KNN**: K-Nearest Neighbors (simple, baseline)

### Example: Train SVM

```python
# In train_traditional.py, change:
MODEL_TYPE = 'SVM'
```

Then run:
```bash
python train_traditional.py
```

### Making Predictions

```python
from emotion_recognition import EmotionRecognizer

# Load model
rec = EmotionRecognizer(
    model='MLP',
    emotions=['sad', 'neutral', 'happy', 'angry']
)
rec.load_model('models/mlp_4emotions.pkl')

# Predict on audio file
emotion = rec.predict('path/to/audio.wav')
print(f"Predicted emotion: {emotion}")

# Predict with probability
emotion_probs = rec.predict_proba('path/to/audio.wav')
for emotion, prob in emotion_probs.items():
    print(f"{emotion}: {prob:.2%}")
```

### Expected Performance

- **Training Time**: 1-5 minutes (CPU)
- **Test Accuracy**: 60-70%
- **Model Size**: 1-10 MB
- **Inference Speed**: < 0.1 seconds per file

---

## 2. Deep Learning Models (LSTM, GRU)

### Quick Start

```bash
python train_deep_learning.py
```

This trains an LSTM network with default settings.

### Customization

Edit the configuration section in `train_deep_learning.py`:

```python
# Configuration
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'LSTM'  # Options: LSTM, GRU

# Model hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.2
HIDDEN_SIZE = 128
NUM_LAYERS = 2
```

### Available Models

- **LSTM**: Long Short-Term Memory (default, better for longer sequences)
- **GRU**: Gated Recurrent Unit (faster training, similar performance)

### Example: Train GRU

```python
# In train_deep_learning.py, change:
MODEL_TYPE = 'GRU'
EPOCHS = 150  # Train longer for better results
```

Then run:
```bash
python train_deep_learning.py
```

### GPU Acceleration

Deep learning models train much faster with GPU:

```bash
# Check GPU availability
python check_gpu.py

# If GPU is available, training will automatically use it
python train_deep_learning.py
```

### Making Predictions

```python
from deep_emotion_recognition import DeepEmotionRecognizer

# Load model
rec = DeepEmotionRecognizer(
    emotions=['sad', 'neutral', 'happy', 'angry']
)
rec.load('models/lstm_4emotions.h5')

# Predict on audio file
emotion = rec.predict('path/to/audio.wav')
print(f"Predicted emotion: {emotion}")

# Predict with confidence
emotion, confidence = rec.predict_proba('path/to/audio.wav')
print(f"Predicted: {emotion} ({confidence:.2%})")
```

### Expected Performance

- **Training Time**: 10-30 minutes (GPU), 1-3 hours (CPU)
- **Test Accuracy**: 65-75%
- **Model Size**: 10-50 MB
- **Inference Speed**: 0.1-0.5 seconds per file

---

## 3. Transformer Models (Wav2Vec2)

### Quick Start

```bash
# Quick test (3 epochs)
python train_transformer.py --epochs 3 --batch_size 8

# Full training (15 epochs, recommended)
python train_transformer.py --epochs 15 --batch_size 8
```

### Customization

Use command-line arguments:

```bash
# Train with all datasets (default)
python train_transformer.py --epochs 15 --batch_size 8

# Train with English only (exclude Hindi)
python train_transformer.py --epochs 15 --batch_size 8 --exclude_hindi

# Train with 3 emotions
python train_transformer.py --epochs 15 --emotions sad neutral happy

# Train with frozen encoder (recommended for 6GB GPU)
python train_transformer.py --epochs 15 --freeze_encoder

# Reduce memory usage if OOM
python train_transformer.py --epochs 15 --batch_size 4 --max_duration 3.0
```

### Memory Optimization

For 6GB GPU:

```bash
python train_transformer.py \
    --batch_size 8 \
    --max_duration 5.0 \
    --freeze_encoder \
    --epochs 15
```

If Out of Memory:

```bash
python train_transformer.py \
    --batch_size 4 \
    --max_duration 3.0 \
    --freeze_encoder \
    --epochs 15
```

### Making Predictions

```python
from transformer_emotion_recognition import TransformerEmotionRecognizer

# Load model
rec = TransformerEmotionRecognizer(
    emotions=['sad', 'neutral', 'happy', 'angry']
)
rec.load_model('models/transformer_best.pt')

# Predict on audio file
emotion, confidence = rec.predict('path/to/audio.wav')
print(f"Predicted: {emotion} ({confidence:.2%})")
```

### Using Demo Script

```bash
# Show model info
python demo_transformer.py --mode info

# Single file prediction
python demo_transformer.py --mode single --audio_file path/to/audio.wav

# Batch prediction (multiple files)
python demo_transformer.py --mode batch --audio_dir data/ravdess --limit 20
```

### Expected Performance

- **Training Time**: 1-3 hours (6GB GPU)
- **Test Accuracy**: 75-85%
- **Model Size**: 300-400 MB
- **Inference Speed**: 0.5-1.5 seconds per file (GPU), 5-15 seconds (CPU)

---

## Quick Comparison

### When to Use Each Model Type

**Use Traditional ML when:**
- You need quick results (minutes, not hours)
- You don't have a GPU
- You want interpretable models
- You have limited data
- You need small model files

**Use Deep Learning when:**
- You want better accuracy than traditional ML
- You have some GPU resources (optional)
- Training time of 30 minutes is acceptable
- You need moderate model size

**Use Transformer when:**
- You need the best possible accuracy
- You have a GPU (6GB+)
- You can wait 1-3 hours for training
- You want state-of-the-art performance
- You need multilingual support
- You want transfer learning capabilities

---

## Dataset Configuration

All three training scripts use the same datasets by default:

```python
# Default configuration (4 emotions, 3 datasets)
train_csv=['data/csv/train_ravdess_4class.csv',
           'data/csv/train_tess_4class.csv',
           'data/csv/train_hindi_4class.csv']
test_csv=['data/csv/test_ravdess_4class.csv',
          'data/csv/test_tess_4class.csv',
          'data/csv/test_hindi_4class.csv']
emotions=['sad', 'neutral', 'happy', 'angry']
```

### Dataset Statistics (4-Emotion)

| Dataset | Train | Test | Total |
|---------|-------|------|-------|
| RAVDESS | 535 | 137 | 672 |
| TESS | 1,280 | 320 | 1,600 |
| Hindi | 1,279 | 320 | 1,599 |
| **TOTAL** | **3,094** | **777** | **3,871** |

---

## Model Comparison on 4-Emotion Task

| Model | Accuracy | F1-Score | Training Time | GPU Required | Model Size |
|-------|----------|----------|---------------|--------------|------------|
| SVM | 58-65% | 0.58-0.65 | 2 min | No | 5 MB |
| MLP | 60-70% | 0.60-0.70 | 3 min | No | 2 MB |
| Random Forest | 62-68% | 0.62-0.68 | 4 min | No | 50 MB |
| LSTM | 65-72% | 0.65-0.72 | 20 min | Optional | 25 MB |
| GRU | 66-73% | 0.66-0.73 | 18 min | Optional | 20 MB |
| Transformer | 75-85% | 0.75-0.85 | 90 min | Yes (6GB+) | 350 MB |

*Results based on multilingual training (RAVDESS + TESS + Hindi)*

---

## Troubleshooting

### Traditional ML Models

**Problem**: Low accuracy (< 50%)
```
Solution:
1. Check if CSV files are correct
2. Ensure emotions match dataset
3. Try different model types (MLP usually best)
```

**Problem**: Training very slow
```
Solution:
1. Reduce training data size
2. Use simpler model (KNN)
3. Disable verbose output
```

### Deep Learning Models

**Problem**: Training loss not decreasing
```
Solution:
1. Increase learning rate: LEARNING_RATE = 0.01
2. Reduce dropout: DROPOUT = 0.1
3. Train longer: EPOCHS = 200
```

**Problem**: Overfitting (train acc >> test acc)
```
Solution:
1. Increase dropout: DROPOUT = 0.3
2. Reduce model size: HIDDEN_SIZE = 64
3. Add more training data
```

### Transformer Models

**Problem**: CUDA Out of Memory
```
Solution:
1. Reduce batch size: --batch_size 4
2. Reduce audio duration: --max_duration 3.0
3. Freeze encoder: --freeze_encoder
```

**Problem**: Training too slow
```
Solution:
1. Increase batch size: --batch_size 16
2. Reduce epochs: --epochs 10
3. Use smaller dataset
```

---

## Model Files and Locations

After training, models are saved to the `models/` directory:

```
models/
├── mlp_4emotions.pkl              # Traditional ML model
├── svm_4emotions.pkl              # Traditional ML model
├── randomforest_4emotions.pkl     # Traditional ML model
├── lstm_4emotions.h5              # Deep Learning model
├── gru_4emotions.h5               # Deep Learning model
├── transformer_best.pt            # Transformer model (best)
└── transformer_last.pt            # Transformer model (last epoch)
```

---

## Advanced Usage

### Hyperparameter Tuning (Traditional ML)

```bash
python grid_search.py --model MLP --fast
```

See `GRID_SEARCH.md` for details.

### Custom Emotions

All scripts support custom emotion sets:

**Traditional ML**: Edit `EMOTIONS` in script
**Deep Learning**: Edit `EMOTIONS` in script
**Transformer**: Use `--emotions` flag

```bash
python train_transformer.py --emotions angry happy sad neutral fear disgust
```

### Using Your Own Dataset

1. Create CSV files with format: `path,emotion`
2. Update `train_csv` and `test_csv` in training scripts
3. Ensure audio files are in WAV format
4. Run training as usual

---

## Getting Help

- **Documentation**: See `README.md` for project overview
- **Transformer Details**: See `TRANSFORMER_README.md`
- **4-Emotion Guide**: See `4EMOTION_GUIDE.md`
- **Issues**: Report bugs on GitHub

---

## Quick Reference Commands

```bash
# Check GPU
python check_gpu.py

# Train Traditional ML (fastest)
python train_traditional.py

# Train Deep Learning (balanced)
python train_deep_learning.py

# Train Transformer (best accuracy)
python train_transformer.py --epochs 15

# Make predictions
python demo_transformer.py --mode single --audio_file myaudio.wav

# Batch prediction
python demo_transformer.py --mode batch --audio_dir myaudiodir/ --limit 50
```

---

**Questions?** Check the main `README.md` or open a GitHub issue.
