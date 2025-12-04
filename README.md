# Speech Emotion Recognition (SER)

A comprehensive speech emotion recognition system featuring **state-of-the-art transformer models**, traditional machine learning approaches, and deep learning architectures. Supports multilingual emotion detection with English (RAVDESS, TESS) and Hindi datasets.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Model Types](#model-types)
- [Training](#training)
- [Evaluation & Prediction](#evaluation--prediction)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project implements multiple approaches for recognizing emotions from speech audio:

1. **Transformer-based Models** (Wav2Vec2) - State-of-the-art, ~75-85% accuracy
2. **Deep Learning** (LSTM/GRU) - Neural networks, ~70-75% accuracy
3. **Traditional ML** (SVM, MLP, Random Forest) - Classical approaches, ~60-70% accuracy

### Supported Emotions

**Default (4 emotions)**: `sad`, `neutral`, `happy`, `angry`

**Extended**: `fear`, `disgust`, `surprised`, `sarcastic` (dataset dependent)

### Multilingual Support

- **English**: RAVDESS, TESS datasets
- **Hindi**: Custom Hindi emotion speech dataset
- **Multilingual training**: Combined English + Hindi for better generalization

---

## ✨ Key Features

### 🚀 Transformer Models
- **Wav2Vec2-based** architecture for superior audio understanding
- **Memory-efficient training** optimized for 6GB GPU
- **Mixed precision training** (FP16) for faster computation
- **Gradient checkpointing** to reduce memory usage
- **Direct waveform processing** - no manual feature extraction needed

### 🎓 Multiple Model Types
- **Traditional ML**: SVM, MLP, Random Forest, Gradient Boosting, KNN
- **Deep Learning**: LSTM, GRU with batch normalization and dropout
- **Transformers**: Wav2Vec2, HuBERT (pretrained on millions of hours)

### 📊 Comprehensive Dataset Support
- **RAVDESS**: 672 samples (4 emotions)
- **TESS**: 1,600 samples (4 emotions)
- **Hindi**: 1,599 samples (4 emotions)
- **Total**: 3,871 samples across 3 datasets

### 🛠️ Production-Ready Tools
- Feature caching for fast training iterations
- Hyperparameter tuning with grid search
- Model checkpointing and resume training
- Confusion matrix and detailed metrics
- Easy prediction interface for deployment

---

## 🏗️ Architecture

### Transformer Architecture (Recommended)

```
Input Audio (WAV)
    ↓
Wav2Vec2 Feature Extractor (CNN) → frozen/trainable
    ↓
Transformer Encoder (12 layers) → frozen/trainable
    ↓
Mean Pooling
    ↓
Classification Head (MLP)
    ↓
Emotion Prediction (4 classes)
```

**Key Components:**
- **Feature Extractor**: 7-layer CNN (can be frozen to save memory)
- **Transformer Encoder**: 12 attention layers, 768 hidden dimensions
- **Classification Head**: 2-layer MLP with dropout
- **Parameters**: ~95M total, ~1.5M trainable (with frozen extractor)

### Traditional ML Pipeline

```
Audio (WAV)
    ↓
Feature Extraction (MFCC, Chroma, Mel Spectrogram)
    ↓
Feature Vector (180 dimensions)
    ↓
ML Classifier (SVM/MLP/RF)
    ↓
Emotion Prediction
```

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended for transformers)
- 6GB+ GPU memory for transformer training

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### Step 2: Install Dependencies

**With GPU support (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**With GPU support (CUDA 12.1+):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Check GPU availability
python check_gpu.py

# Quick system test
python run_test.py
```

---

## 🚀 Quick Start

### Train Transformer Model (Recommended)

```bash
# Quick test (3 epochs, ~10 minutes)
python train_transformer.py --epochs 3 --batch_size 8

# Full training (15 epochs, ~2 hours)
python train_transformer.py --epochs 15 --batch_size 8

# With all datasets including Hindi (20 epochs, ~3 hours)
python train_transformer.py --epochs 20 --batch_size 8
```

### Make Predictions

```bash
# Single file
python demo_transformer.py --mode single --audio_file path/to/audio.wav

# Batch prediction
python demo_transformer.py --mode batch --audio_dir data/ravdess --limit 20

# Model info
python demo_transformer.py --mode info
```

---

## 📂 Datasets

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Source**: 24 professional actors (12 male, 12 female)
- **Emotions**: Neutral, Happy, Sad, Angry, Fear, Disgust, Surprised
- **Samples**: 1,440 total, 672 for 4-emotion classification
- **Quality**: High-quality studio recordings
- **Format**: 16-bit, 48kHz WAV

**Filename Format**: `03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`

### TESS (Toronto Emotional Speech Set)

- **Source**: 2 female actors (young and old)
- **Emotions**: Neutral, Happy, Sad, Angry, Fear, Disgust, Surprised
- **Samples**: 2,800 total, 1,600 for 4-emotion classification
- **Quality**: Clear, consistent recordings
- **Format**: 16-bit, 24kHz WAV

**Filename Format**: `{actor}_{word}_{emotion}.wav`

### Hindi Emotion Dataset

- **Source**: Multiple Hindi speakers
- **Emotions**: Neutral, Happy, Sad, Angry, Fear, Disgust, Sarcastic, Surprised
- **Samples**: ~1,600 samples, 1,599 for 4-emotion classification
- **Language**: Hindi (Devanagari script speakers)
- **Format**: Various sample rates, converted to 16kHz

### Dataset Statistics (4 Emotions)

| Dataset | Sad | Neutral | Happy | Angry | Total | Split |
|---------|-----|---------|-------|-------|-------|-------|
| **RAVDESS** | 192 | 96 | 192 | 192 | 672 | 80/20 |
| **TESS** | 400 | 400 | 400 | 400 | 1,600 | 80/20 |
| **Hindi** | 400 | 400 | 399 | 400 | 1,599 | 80/20 |
| **Combined** | 992 | 896 | 991 | 992 | 3,871 | 80/20 |

**Train/Test Split**: 80% training (3,094 samples) / 20% testing (777 samples)

---

## 🤖 Model Types

### 1. Transformer Models (Best Performance)

**Advantages:**
- Highest accuracy (75-85% for 4 emotions)
- Learns features automatically
- Pretrained on millions of hours of speech
- Excellent for multilingual scenarios
- State-of-the-art results

**Disadvantages:**
- Requires GPU for practical training
- Longer training time (1-3 hours)
- Larger model size (~500MB)
- Higher computational requirements

**Use Cases:**
- Production systems requiring high accuracy
- Multilingual emotion detection
- Research and benchmarking
- Real-time applications with GPU

### 2. Deep Learning (LSTM/GRU)

**Advantages:**
- Good accuracy (70-75%)
- Moderate training time (10-30 minutes)
- Works on both CPU and GPU
- Handles temporal dependencies well

**Disadvantages:**
- Manual feature extraction needed
- Moderate model size (~100MB)
- Requires careful hyperparameter tuning

**Use Cases:**
- Medium-scale applications
- Resource-constrained environments
- Quick experimentation

### 3. Traditional ML (SVM, Random Forest, MLP)

**Advantages:**
- Fast training (1-5 minutes)
- Works on CPU
- Small model size (1-10MB)
- Interpretable results
- Easy to deploy

**Disadvantages:**
- Lower accuracy (60-70%)
- Manual feature engineering required
- Limited capacity for complex patterns

**Use Cases:**
- Prototyping and baseline
- Edge devices with limited resources
- Real-time CPU inference
- Educational purposes

---

## 🎓 Training

### Transformer Training

#### Basic Training
```bash
# Default: 4 emotions, all datasets, 15 epochs
python train_transformer.py --epochs 15 --batch_size 8
```

#### English Only
```bash
python train_transformer.py --exclude_hindi --epochs 15 --batch_size 8
```

#### Custom Emotions
```bash
python train_transformer.py \
    --emotions sad neutral happy angry fear disgust \
    --epochs 20 \
    --batch_size 8
```

#### Advanced Options
```bash
python train_transformer.py \
    --epochs 20 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --max_duration 3.0 \
    --freeze_encoder \
    --model_name facebook/wav2vec2-base \
    --model_name_suffix custom_v1
```

**Key Parameters:**
- `--epochs`: Number of training epochs (default: 15)
- `--batch_size`: Batch size, reduce if OOM (default: 8)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--max_duration`: Audio clip duration in seconds (default: 5.0)
- `--freeze_encoder`: Freeze CNN layers to save memory
- `--no_mixed_precision`: Disable FP16 training

### Traditional ML Training

#### Quick Test
```bash
python run_test.py
```

#### With Custom Configuration
```python
from emotion_recognition import EmotionRecognizer

rec = EmotionRecognizer(
    emotions=['sad', 'neutral', 'happy', 'angry'],
    use_ravdess=True,
    use_tess=True,
    use_hindi=False,
    balance=True
)

rec.train()
print(f"Accuracy: {rec.test_score():.2%}")
rec.save_model('models/custom_model.pkl')
```

### Hyperparameter Tuning

```bash
# Grid search for best parameters
python grid_search.py --model MLP --fast

# Full grid search (slower)
python grid_search.py --model SVM

# All models
python grid_search.py
```

---

## 🔮 Evaluation & Prediction

### Evaluate Transformer Model

```python
from transformer_emotion_recognition import TransformerEmotionRecognizer
import pandas as pd

# Load model
rec = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy', 'angry'])
rec.load_model('models/transformer_best.pt')

# Load test data
test_df = pd.concat([
    pd.read_csv('data/csv/test_ravdess_4class.csv'),
    pd.read_csv('data/csv/test_tess_4class.csv')
])

test_paths = test_df['path'].tolist()
test_labels = test_df['emotion'].tolist()

# Prepare data
rec.prepare_data(
    train_paths=test_paths,
    train_labels=test_labels,
    test_paths=test_paths,
    test_labels=test_labels,
    batch_size=8
)

# Evaluate
accuracy, f1, loss = rec.evaluate()
print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = rec.get_confusion_matrix()
print(cm)
```

### Single File Prediction

```python
from transformer_emotion_recognition import TransformerEmotionRecognizer

rec = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy', 'angry'])
rec.load_model('models/transformer_best.pt')

emotion, confidence = rec.predict('path/to/audio.wav')
print(f"Emotion: {emotion} (confidence: {confidence:.1%})")
```

### Batch Prediction

```bash
# Command line
python demo_transformer.py --mode batch --audio_dir data/ravdess --limit 50

# Python API
from glob import glob

audio_files = glob('data/ravdess/**/*.wav', recursive=True)
for audio_file in audio_files[:10]:
    emotion, conf = rec.predict(audio_file)
    print(f"{audio_file}: {emotion} ({conf:.1%})")
```

---

## 📁 Project Structure

```
speech-emotion-recognition/
│
├── data/                           # Dataset storage
│   ├── csv/                        # CSV files mapping audio to labels
│   │   ├── train_ravdess_4class.csv
│   │   ├── test_ravdess_4class.csv
│   │   ├── train_tess_4class.csv
│   │   ├── test_tess_4class.csv
│   │   ├── train_hindi_4class.csv
│   │   └── test_hindi_4class.csv
│   ├── ravdess/                    # RAVDESS audio files
│   ├── tess/                       # TESS audio files
│   └── hindi/                      # Hindi audio files
│
├── models/                         # Saved model checkpoints
│   ├── transformer_best.pt         # Best transformer model
│   └── *.pkl                       # Traditional ML models
│
├── features/                       # Cached audio features
│   └── *.npz                       # Feature cache files
│
├── grid/                           # Grid search results
│   └── *.json                      # Hyperparameter tuning results
│
├── Core Scripts
│   ├── transformer_emotion_recognition.py  # Transformer model
│   ├── emotion_recognition.py              # Traditional ML
│   ├── deep_emotion_recognition.py         # LSTM/GRU models
│   ├── utils.py                            # Feature extraction
│   ├── data_extractor.py                   # Dataset loading
│   └── create_csv.py                       # CSV generation
│
├── Training Scripts
│   ├── train_transformer.py        # Train transformers
│   ├── run_test.py                 # Quick setup test
│   ├── grid_search.py              # Hyperparameter tuning
│   └── parameters.py               # Model configurations
│
├── Utility Scripts
│   ├── demo_transformer.py         # Prediction demo
│   ├── generate_4class_csv.py      # Generate 4-emotion CSVs
│   ├── generate_hindi_csv.py       # Generate Hindi CSVs
│   └── check_gpu.py                # GPU diagnostic tool
│
├── Documentation
│   ├── README.md                   # This file
│   ├── TRANSFORMER_README.md       # Transformer detailed docs
│   ├── 4EMOTION_GUIDE.md           # 4-emotion configuration
│   └── requirements.txt            # Python dependencies
│
└── Debug Scripts
    ├── debug3.py                   # File checker
    ├── debug4.py                   # Utils test
    └── debug5.py                   # Import test
```

---

## 📊 Performance

### Expected Accuracy by Configuration

| Model Type | Emotions | Accuracy | F1 Score | Training Time | Inference |
|------------|----------|----------|----------|---------------|-----------|
| **Transformer** | 3-class | 75-85% | 0.75-0.85 | 1-2 hours | ~100ms |
| **Transformer** | 4-class | 70-80% | 0.70-0.80 | 2-3 hours | ~100ms |
| **Transformer** | 6-class | 65-75% | 0.65-0.75 | 3-4 hours | ~100ms |
| **Deep LSTM** | 4-class | 70-75% | 0.68-0.73 | 10-30 min | ~50ms |
| **Traditional ML** | 4-class | 60-70% | 0.58-0.68 | 1-5 min | ~10ms |

### Confusion Matrix Example (4 Emotions)

```
              Predicted
           sad  neu  hap  ang
Actual sad [85   5    5    5]  85%
       neu [ 8  82    8    2]  82%
       hap [ 5   5   85    5]  85%
       ang [ 3   2    5   90]  90%
```

### Performance Factors

**Positive Impact:**
- More training data
- Balanced dataset
- Longer training (more epochs)
- Multilingual training
- Pretrained models

**Negative Impact:**
- More emotion classes
- Imbalanced data
- Short audio clips
- Background noise
- Low audio quality

---

## 🔧 Advanced Usage

### Custom Dataset Integration

```python
from create_csv import write_custom_csv

# Generate CSV for custom dataset
write_custom_csv(
    data_path='path/to/custom/dataset',
    emotions=['sad', 'happy', 'angry'],
    train_name='data/csv/train_custom.csv',
    test_name='data/csv/test_custom.csv',
    train_size=0.8,
    verbose=1
)

# Train with custom dataset
python train_transformer.py \
    --train_csv data/csv/train_custom.csv \
    --test_csv data/csv/test_custom.csv \
    --emotions sad happy angry
```

### REST API Deployment

```python
from flask import Flask, request, jsonify
from transformer_emotion_recognition import TransformerEmotionRecognizer

app = Flask(__name__)
model = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy', 'angry'])
model.load_model('models/transformer_best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    temp_path = '/tmp/temp_audio.wav'
    audio_file.save(temp_path)

    emotion, confidence = model.predict(temp_path)

    return jsonify({
        'emotion': emotion,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Transfer Learning

```python
# Load pretrained model
rec = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy', 'angry'])
rec.load_model('models/transformer_best.pt')

# Fine-tune on new data
rec.prepare_data(new_train_paths, new_train_labels, ...)
rec.train(epochs=5, learning_rate=1e-5)  # Lower LR for fine-tuning
```

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check GPU
python check_gpu.py

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM)

```bash
# Reduce batch size
python train_transformer.py --batch_size 4

# Reduce audio duration
python train_transformer.py --max_duration 3.0

# Freeze encoder
python train_transformer.py --freeze_encoder
```

### Low Accuracy

1. Train longer: `--epochs 30`
2. Add more data or use data augmentation
3. Try different learning rates: `--learning_rate 5e-5`
4. Use multilingual training (included by default)
5. Ensure balanced dataset

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

### Development Setup

```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```

### Areas for Contribution

- Additional dataset support
- More emotion categories
- Real-time processing optimizations
- Web interface
- Mobile deployment
- Data augmentation techniques
- New model architectures

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{speech_emotion_recognition_2024,
  author = {Shirssack},
  title = {Speech Emotion Recognition with Transformers},
  year = {2024},
  url = {https://github.com/yourusername/speech-emotion-recognition}
}
```

### Referenced Models

**Wav2Vec2:**
```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```

### Datasets

- **RAVDESS**: [Livingstone & Russo, 2018](https://zenodo.org/record/1188976)
- **TESS**: [Pichora-Fuller & Dupuis, 2020](https://doi.org/10.5683/SP2/E8H2MF)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Hugging Face Transformers team for pretrained models
- RAVDESS and TESS dataset creators
- PyTorch and scikit-learn communities
- All contributors and users

---

## 📞 Contact

- **Author**: Shirssack
- **Email**: [your-email@example.com]
- **GitHub**: [https://github.com/yourusername](https://github.com/yourusername)
- **Issues**: [GitHub Issues](https://github.com/yourusername/speech-emotion-recognition/issues)

---

## 🗺️ Roadmap

### Upcoming Features

- [ ] Real-time emotion detection from microphone
- [ ] Web-based demo interface
- [ ] Mobile app deployment (Android/iOS)
- [ ] Additional language support (Spanish, French, Chinese)
- [ ] Emotion intensity prediction (not just category)
- [ ] Multi-speaker scenarios
- [ ] Background noise robustness improvements
- [ ] Model quantization for edge deployment
- [ ] Docker container for easy deployment
- [ ] Continuous emotion tracking over time

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by Shirssack
