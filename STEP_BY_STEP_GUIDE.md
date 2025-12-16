# Step-by-Step Guide: Training and Prediction

## Complete Guide for Adversarial HuBERT Emotion Recognition

This guide will walk you through **every step** from installation to making predictions.

---

## 📋 Table of Contents

1. [Environment Setup](#step-1-environment-setup)
2. [Data Verification](#step-2-data-verification)
3. [Training the Model](#step-3-training-the-model)
4. [Monitoring Training](#step-4-monitoring-training)
5. [Evaluating Cross-Lingual Transfer](#step-5-evaluating-cross-lingual-transfer)
6. [Making Predictions](#step-6-making-predictions)
7. [Troubleshooting](#step-7-troubleshooting)

---

## Step 1: Environment Setup

### 1.1 Check Python Version

```bash
python --version
# Should be Python 3.8 or higher
```

### 1.2 Install Dependencies

```bash
# If you haven't already, install all requirements
pip install -r requirements.txt

# Verify key packages
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import soundfile; print(f'✓ SoundFile installed')"
python -c "import librosa; print(f'✓ Librosa installed')"
```

Expected output:
```
✓ PyTorch 2.0.0+cu118
✓ Transformers 4.30.0
✓ SoundFile installed
✓ Librosa installed
```

### 1.3 Check GPU Availability

```bash
python check_gpu.py
```

Or manually:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output (if GPU available):
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## Step 2: Data Verification

### 2.1 Check Data Directories

```bash
# List all CSV files
ls -lh data/csv/*.csv
```

Expected files:
```
train_ravdess_4class.csv    (536 samples - English)
train_tess_4class.csv       (961 samples - English)
train_hindi_4class.csv      (1280 samples - Hindi)
test_ravdess_4class.csv     (136 samples - English)
test_tess_4class.csv        (321 samples - English)
test_hindi_4class.csv       (321 samples - Hindi)
```

### 2.2 Verify CSV Format

```bash
# Check first few lines of each CSV
head -n 5 data/csv/train_ravdess_4class.csv
head -n 5 data/csv/train_hindi_4class.csv
```

Expected format:
```
path,emotion
data/ravdess/Actor_01/03-01-01-01-01-01-01.wav,neutral
data/ravdess/Actor_01/03-01-02-01-01-01-01.wav,calm
...
```

### 2.3 Check Audio Files Exist

```bash
# Count audio files
find data/ravdess -name "*.wav" | wc -l
find data/tess -name "*.wav" | wc -l
find data/hindi -name "*.wav" | wc -l
```

Expected counts:
- RAVDESS: ~1440 files
- TESS: ~2800 files
- Hindi: ~1599 files

### 2.4 Test Data Loading

```bash
# Quick test that data can be loaded
python -c "
import pandas as pd
import os

csv_files = [
    'data/csv/train_ravdess_4class.csv',
    'data/csv/train_tess_4class.csv',
    'data/csv/train_hindi_4class.csv'
]

for csv_file in csv_files:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f'✓ {csv_file}: {len(df)} samples')
        # Check first file exists
        if os.path.exists(df.iloc[0]['path']):
            print(f'  ✓ Audio files accessible')
        else:
            print(f'  ✗ Audio files NOT found at: {df.iloc[0][\"path\"]}')
    else:
        print(f'✗ {csv_file} NOT FOUND')
"
```

---

## Step 3: Training the Model

### 3.1 Choose Your Configuration

#### Option A: Default (Recommended for First Run)
```bash
python train_adversarial_hubert.py \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --emotion_weight 1.0 \
    --language_weight 0.1 \
    --adversarial_layers 3 6 9 12 \
    --output_dir models/adversarial_hubert \
    --experiment_name my_first_model
```

**When to use**: First time training, 6GB+ GPU, want balanced performance

#### Option B: Memory-Efficient (4-6GB GPU)
```bash
python train_adversarial_hubert.py \
    --epochs 20 \
    --batch_size 4 \
    --adversarial_layers 6 12 \
    --hidden_dim 128 \
    --max_duration 4.0 \
    --gradient_checkpointing \
    --output_dir models/adversarial_hubert \
    --experiment_name memory_efficient
```

**When to use**: Limited GPU memory (GTX 1660, RTX 2060)

#### Option C: Strong Adversarial (Best Cross-Lingual)
```bash
python train_adversarial_hubert.py \
    --epochs 25 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --emotion_weight 1.0 \
    --language_weight 0.3 \
    --lambda_schedule progressive \
    --output_dir models/adversarial_hubert \
    --experiment_name strong_adversarial
```

**When to use**: Want best cross-lingual transfer, willing to train longer

#### Option D: Quick Test (Fast Iteration)
```bash
python train_adversarial_hubert.py \
    --epochs 5 \
    --batch_size 8 \
    --adversarial_layers 12 \
    --output_dir models/adversarial_hubert \
    --experiment_name quick_test
```

**When to use**: Testing the pipeline, debugging, quick experiments

### 3.2 Start Training

Let's use the default configuration:

```bash
python train_adversarial_hubert.py \
    --epochs 20 \
    --batch_size 8 \
    --output_dir models/adversarial_hubert \
    --experiment_name default_run
```

### 3.3 What Happens During Training

You'll see output like this:

```
================================================================================
Training Adversarial HuBERT for Cross-Lingual Emotion Recognition
================================================================================
Device: cuda
GPU: NVIDIA GeForce RTX 3090
GPU Memory: 24.00 GB

Emotions: ['sad', 'neutral', 'happy', 'angry']
Number of emotions: 4

================================================================================
Loading Data
================================================================================

English Training Data:
  Loaded 536 samples from data/csv/train_ravdess_4class.csv (english)
  Loaded 961 samples from data/csv/train_tess_4class.csv (english)

English Test Data:
  Loaded 136 samples from data/csv/test_ravdess_4class.csv (english)
  Loaded 321 samples from data/csv/test_tess_4class.csv (english)

Hindi Training Data:
  Loaded 1280 samples from data/csv/train_hindi_4class.csv (hindi)

Hindi Test Data:
  Loaded 321 samples from data/csv/test_hindi_4class.csv (hindi)

================================================================================
Dataset Summary
================================================================================
English Train: 1497 samples
English Test: 457 samples
Hindi Train: 1280 samples
Hindi Test: 321 samples
Total Train: 2777 samples
Total Test: 778 samples

================================================================================
Loading Processor
================================================================================
[Downloading HuBERT processor...]

================================================================================
Creating DataLoaders
================================================================================
Train batches: 347
Test batches (combined): 98
Test batches (English): 58
Test batches (Hindi): 41

================================================================================
Creating Model
================================================================================
Loading HuBERT model: facebook/hubert-base-ls960
Freezing HuBERT feature extractor (CNN layers)

======================================================================
Layer-Wise Adversarial HuBERT Model
======================================================================
Model: facebook/hubert-base-ls960
Emotions: 4 classes
Languages: 2 classes
Adversarial Layers: [3, 6, 9, 12]
Feature Dim: 768
Hidden Dim: 256
Dropout: 0.3
Frozen Feature Extractor: True
Gradient Checkpointing: False
======================================================================

Trainable Parameters: 2,456,832 (2.46M)
Total Parameters: 94,891,264 (94.89M)
Frozen Parameters: 92,434,432 (92.43M)

Mixed precision training: ENABLED

================================================================================
Training
================================================================================
```

---

## Step 4: Monitoring Training

### 4.1 Understanding Training Output

During each epoch, you'll see:

```
Epoch 1/20
100%|████████████| 347/347 [08:23<00:00,  1.45s/it, loss=1.2345, λ=0.123]

Epoch 1/20
================================================================================
Train Loss: 1.2345 | Emotion Loss: 1.1234 | Language Loss: 0.1111
Train Emotion Acc: 0.4567 | Train Emotion F1: 0.4321
Train Language Acc: 0.8900 (Lower is better - indicates language invariance)

Test Results:
  Combined  - Emotion Acc: 0.4234 | F1: 0.4123 | Language Acc: 0.8765
  English   - Emotion Acc: 0.4567 | F1: 0.4456
  Hindi     - Emotion Acc: 0.3901 | F1: 0.3790
  Cross-lingual Transfer Gap: 0.0666 (Lower is better)
  *** New best model saved! F1: 0.4123 ***
================================================================================
```

### 4.2 Key Metrics to Watch

| Metric | What to Look For | Good Trend |
|--------|-----------------|------------|
| **Total Loss** | Should decrease | 1.5 → 0.8 |
| **Emotion Accuracy** | Should increase | 45% → 75% |
| **Language Accuracy** | Should decrease | 90% → 55% |
| **Transfer Gap** | Should decrease | 15% → 8% |
| **Lambda (λ)** | Progressive increase | 0.0 → 1.0 |

### 4.3 Good vs Bad Training

**✓ Good Training Signs:**
- Emotion accuracy increasing steadily
- Language accuracy decreasing (model learning invariance)
- Transfer gap shrinking
- Both English and Hindi accuracies improving together

**✗ Warning Signs:**
- Loss not decreasing after 5 epochs → reduce learning rate
- Language accuracy stays >75% → increase language_weight
- Emotion accuracy <60% after 10 epochs → decrease language_weight
- Large transfer gap (>15%) → train longer or increase adversarial strength

### 4.4 Training Time Estimates

| GPU | Batch Size | Time per Epoch | Total (20 epochs) |
|-----|------------|----------------|-------------------|
| RTX 3090 | 8 | ~8 min | ~2.5 hours |
| RTX 2080 Ti | 8 | ~12 min | ~4 hours |
| RTX 3060 | 4 | ~15 min | ~5 hours |
| GTX 1660 | 4 | ~20 min | ~6.5 hours |

### 4.5 Monitoring During Training

Open a **new terminal** and check progress:

```bash
# View training history
cat models/adversarial_hubert/default_run/training_history.json | python -m json.tool | tail -50

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor output files
ls -lht models/adversarial_hubert/default_run/
```

---

## Step 5: Evaluating Cross-Lingual Transfer

### 5.1 After Training Completes

Once training finishes, you'll see:

```
================================================================================
Training Complete!
================================================================================
Best Combined F1: 0.7234
Models saved to: models/adversarial_hubert/default_run
================================================================================
```

### 5.2 Run Cross-Lingual Evaluation

```bash
python evaluate_crosslingual.py \
    --model_path models/adversarial_hubert/default_run/best_model.pth \
    --output_dir results/crosslingual/default_run \
    --evaluate_both_directions \
    --batch_size 16
```

### 5.3 Evaluation Output

You'll see detailed results:

```
================================================================================
Cross-Lingual Evaluation for Adversarial HuBERT
================================================================================
Device: cuda
Model: models/adversarial_hubert/default_run/best_model.pth

Loading model checkpoint...
Creating model...
Model loaded successfully!
Emotions: ['sad', 'neutral', 'happy', 'angry']

================================================================================
Loading Data
================================================================================

Loading English data...
  Loaded 136 samples from data/csv/test_ravdess_4class.csv (english)
  Loaded 321 samples from data/csv/test_tess_4class.csv (english)

Loading Hindi data...
  Loaded 321 samples from data/csv/test_hindi_4class.csv (hindi)

################################################################################
Scenario 1: Train on HINDI, Test on ENGLISH
################################################################################

Creating dataloaders...

Evaluating within-language (HINDI)...
100%|████████████| 21/21 [00:15<00:00,  1.35it/s]

Evaluating cross-lingual (ENGLISH)...
100%|████████████| 29/29 [00:21<00:00,  1.38it/s]

================================================================================
Results: HINDI → ENGLISH
================================================================================
Within-language (HINDI):
  Accuracy: 0.7289
  F1 Score: 0.7234

Cross-lingual (ENGLISH):
  Accuracy: 0.6872
  F1 Score: 0.6801

Transfer Gap: 0.0433 (Lower is better)
Transfer Rate: 93.98%
================================================================================

Classification Report - Within-Language (HINDI):
              precision    recall  f1-score   support

         sad     0.7500    0.7200    0.7347        75
     neutral     0.7100    0.7800    0.7434        82
       happy     0.7300    0.7000    0.7147        80
       angry     0.7200    0.7500    0.7347        84

    accuracy                         0.7289       321
   macro avg     0.7275    0.7375    0.7319       321
weighted avg     0.7298    0.7289    0.7234       321


Classification Report - Cross-Lingual (ENGLISH):
              precision    recall  f1-score   support

         sad     0.7000    0.6700    0.6847       112
     neutral     0.6800    0.7100    0.6947       118
       happy     0.6900    0.6600    0.6747       110
       angry     0.6700    0.7000    0.6847       117

    accuracy                         0.6872       457
   macro avg     0.6850    0.6850    0.6847       457
weighted avg     0.6874    0.6872    0.6801       457


################################################################################
Scenario 2: Train on ENGLISH, Test on HINDI
################################################################################

[Similar output for English → Hindi]

================================================================================
Language Invariance Analysis
================================================================================

Creating combined test set...
Analyzing language discriminator performance...
(Lower accuracy = better language invariance)

Language Discriminator Performance:
  layer_3: Accuracy = 0.5821, Confusion = 0.4179
  layer_6: Accuracy = 0.5645, Confusion = 0.4355
  layer_9: Accuracy = 0.5498, Confusion = 0.4502
  layer_12: Accuracy = 0.5234, Confusion = 0.4766

  Average: Accuracy = 0.5550, Confusion = 0.4450

  ✓ Good language invariance! Discriminator performs poorly.

================================================================================
Evaluation Complete!
================================================================================
Results saved to: results/crosslingual/default_run/crosslingual_evaluation.json
Confusion matrices saved to: results/crosslingual/default_run
================================================================================
```

### 5.4 View Results Summary

```bash
python -c "
import json
with open('results/crosslingual/default_run/crosslingual_evaluation.json') as f:
    results = json.load(f)

    print('\n' + '='*70)
    print('CROSS-LINGUAL EVALUATION SUMMARY')
    print('='*70)

    # Hindi → English
    h2e = results['hindi_to_english']
    print(f'\nHindi → English Transfer:')
    print(f'  Within-language F1: {h2e[\"within_language\"][\"f1\"]:.4f}')
    print(f'  Cross-lingual F1:   {h2e[\"cross_lingual\"][\"f1\"]:.4f}')
    print(f'  Transfer Gap:       {h2e[\"transfer_gap\"]:.4f}')
    print(f'  Transfer Rate:      {h2e[\"transfer_rate\"]:.2f}%')

    # English → Hindi
    if 'english_to_hindi' in results:
        e2h = results['english_to_hindi']
        print(f'\nEnglish → Hindi Transfer:')
        print(f'  Within-language F1: {e2h[\"within_language\"][\"f1\"]:.4f}')
        print(f'  Cross-lingual F1:   {e2h[\"cross_lingual\"][\"f1\"]:.4f}')
        print(f'  Transfer Gap:       {e2h[\"transfer_gap\"]:.4f}')
        print(f'  Transfer Rate:      {e2h[\"transfer_rate\"]:.2f}%')

    # Language invariance
    lang_inv = results['language_invariance']['average']
    print(f'\nLanguage Invariance:')
    print(f'  Discriminator Accuracy: {lang_inv[\"accuracy\"]:.4f}')
    print(f'  (50% = random = perfect invariance)')

    print('='*70 + '\n')
"
```

### 5.5 View Confusion Matrices

```bash
# View confusion matrices (if you have image viewer)
xdg-open results/crosslingual/default_run/confusion_within_hindi.png
xdg-open results/crosslingual/default_run/confusion_cross_hindi_to_english.png
```

---

## Step 6: Making Predictions

### 6.1 Create a Prediction Script

Create `predict_adversarial.py`:

```python
"""
Simple prediction script for adversarial HuBERT
"""
import torch
import soundfile as sf
import librosa
import argparse
from transformers import Wav2Vec2Processor
from adversarial_hubert_emotion import LayerWiseAdversarialHuBERT


def load_model(model_path, device='cuda'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']
    id_to_emotion = checkpoint['id_to_emotion']

    # Create model
    model = LayerWiseAdversarialHuBERT(
        model_name=config['model_name'],
        num_emotions=len(id_to_emotion),
        num_languages=2,
        adversarial_layers=config['adversarial_layers'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        freeze_feature_extractor=config.get('freeze_feature_extractor', True),
        gradient_checkpointing=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, id_to_emotion, config


def predict_emotion(audio_path, model, processor, id_to_emotion, device='cuda', max_duration=5.0):
    """Predict emotion for a single audio file"""

    # Load audio
    speech, sr = sf.read(audio_path)

    # Resample if needed
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)

    # Truncate or pad
    max_length = int(max_duration * 16000)
    if len(speech) > max_length:
        speech = speech[:max_length]
    else:
        speech = librosa.util.pad_center(speech, size=max_length)

    # Process
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask, return_adversarial=False)
        logits = outputs['emotion_logits']
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    emotion = id_to_emotion[pred]
    confidence = probs[0][pred].item()

    # Get all probabilities
    all_probs = {id_to_emotion[i]: probs[0][i].item() for i in range(len(id_to_emotion))}

    return emotion, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description='Predict emotion from audio file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, id_to_emotion, config = load_model(args.model_path, device)

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(config['model_name'])

    # Predict
    print(f"\nPredicting emotion for: {args.audio_path}")
    emotion, confidence, all_probs = predict_emotion(
        args.audio_path, model, processor, id_to_emotion, device
    )

    print(f"\n{'='*60}")
    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'='*60}")
    print(f"\nAll Probabilities:")
    for emo, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emo:10s}: {prob:.2%} {'█' * int(prob * 50)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
```

Save this to `predict_adversarial.py`.

### 6.2 Make Single Prediction

```bash
python predict_adversarial.py \
    --model_path models/adversarial_hubert/default_run/best_model.pth \
    --audio_path data/ravdess/Actor_01/03-01-04-01-01-01-01.wav
```

Expected output:
```
Loading model from models/adversarial_hubert/default_run/best_model.pth...
Model loaded successfully!

Predicting emotion for: data/ravdess/Actor_01/03-01-04-01-01-01-01.wav

============================================================
Predicted Emotion: ANGRY
Confidence: 87.34%
============================================================

All Probabilities:
  angry     : 87.34% █████████████████████████████████████████████
  sad       : 8.23%  ████
  neutral   : 3.12%  █
  happy     : 1.31%
============================================================
```

### 6.3 Batch Predictions

Create `batch_predict_adversarial.py`:

```python
"""
Batch prediction script
"""
import os
import pandas as pd
import torch
from tqdm import tqdm
from predict_adversarial import load_model, predict_emotion
from transformers import Wav2Vec2Processor


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch predict emotions')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, id_to_emotion, config = load_model(args.model_path, device)
    processor = Wav2Vec2Processor.from_pretrained(config['model_name'])

    # Load CSV
    df = pd.read_csv(args.csv_path)

    # Predict for each file
    predictions = []
    confidences = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        audio_path = row['path']

        try:
            emotion, confidence, _ = predict_emotion(
                audio_path, model, processor, id_to_emotion, device
            )
            predictions.append(emotion)
            confidences.append(confidence)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            predictions.append("error")
            confidences.append(0.0)

    # Add predictions to dataframe
    df['predicted_emotion'] = predictions
    df['confidence'] = confidences

    # Calculate accuracy if ground truth exists
    if 'emotion' in df.columns:
        df['correct'] = df['emotion'] == df['predicted_emotion']
        accuracy = df['correct'].mean()
        print(f"\nAccuracy: {accuracy:.2%}")

    # Save results
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
```

Run batch prediction:

```bash
python batch_predict_adversarial.py \
    --model_path models/adversarial_hubert/default_run/best_model.pth \
    --csv_path data/csv/test_hindi_4class.csv \
    --output_csv results/hindi_predictions.csv
```

### 6.4 Interactive Prediction

```bash
# Create interactive script
cat > interactive_predict.py << 'EOF'
import torch
from predict_adversarial import load_model, predict_emotion
from transformers import Wav2Vec2Processor

# Load model once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "models/adversarial_hubert/default_run/best_model.pth"

print("Loading model...")
model, id_to_emotion, config = load_model(model_path, device)
processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
print("Ready!\n")

while True:
    audio_path = input("Enter audio file path (or 'quit' to exit): ").strip()

    if audio_path.lower() == 'quit':
        break

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        continue

    try:
        emotion, confidence, all_probs = predict_emotion(
            audio_path, model, processor, id_to_emotion, device
        )

        print(f"\nEmotion: {emotion.upper()} ({confidence:.1%})")
        print("All probabilities:")
        for emo, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emo}: {prob:.1%}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
EOF

python interactive_predict.py
```

---

## Step 7: Troubleshooting

### Issue 1: Out of Memory (OOM) During Training

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**Solutions:**

```bash
# Solution 1: Reduce batch size
python train_adversarial_hubert.py --batch_size 4

# Solution 2: Enable gradient checkpointing
python train_adversarial_hubert.py --gradient_checkpointing

# Solution 3: Reduce audio duration
python train_adversarial_hubert.py --max_duration 3.0

# Solution 4: Use fewer adversarial layers
python train_adversarial_hubert.py --adversarial_layers 6 12

# Solution 5: Reduce hidden dimension
python train_adversarial_hubert.py --hidden_dim 128

# Solution 6: All of the above (memory-efficient mode)
python train_adversarial_hubert.py \
    --batch_size 4 \
    --gradient_checkpointing \
    --max_duration 3.0 \
    --adversarial_layers 6 12 \
    --hidden_dim 128
```

### Issue 2: Training Too Slow

**Solutions:**

```bash
# Ensure mixed precision is enabled (default)
# Check if disabled by mistake
python train_adversarial_hubert.py  # Mixed precision ON by default

# Increase data loading workers
python train_adversarial_hubert.py --num_workers 8

# Freeze feature extractor (should be default)
python train_adversarial_hubert.py --freeze_feature_extractor
```

### Issue 3: Poor Emotion Accuracy (<60%)

**Symptoms:** Emotion accuracy stuck below 60% after many epochs

**Solutions:**

```bash
# Reduce adversarial strength
python train_adversarial_hubert.py --language_weight 0.05

# Train longer
python train_adversarial_hubert.py --epochs 30

# Increase learning rate
python train_adversarial_hubert.py --learning_rate 5e-5

# Try different lambda schedule
python train_adversarial_hubert.py --lambda_schedule linear
```

### Issue 4: Poor Cross-Lingual Transfer (>15% gap)

**Symptoms:** Large difference between English and Hindi performance

**Solutions:**

```bash
# Increase adversarial strength
python train_adversarial_hubert.py --language_weight 0.3

# Add more adversarial layers
python train_adversarial_hubert.py --adversarial_layers 1 3 6 9 12

# Train longer
python train_adversarial_hubert.py --epochs 30

# Use progressive lambda schedule
python train_adversarial_hubert.py --lambda_schedule progressive
```

### Issue 5: Language Discriminator Too Accurate (>75%)

**Symptoms:** Language accuracy stays high, model not learning invariance

**Solutions:**

```bash
# Increase language weight
python train_adversarial_hubert.py --language_weight 0.3

# Use progressive schedule
python train_adversarial_hubert.py --lambda_schedule progressive

# Train longer
python train_adversarial_hubert.py --epochs 25
```

### Issue 6: Model Not Loading for Prediction

**Error:**
```
KeyError: 'config'
```

**Solution:**
Make sure you're using the correct checkpoint:

```bash
# Use best_model.pth, not checkpoint_epochX.pth
python predict_adversarial.py \
    --model_path models/adversarial_hubert/default_run/best_model.pth \
    --audio_path your_audio.wav
```

---

## 📊 Quick Reference

### Training Command Template

```bash
python train_adversarial_hubert.py \
    --epochs [10-30] \
    --batch_size [4-16] \
    --learning_rate [1e-5 to 5e-5] \
    --emotion_weight [0.8-1.2] \
    --language_weight [0.05-0.3] \
    --lambda_schedule [constant|linear|progressive] \
    --adversarial_layers [space-separated layer numbers] \
    --output_dir models/adversarial_hubert \
    --experiment_name [your_experiment_name]
```

### Evaluation Command Template

```bash
python evaluate_crosslingual.py \
    --model_path models/adversarial_hubert/[experiment]/best_model.pth \
    --output_dir results/crosslingual/[experiment] \
    --evaluate_both_directions \
    --batch_size [8-32]
```

### Prediction Command Template

```bash
python predict_adversarial.py \
    --model_path models/adversarial_hubert/[experiment]/best_model.pth \
    --audio_path [path_to_audio.wav] \
    --device [cuda|cpu]
```

---

## ✅ Success Checklist

After completing all steps, you should have:

- [ ] Successfully installed all dependencies
- [ ] Verified dataset files exist and are accessible
- [ ] Trained model for at least 20 epochs
- [ ] Achieved >70% emotion accuracy on test set
- [ ] Transfer gap <15% between languages
- [ ] Language discriminator accuracy <65%
- [ ] Saved model checkpoints in `models/adversarial_hubert/`
- [ ] Generated cross-lingual evaluation results
- [ ] Confusion matrices saved in `results/crosslingual/`
- [ ] Successfully made predictions on new audio files

---

## 🎓 Next Steps

Now that you have a trained model:

1. **Experiment with configurations**: Try different hyperparameters
2. **Ablation studies**: Compare different numbers of adversarial layers
3. **Baseline comparison**: Train without adversarial (set `--language_weight 0.0`)
4. **Additional languages**: Extend to other language pairs
5. **Production deployment**: Export model for inference API

---

## 📚 Additional Resources

- **Full Documentation**: `ADVERSARIAL_HUBERT_README.md`
- **Quick Start**: `QUICKSTART_ADVERSARIAL.md`
- **Model Architecture**: `adversarial_hubert_emotion.py`
- **Configuration Presets**: `adversarial_configs.json`

---

**Congratulations! You now know how to train and use the adversarial HuBERT model!** 🎉
