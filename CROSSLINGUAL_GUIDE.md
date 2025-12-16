# Quick Guide: Cross-Lingual Emotion Recognition

Fast reference for using the HuBERT cross-lingual system.

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install transformers accelerate
```

### 2. Train the Model
```bash
python train_hubert_crosslingual.py --epochs 15 --batch_size 8
```

### 3. Evaluate Cross-Lingual Performance
```bash
python evaluate_crosslingual.py
```

---

## 📋 Common Commands

### Training Configurations

**Standard Training** (recommended for 6-8GB GPU):
```bash
python train_hubert_crosslingual.py \
    --epochs 15 \
    --batch_size 8 \
    --lr 1e-4 \
    --lambda_adv 1.0 \
    --alpha_schedule progressive
```

**Fast Training** (for testing):
```bash
python train_hubert_crosslingual.py --epochs 3 --batch_size 4
```

**Low Memory** (4GB GPU):
```bash
python train_hubert_crosslingual.py \
    --batch_size 4 \
    --max_duration 3.0
```

**Strong Adversarial** (better cross-lingual):
```bash
python train_hubert_crosslingual.py \
    --lambda_adv 2.0 \
    --adversarial_layers 6 9 12
```

**Single Layer** (faster, less effective):
```bash
python train_hubert_crosslingual.py \
    --adversarial_layers 12
```

### Baseline Comparisons

**No Adversarial** (baseline):
```bash
python train_hubert_crosslingual.py --lambda_adv 0.0
```

**English Only**:
```bash
python train_hubert_crosslingual.py \
    --train_csv data/csv/train_ravdess_4class.csv data/csv/train_tess_4class.csv \
    --val_csv data/csv/test_ravdess_4class.csv data/csv/test_tess_4class.csv
```

**Hindi Only**:
```bash
python train_hubert_crosslingual.py \
    --train_csv data/csv/train_hindi_4class.csv \
    --val_csv data/csv/test_hindi_4class.csv
```

---

## 🎯 Model Parameters Explained

| Parameter | What it Does | Typical Values |
|-----------|--------------|----------------|
| `--epochs` | Training iterations | 10-20 (15 recommended) |
| `--batch_size` | Samples per batch | 4-16 (8 recommended) |
| `--lr` | Learning rate | 5e-5 to 2e-4 |
| `--lambda_adv` | Adversarial strength | 0.5-2.0 (1.0 recommended) |
| `--adversarial_layers` | Which layers to use | 6 9 12 (recommended) |
| `--alpha_schedule` | How alpha changes | progressive (recommended) |
| `--max_duration` | Audio length (seconds) | 3.0-8.0 (5.0 recommended) |

---

## 📊 Interpreting Results

### Training Output

```
Epoch 1/15
Training Results:
  Total Loss: 2.3456      ← Should decrease over time
  Emotion Loss: 1.2345    ← Main task loss
  Language Loss: 1.1111   ← Should stay ~0.69 (log(2) for 2 classes)
  Accuracy: 45.67%        ← Should increase over time

Validation Results:
  Loss: 2.1234
  Accuracy: 48.23%
```

**Good Training**:
- Emotion loss decreases steadily
- Language loss stays around 0.69 (random guessing)
- Accuracy increases to 70-85%

**Bad Training**:
- Language loss near 0 = adversarial not working
- Emotion loss stuck = model not learning
- Accuracy plateaus early = overfitting or bad hyperparameters

### Evaluation Output

```
[In-domain English]
Accuracy: 78.45%

[In-domain Hindi]
Accuracy: 73.21%

Cross-Lingual Analysis:
  Average In-Domain: 75.83%
  Cross-Lingual:     72.15%
  Gap:               3.68%    ← Should be <10%
```

**Success Metrics**:
- ✓ Gap < 5%: Excellent cross-lingual transfer
- ✓ Gap 5-10%: Good cross-lingual transfer
- ⚠ Gap > 10%: Poor cross-lingual transfer (increase lambda_adv)

---

## 🔧 Troubleshooting

### Problem: Out of Memory

**Solution**:
```bash
python train_hubert_crosslingual.py --batch_size 4 --max_duration 3.0
```

### Problem: Training Very Slow

**Possible causes**:
- No GPU available
- CPU inference being used

**Check**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Problem: Language Discriminator Too Good (>90% accuracy)

**Meaning**: Adversarial training not working
**Solution**: Increase lambda_adv
```bash
python train_hubert_crosslingual.py --lambda_adv 2.0
```

### Problem: Poor Emotion Accuracy (<60%)

**Possible causes**:
- Too much adversarial confusion
- Learning rate too high

**Solution**:
```bash
python train_hubert_crosslingual.py --lambda_adv 0.5 --lr 5e-5
```

### Problem: Large Cross-Lingual Gap (>15%)

**Meaning**: Model not generalizing well across languages
**Solutions**:
```bash
# Increase adversarial strength
python train_hubert_crosslingual.py --lambda_adv 2.0

# Use more layers
python train_hubert_crosslingual.py --adversarial_layers 3 6 9 12

# Train longer
python train_hubert_crosslingual.py --epochs 25
```

---

## 📈 Experiment Workflow

### Complete Ablation Study

```bash
# 1. Baseline (no adversarial)
python train_hubert_crosslingual.py --lambda_adv 0.0 --save_dir models/baseline

# 2. Single layer
python train_hubert_crosslingual.py --adversarial_layers 12 --save_dir models/single_layer

# 3. Multi-layer (proposed)
python train_hubert_crosslingual.py --adversarial_layers 6 9 12 --save_dir models/multi_layer

# 4. Evaluate all
python evaluate_crosslingual.py --model_path models/baseline/hubert_crosslingual_best.pt
python evaluate_crosslingual.py --model_path models/single_layer/hubert_crosslingual_best.pt
python evaluate_crosslingual.py --model_path models/multi_layer/hubert_crosslingual_best.pt
```

### Lambda Sensitivity Analysis

```bash
for lambda in 0.1 0.5 1.0 2.0 5.0; do
    python train_hubert_crosslingual.py \
        --lambda_adv $lambda \
        --save_dir models/lambda_$lambda \
        --epochs 15
done
```

---

## 🎓 Research Paper Workflow

### 1. Train Models

```bash
# Baseline
python train_hubert_crosslingual.py --lambda_adv 0.0 --save_dir models/baseline

# Proposed (multi-layer adversarial)
python train_hubert_crosslingual.py --adversarial_layers 6 9 12 --save_dir models/proposed
```

### 2. Evaluate

```bash
python evaluate_crosslingual.py --model_path models/baseline/hubert_crosslingual_best.pt > results_baseline.txt
python evaluate_crosslingual.py --model_path models/proposed/hubert_crosslingual_best.pt > results_proposed.txt
```

### 3. Compare Results

The key metrics for your paper:

| Method | English | Hindi | Avg In-Domain | Cross-Lingual Gap |
|--------|---------|-------|---------------|-------------------|
| Baseline (No Adv) | XX% | XX% | XX% | XX% |
| Single-Layer Adv | XX% | XX% | XX% | XX% |
| **Multi-Layer Adv (Proposed)** | **XX%** | **XX%** | **XX%** | **XX%** ← Should be lowest |

---

## 💡 Tips for Best Results

1. **Start with default settings** - they're well-tuned
2. **Use progressive alpha** - better than constant
3. **Monitor language loss** - should stay ~0.69
4. **Train for 15+ epochs** - earlier epochs may not converge
5. **Use validation set** - prevents overfitting
6. **Try different random seeds** - report average of 3-5 runs

---

## 📝 Making Predictions

After training, predict on new audio:

```python
from hubert_crosslingual_emotion import CrossLingualEmotionRecognizer

# Load model
rec = CrossLingualEmotionRecognizer(
    emotions=['sad', 'neutral', 'happy', 'angry'],
    adversarial_layers=[6, 9, 12]
)
rec.load_model('models/hubert_crosslingual_best.pt')

# Predict
emotion, confidence = rec.predict('path/to/audio.wav')
print(f"Predicted: {emotion} ({confidence:.2%})")
```

---

## 📚 Files Overview

| File | Purpose |
|------|---------|
| `hubert_crosslingual_emotion.py` | Main model implementation |
| `train_hubert_crosslingual.py` | Training script |
| `evaluate_crosslingual.py` | Cross-lingual evaluation |
| `CROSSLINGUAL_README.md` | Full documentation |
| `CROSSLINGUAL_GUIDE.md` | This quick reference |

---

## ⏱️ Training Time Estimates

| Configuration | Time (6GB GPU) | Time (CPU) |
|---------------|----------------|------------|
| 3 epochs (test) | 15-30 min | 2-4 hours |
| 15 epochs (recommended) | 1-2 hours | 10-15 hours |
| 25 epochs (thorough) | 2-3 hours | 15-25 hours |

**Tip**: Use GPU! CPU training is ~10x slower.

---

For detailed information, see `CROSSLINGUAL_README.md`
