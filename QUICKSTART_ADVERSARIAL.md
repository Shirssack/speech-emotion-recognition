# Quick Start Guide: Adversarial HuBERT

## Overview

This guide will help you get started with training and evaluating the layer-wise adversarial HuBERT model for cross-lingual speech emotion recognition.

## Prerequisites

```bash
# Ensure you have Python 3.8+ installed
python --version

# Install required packages (if not already installed)
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verify transformers installation
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

## Step 1: Verify Data

Ensure you have the required datasets in the correct locations:

```bash
ls -lh data/csv/train_*_4class.csv
ls -lh data/csv/test_*_4class.csv
```

Expected files:
- `train_ravdess_4class.csv` (English)
- `train_tess_4class.csv` (English)
- `train_hindi_4class.csv` (Hindi)
- `test_ravdess_4class.csv` (English)
- `test_tess_4class.csv` (English)
- `test_hindi_4class.csv` (Hindi)

## Step 2: Test Model Architecture

Verify the model architecture loads correctly:

```bash
python adversarial_hubert_emotion.py
```

Expected output:
- Model architecture summary
- Parameter counts (trainable and total)
- Test forward pass results
- Loss computation test
- Lambda scheduling test

## Step 3: Training

### Option A: Default Configuration (Recommended)

```bash
python train_adversarial_hubert.py \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --output_dir models/adversarial_hubert \
    --experiment_name default
```

### Option B: Memory-Efficient (for 4-6GB GPUs)

```bash
python train_adversarial_hubert.py \
    --adversarial_layers 6 12 \
    --hidden_dim 128 \
    --batch_size 4 \
    --max_duration 4.0 \
    --gradient_checkpointing \
    --output_dir models/adversarial_hubert \
    --experiment_name memory_efficient
```

### Option C: Strong Adversarial Training

```bash
python train_adversarial_hubert.py \
    --epochs 25 \
    --language_weight 0.3 \
    --output_dir models/adversarial_hubert \
    --experiment_name strong_adversarial
```

### Training Time Estimates

| Configuration | GPU | Batch Size | Time per Epoch | Total Time (20 epochs) |
|---------------|-----|------------|----------------|------------------------|
| Default | RTX 3090 | 8 | ~8 min | ~2.5 hours |
| Default | RTX 2080 Ti | 8 | ~12 min | ~4 hours |
| Memory-Efficient | GTX 1660 | 4 | ~20 min | ~6.5 hours |

## Step 4: Monitor Training

During training, you'll see:

```
Epoch 1/20
  loss: 1.2345, λ: 0.123
  Train Loss: 1.2345 | Emotion Loss: 1.1234 | Language Loss: 0.1111
  Train Emotion Acc: 0.4567 | Train Emotion F1: 0.4321
  Train Language Acc: 0.8900 (Lower is better - indicates language invariance)

  Test Results:
    Combined  - Emotion Acc: 0.4234 | F1: 0.4123 | Language Acc: 0.8765
    English   - Emotion Acc: 0.4567 | F1: 0.4456
    Hindi     - Emotion Acc: 0.3901 | F1: 0.3790
    Cross-lingual Transfer Gap: 0.0666 (Lower is better)
```

## Step 5: Evaluation

After training completes, evaluate cross-lingual transfer:

```bash
python evaluate_crosslingual.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --output_dir results/crosslingual/default \
    --evaluate_both_directions
```

This will:
1. Load the trained model
2. Evaluate Hindi → English transfer
3. Evaluate English → Hindi transfer
4. Analyze language invariance
5. Generate confusion matrices
6. Save results to JSON

## Step 6: Analyze Results

Check the output directory for results:

```bash
ls -lh results/crosslingual/default/
```

Files generated:
- `crosslingual_evaluation.json` - Quantitative metrics
- `confusion_within_english.png` - Within-language confusion matrix (English)
- `confusion_within_hindi.png` - Within-language confusion matrix (Hindi)
- `confusion_cross_hindi_to_english.png` - Cross-lingual confusion matrix
- `confusion_cross_english_to_hindi.png` - Cross-lingual confusion matrix

View results:

```bash
# Print summary
python -c "
import json
with open('results/crosslingual/default/crosslingual_evaluation.json') as f:
    results = json.load(f)
    print('Hindi → English Transfer:')
    print(f\"  Within-language F1: {results['hindi_to_english']['within_language']['f1']:.4f}\")
    print(f\"  Cross-lingual F1: {results['hindi_to_english']['cross_lingual']['f1']:.4f}\")
    print(f\"  Transfer Gap: {results['hindi_to_english']['transfer_gap']:.4f}\")
    print(f\"\\nLanguage Invariance:\")
    print(f\"  Discriminator Accuracy: {results['language_invariance']['average']['accuracy']:.4f}\")
    print(f\"  (Closer to 0.5 = better)\")
"
```

## Expected Results

### Good Results

- **Cross-lingual Transfer Gap**: < 10%
- **Language Discriminator Accuracy**: 50-60% (close to random)
- **Within-language F1**: > 70%
- **Cross-lingual F1**: > 65%

### Signs of Successful Training

1. **Language discriminator accuracy decreases** as training progresses
   - Epoch 1: ~90% → Epoch 20: ~55%
   - Indicates model is learning language-invariant features

2. **Transfer gap decreases** over epochs
   - Epoch 1: ~20% → Epoch 20: ~8%
   - Indicates better cross-lingual generalization

3. **Emotion accuracy increases** on both languages
   - Both English and Hindi test accuracies should improve together

## Troubleshooting

### Problem: Out of Memory (OOM)

**Solution**:
```bash
python train_adversarial_hubert.py \
    --batch_size 4 \
    --gradient_checkpointing \
    --adversarial_layers 6 12 \
    --max_duration 3.0
```

### Problem: Language Discriminator Too Accurate (>75%)

This means weak language invariance.

**Solution**:
```bash
python train_adversarial_hubert.py \
    --language_weight 0.2 \
    --epochs 25
```

### Problem: Low Emotion Accuracy (<60%)

**Solution**:
```bash
python train_adversarial_hubert.py \
    --language_weight 0.05 \
    --epochs 30 \
    --learning_rate 5e-5
```

### Problem: Training Too Slow

**Solution**:
```bash
python train_adversarial_hubert.py \
    --batch_size 12 \
    --num_workers 8 \
    --freeze_feature_extractor
```

## Next Steps

### Experiment 1: Ablation Study on Adversarial Layers

```bash
for layers in "12" "6 12" "3 6 9 12" "1 3 6 9 12"; do
    python train_adversarial_hubert.py \
        --adversarial_layers $layers \
        --experiment_name layers_$(echo $layers | tr ' ' '_')
done
```

### Experiment 2: Language Weight Sweep

```bash
for weight in 0.0 0.05 0.1 0.15 0.2 0.3; do
    python train_adversarial_hubert.py \
        --language_weight $weight \
        --experiment_name lang_weight_$weight
done
```

### Experiment 3: Lambda Schedule Comparison

```bash
for schedule in constant linear progressive; do
    python train_adversarial_hubert.py \
        --lambda_schedule $schedule \
        --experiment_name schedule_$schedule
done
```

## Performance Benchmarks

### GPU Memory Usage

| Configuration | Batch Size | GPU Memory | Speed |
|---------------|------------|------------|-------|
| Default + Frozen Extractor | 8 | ~6 GB | 1.0x |
| Default + Gradient Checkpointing | 8 | ~4 GB | 0.7x |
| Memory-Efficient | 4 | ~3 GB | 0.5x |
| Large Batch | 16 | ~11 GB | 1.3x |

### Expected Accuracy

| Dataset Split | Baseline HuBERT | Adversarial HuBERT | Improvement |
|---------------|-----------------|-------------------|-------------|
| English Test | 72% | 75% | +3% |
| Hindi Test | 68% | 71% | +3% |
| Hindi → English | 58% | 68% | +10% |
| English → Hindi | 55% | 65% | +10% |

## Getting Help

1. **Check logs**: Training logs are saved in `models/adversarial_hubert/<experiment_name>/training_history.json`
2. **Review README**: See `ADVERSARIAL_HUBERT_README.md` for detailed documentation
3. **Test architecture**: Run `python adversarial_hubert_emotion.py` to verify model works
4. **GitHub Issues**: Report bugs or ask questions on GitHub

## Summary

Congratulations! You've successfully:
- ✅ Set up the adversarial HuBERT environment
- ✅ Trained a layer-wise adversarial model
- ✅ Evaluated cross-lingual transfer
- ✅ Analyzed language invariance

For more details, see:
- `ADVERSARIAL_HUBERT_README.md` - Comprehensive documentation
- `adversarial_configs.json` - Configuration presets
- `adversarial_hubert_emotion.py` - Model architecture
- `train_adversarial_hubert.py` - Training script
- `evaluate_crosslingual.py` - Evaluation script
