# Training and Prediction Workflow - Quick Reference

## Visual Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STEP 1: SETUP                                │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ pip install -r requirements.txt                               ║  │
│  ║ python check_gpu.py                                           ║  │
│  ║ ls data/csv/*.csv                                             ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STEP 2: TRAINING                             │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ python train_adversarial_hubert.py \                          ║  │
│  ║     --epochs 20 \                                             ║  │
│  ║     --batch_size 8 \                                          ║  │
│  ║     --output_dir models/adversarial_hubert \                  ║  │
│  ║     --experiment_name my_model                                ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│  Progress: [████████████████████] Epoch 20/20                        │
│  Best F1: 0.7234                                                     │
│  Transfer Gap: 8.2% ✓                                                │
│                                                                       │
│  Output: models/adversarial_hubert/my_model/best_model.pth          │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 3: EVALUATION                               │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ python evaluate_crosslingual.py \                             ║  │
│  ║     --model_path models/adversarial_hubert/my_model/best...  ║  │
│  ║     --output_dir results/crosslingual \                       ║  │
│  ║     --evaluate_both_directions                                ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│  Results:                                                            │
│    Hindi → English: 68.2% ✓                                          │
│    English → Hindi: 65.7% ✓                                          │
│    Language Invariance: 55% ✓ (closer to 50% = better)              │
│                                                                       │
│  Output: results/crosslingual/crosslingual_evaluation.json          │
│          results/crosslingual/confusion_*.png                        │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 4A: SINGLE PREDICTION                        │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ python predict_adversarial.py \                               ║  │
│  ║     --model_path models/adversarial_hubert/my_model/best...  ║  │
│  ║     --audio_path path/to/audio.wav                            ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│  Output:                                                             │
│    PREDICTED EMOTION: ANGRY                                          │
│    Confidence: 87.34%                                                │
│                                                                       │
│    angry    : 87.34% █████████████████████████████████████████      │
│    sad      :  8.23% ████                                            │
│    neutral  :  3.12% █                                               │
│    happy    :  1.31%                                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 4B: BATCH PREDICTION                         │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║ python batch_predict_adversarial.py \                         ║  │
│  ║     --model_path models/adversarial_hubert/my_model/best...  ║  │
│  ║     --csv_path data/csv/test_hindi_4class.csv \               ║  │
│  ║     --output_csv results/predictions.csv                      ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│  Output:                                                             │
│    Total Samples: 321                                                │
│    Accuracy: 72.89%                                                  │
│    F1 Score: 0.7234                                                  │
│                                                                       │
│  File: results/predictions.csv                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Command Cheat Sheet

### 1. Training Commands

#### Basic Training (Default)
```bash
python train_adversarial_hubert.py \
    --epochs 20 \
    --batch_size 8 \
    --output_dir models/adversarial_hubert \
    --experiment_name default
```

#### Memory-Efficient (4-6GB GPU)
```bash
python train_adversarial_hubert.py \
    --batch_size 4 \
    --adversarial_layers 6 12 \
    --gradient_checkpointing \
    --max_duration 4.0 \
    --experiment_name memory_efficient
```

#### Strong Adversarial (Best Cross-Lingual)
```bash
python train_adversarial_hubert.py \
    --epochs 25 \
    --language_weight 0.3 \
    --experiment_name strong_adversarial
```

#### Quick Test (Fast)
```bash
python train_adversarial_hubert.py \
    --epochs 5 \
    --adversarial_layers 12 \
    --experiment_name quick_test
```

### 2. Evaluation Commands

#### Basic Evaluation
```bash
python evaluate_crosslingual.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --output_dir results/crosslingual/default
```

#### Full Evaluation (Both Directions)
```bash
python evaluate_crosslingual.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --output_dir results/crosslingual/default \
    --evaluate_both_directions
```

### 3. Prediction Commands

#### Single File
```bash
python predict_adversarial.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --audio_path data/ravdess/Actor_01/03-01-04-01-01-01-01.wav
```

#### Batch Processing
```bash
python batch_predict_adversarial.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --csv_path data/csv/test_hindi_4class.csv \
    --output_csv results/hindi_predictions.csv
```

#### With Confusion Matrix
```bash
python batch_predict_adversarial.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --csv_path data/csv/test_hindi_4class.csv \
    --output_csv results/predictions.csv \
    --plot_confusion
```

---

## File Locations After Training

```
speech-emotion-recognition/
├── models/adversarial_hubert/
│   └── [experiment_name]/
│       ├── best_model.pth              ← Use this for prediction
│       ├── config.json                 ← Model configuration
│       ├── training_history.json       ← Training metrics
│       └── checkpoint_epoch*.pth       ← Per-epoch checkpoints
│
├── results/
│   ├── crosslingual/
│   │   └── [experiment_name]/
│   │       ├── crosslingual_evaluation.json
│   │       ├── confusion_within_english.png
│   │       ├── confusion_within_hindi.png
│   │       ├── confusion_cross_hindi_to_english.png
│   │       └── confusion_cross_english_to_hindi.png
│   │
│   └── predictions.csv                 ← Batch predictions
```

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of Memory | Use `--batch_size 4 --gradient_checkpointing` |
| Training too slow | Add `--freeze_feature_extractor --num_workers 8` |
| Low emotion accuracy | Try `--language_weight 0.05 --epochs 30` |
| Poor cross-lingual transfer | Try `--language_weight 0.3 --epochs 25` |
| Language discriminator too accurate | Try `--language_weight 0.3` or `--lambda_schedule progressive` |

---

## Expected Training Time

| GPU | Batch Size | Configuration | Time per Epoch | Total (20 epochs) |
|-----|------------|---------------|----------------|-------------------|
| RTX 3090 | 8 | Default | ~8 min | ~2.5 hours |
| RTX 2080 Ti | 8 | Default | ~12 min | ~4 hours |
| RTX 3060 | 4 | Memory-Efficient | ~15 min | ~5 hours |
| GTX 1660 | 4 | Memory-Efficient | ~20 min | ~6.5 hours |

---

## Performance Targets

### Training Metrics (after convergence)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Emotion Accuracy** | >65% | >70% | >75% |
| **Emotion F1 Score** | >0.65 | >0.70 | >0.75 |
| **Transfer Gap** | <15% | <10% | <8% |
| **Language Discriminator** | <70% | <60% | <55% |

### Evaluation Metrics

| Metric | Baseline | Adversarial | Target |
|--------|----------|-------------|--------|
| **Within-language F1** | 0.72 | 0.75 | >0.75 |
| **Cross-lingual F1** | 0.58 | 0.68 | >0.65 |
| **Transfer Gap** | 15% | 8% | <10% |

---

## One-Liner Commands

### Complete Pipeline
```bash
# Train, evaluate, and predict in one go
python train_adversarial_hubert.py --epochs 20 --experiment_name my_model && \
python evaluate_crosslingual.py --model_path models/adversarial_hubert/my_model/best_model.pth --output_dir results/crosslingual/my_model --evaluate_both_directions && \
python predict_adversarial.py --model_path models/adversarial_hubert/my_model/best_model.pth --audio_path data/ravdess/Actor_01/03-01-04-01-01-01-01.wav
```

### Test All Datasets
```bash
# Predict on all test sets
for dataset in ravdess tess hindi; do
    python batch_predict_adversarial.py \
        --model_path models/adversarial_hubert/default/best_model.pth \
        --csv_path data/csv/test_${dataset}_4class.csv \
        --output_csv results/${dataset}_predictions.csv
done
```

---

## Next Steps

After successful training and prediction:

1. **🔬 Run Experiments**: Try different configurations from `adversarial_configs.json`
2. **📊 Analyze Results**: Compare cross-lingual transfer gaps
3. **🎯 Fine-tune**: Adjust hyperparameters based on your results
4. **📝 Document**: Record your best configurations and results
5. **🚀 Deploy**: Use the model in your application

---

## Documentation Links

- **📘 [Step-by-Step Guide](STEP_BY_STEP_GUIDE.md)** - Complete walkthrough
- **📚 [Full Documentation](ADVERSARIAL_HUBERT_README.md)** - Research details
- **⚡ [Quick Start](QUICKSTART_ADVERSARIAL.md)** - Fast setup
- **⚙️ [Configuration Guide](adversarial_configs.json)** - Preset configurations

---

## Support

**Having issues?**
1. Check [Step-by-Step Guide - Troubleshooting](STEP_BY_STEP_GUIDE.md#step-7-troubleshooting)
2. Review [Expected Results](ADVERSARIAL_HUBERT_README.md#-expected-results)
3. Verify your data with `ls data/csv/*.csv`
4. Test GPU with `python check_gpu.py`

**Need help?**
- Open an issue on GitHub
- Check existing documentation
- Review example commands above

---

**Happy Training!** 🎉
