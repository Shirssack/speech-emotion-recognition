# Transformer-Based Speech Emotion Recognition

Advanced emotion recognition system using **Wav2Vec2** transformers with support for multilingual datasets (English + Hindi).

## Features

- **State-of-the-art Transformer Architecture**: Wav2Vec2-based model for superior audio understanding
- **Memory-Efficient Training**: Optimized for 6GB GPU with:
  - Gradient checkpointing
  - Mixed precision training (FP16)
  - Configurable batch sizes
- **Multilingual Support**: Train on English (RAVDESS, TESS) and Hindi emotion datasets
- **Easy Fine-tuning**: Pretrained models allow quick adaptation to your data
- **Production Ready**: Save/load models, predict on new audio files

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- librosa >= 0.9.0
- accelerate >= 0.20.0

## Quick Start

### 1. Generate Hindi Dataset CSVs (if using Hindi data)

```bash
python generate_hindi_csv.py
```

This creates:
- `data/csv/train_hindi.csv`
- `data/csv/test_hindi.csv`

### 2. Train Model (Quick Test)

```bash
# Quick 3-epoch test on English datasets
python train_transformer.py \
    --epochs 3 \
    --batch_size 8 \
    --emotions sad neutral happy
```

### 3. Train Model (Full Training)

```bash
# Standard training (15 epochs)
python train_transformer.py \
    --epochs 15 \
    --batch_size 8 \
    --emotions sad neutral happy \
    --model_name facebook/wav2vec2-base
```

### 4. Train with All Datasets (Multilingual - Default)

```bash
# Multilingual training with Hindi + English (default behavior)
python train_transformer.py \
    --epochs 20 \
    --batch_size 8 \
    --emotions sad neutral happy angry
```

To exclude Hindi and use only English datasets:
```bash
python train_transformer.py \
    --exclude_hindi \
    --epochs 15 \
    --batch_size 8
```

## Training Configurations

### For 6GB GPU (Recommended)

```bash
python train_transformer.py \
    --model_name facebook/wav2vec2-base \
    --batch_size 8 \
    --max_duration 5.0 \
    --freeze_encoder \
    --epochs 15
```

### If Out of Memory (OOM)

Reduce batch size or audio duration:

```bash
python train_transformer.py \
    --batch_size 4 \
    --max_duration 3.0 \
    --epochs 15
```

### Advanced: Full Fine-tuning (requires more memory)

```bash
python train_transformer.py \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --epochs 20
    # Note: no --freeze_encoder flag
```

## Command-Line Arguments

### Data Arguments
- `--train_csv`: Training CSV file(s) (default: RAVDESS + TESS + Hindi)
- `--test_csv`: Testing CSV file(s) (default: RAVDESS + TESS + Hindi)
- `--exclude_hindi`: Exclude Hindi dataset (use only English datasets)
- `--emotions`: Emotions to train on (default: `sad neutral happy angry`)

### Model Arguments
- `--model_name`: Pretrained model (default: `facebook/wav2vec2-base`)
  - Options: `facebook/wav2vec2-base`, `facebook/hubert-base-ls960`
- `--freeze_encoder`: Freeze feature extractor (recommended for 6GB GPU)
- `--weighted_layers`: Use weighted sum of all transformer layers

### Training Arguments
- `--epochs`: Number of epochs (default: 15)
- `--batch_size`: Batch size (default: 8, reduce to 4 if OOM)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--warmup_ratio`: Warmup ratio for LR scheduler (default: 0.1)
- `--max_duration`: Max audio duration in seconds (default: 5.0)
- `--no_mixed_precision`: Disable FP16 training

### Output Arguments
- `--output_dir`: Directory to save models (default: `models/`)
- `--model_name_suffix`: Model filename suffix (default: `transformer`)

## Usage Examples

### Example 1: English 3-Class Emotion Recognition

```bash
python train_transformer.py \
    --emotions sad neutral happy \
    --epochs 15 \
    --batch_size 8
```

### Example 2: English 6-Class Extended

```bash
python train_transformer.py \
    --emotions angry sad neutral happy fear disgust \
    --epochs 20 \
    --batch_size 8
```

### Example 3: Multilingual (English + Hindi - Default)

```bash
# Multilingual is now default, just train normally
python train_transformer.py \
    --emotions sad neutral happy angry \
    --epochs 20 \
    --batch_size 8 \
    --model_name_suffix multilingual
```

### Example 4: Hindi Only

```bash
python train_transformer.py \
    --train_csv data/csv/train_hindi.csv \
    --test_csv data/csv/test_hindi.csv \
    --emotions angry happy neutral sad fear disgust \
    --epochs 15 \
    --model_name_suffix hindi_only
```

## Using Trained Models

### Load and Predict

```python
from transformer_emotion_recognition import TransformerEmotionRecognizer

# Initialize recognizer
recognizer = TransformerEmotionRecognizer(
    emotions=['sad', 'neutral', 'happy']
)

# Load trained model
recognizer.load_model('models/transformer_best.pt')

# Predict on new audio
emotion, confidence = recognizer.predict('path/to/audio.wav')
print(f"Emotion: {emotion} (confidence: {confidence:.2f})")
```

### Evaluate Model

```python
# Prepare test data
recognizer.prepare_data(
    train_paths, train_labels,
    test_paths, test_labels,
    batch_size=8
)

# Evaluate
accuracy, f1, loss = recognizer.evaluate()
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Get confusion matrix
cm = recognizer.get_confusion_matrix()
print(cm)
```

## Configuration File

See `transformer_config.json` for:
- Preset training configurations
- Model variants and GPU memory requirements
- Dataset configurations
- Memory optimization tips

## Model Performance

### Expected Results (with sufficient training data)

| Dataset | Emotions | Accuracy | F1 Score |
|---------|----------|----------|----------|
| RAVDESS + TESS (3-class) | sad, neutral, happy | ~75-85% | ~0.75-0.85 |
| RAVDESS + TESS (6-class) | angry, sad, neutral, happy, fear, disgust | ~65-75% | ~0.65-0.75 |
| Multilingual (with Hindi) | 6-class | ~70-80% | ~0.70-0.80 |

Results vary based on:
- Dataset size and quality
- Number of training epochs
- Model architecture
- Hyperparameter tuning

## Memory Optimization Tips

### For 6GB GPU:

1. **Use gradient checkpointing** (always enabled)
2. **Enable mixed precision** (FP16, enabled by default)
3. **Freeze feature extractor** (`--freeze_encoder`)
4. **Start with batch_size=8**, reduce to 4 if OOM
5. **Use max_duration=5.0**, reduce to 3.0 if OOM
6. **Use base models** (`wav2vec2-base`) not large variants

### Monitor GPU Usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## File Structure

```
.
├── transformer_emotion_recognition.py  # Main model implementation
├── train_transformer.py                # Training script
├── generate_hindi_csv.py               # Hindi dataset CSV generator
├── transformer_config.json             # Configuration presets
├── TRANSFORMER_README.md               # This file
│
├── data/
│   ├── csv/
│   │   ├── train_hindi.csv            # Hindi training data
│   │   ├── test_hindi.csv             # Hindi test data
│   │   ├── train_ravdess.csv          # RAVDESS training data
│   │   ├── test_ravdess.csv           # RAVDESS test data
│   │   ├── train_tess.csv             # TESS training data
│   │   └── test_tess.csv              # TESS test data
│   │
│   ├── hindi/                          # Hindi audio files
│   ├── ravdess/                        # RAVDESS audio files
│   └── tess/                           # TESS audio files
│
└── models/
    └── transformer_best.pt             # Saved model checkpoint
```

## Troubleshooting

### CUDA Out of Memory Error

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch_size 4`
2. Reduce audio duration: `--max_duration 3.0`
3. Ensure mixed precision is enabled (default)
4. Freeze encoder: `--freeze_encoder`

### Model Takes Too Long to Train

**Solutions:**
1. Reduce epochs: `--epochs 10`
2. Use smaller dataset
3. Increase batch size (if memory allows): `--batch_size 16`

### Poor Accuracy

**Solutions:**
1. Train for more epochs: `--epochs 30`
2. Add more training data
3. Try full fine-tuning (remove `--freeze_encoder`)
4. Adjust learning rate: `--learning_rate 5e-5`
5. Hindi dataset is included by default for more diversity (use `--exclude_hindi` if not needed)

## Comparison: Traditional vs Transformer

| Aspect | Traditional ML | Transformer (This) |
|--------|----------------|-------------------|
| **Features** | Hand-crafted (MFCC, etc.) | Learned representations |
| **Accuracy** | ~60-70% | ~75-85% |
| **Training Time** | Minutes | Hours |
| **GPU Required** | No | Yes (6GB+) |
| **Multilingual** | Limited | Excellent |
| **Transfer Learning** | No | Yes (pretrained) |
| **Novel Data** | Retraining needed | Fine-tune quickly |

## Citation

If using this work, please cite:

```
Wav2Vec2: facebook/wav2vec2-base
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

Same as the parent repository.

## Author

Shirssack

For questions or issues, please open a GitHub issue.
