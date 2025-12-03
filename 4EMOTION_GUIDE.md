# 4-Emotion Configuration Guide

The codebase has been updated to use **4 emotions by default**: `sad`, `neutral`, `happy`, `angry`

## Changes Made

### Default Emotions Updated
All main scripts now default to 4 emotions instead of 3:
- ✅ `demo_transformer.py` - DEFAULT_EMOTIONS
- ✅ `train_transformer.py` - Default --emotions argument
- ✅ `run_test.py` - Emotion initialization
- ✅ `grid_search.py` - Default emotions parameter

### Default CSV Files Updated
Training scripts now use 4-class CSV files by default:
- ✅ `train_transformer.py` uses `*_4class.csv` files
- ✅ Hindi integration uses `*_4class.csv` files

## Quick Start with 4 Emotions

### Train Transformer Model (English only)
```bash
python train_transformer.py --epochs 15 --batch_size 8
```

This automatically uses:
- Emotions: `sad, neutral, happy, angry`
- Training: `train_ravdess_4class.csv`, `train_tess_4class.csv`
- Testing: `test_ravdess_4class.csv`, `test_tess_4class.csv`

### Train with Hindi (Multilingual)
```bash
python train_transformer.py --include_hindi --epochs 20 --batch_size 8
```

### Demo/Prediction
```bash
# Simple run with defaults
python demo_transformer.py

# Single file prediction
python demo_transformer.py --mode single --audio_file path/to/audio.wav

# Batch prediction
python demo_transformer.py --mode batch --audio_dir data/ravdess --limit 20
```

### Grid Search
```bash
python grid_search.py --model MLP --fast
```

## Available Emotion Configurations

### 4-Emotion (Default - NEW!)
```python
emotions = ['sad', 'neutral', 'happy', 'angry']
# CSV files: *_4class.csv
```
- **Total samples**: 3,871 (RAVDESS + TESS + Hindi)
- **Balanced**: Yes
- **Recommended for**: General emotion recognition

### 3-Emotion (Legacy)
```python
emotions = ['sad', 'neutral', 'happy']
# CSV files: train_ravdess.csv, train_tess.csv
```
To use 3 emotions:
```bash
python train_transformer.py \
    --emotions sad neutral happy \
    --train_csv data/csv/train_ravdess.csv data/csv/train_tess.csv \
    --test_csv data/csv/test_ravdess.csv data/csv/test_tess.csv
```

### 6-Emotion (Extended)
```python
emotions = ['sad', 'neutral', 'happy', 'angry', 'fear', 'disgust']
```
Generate CSV files:
```bash
python generate_6class_csv.py  # You can create this for 6 emotions
```

## Dataset Statistics (4-Emotion)

| Dataset | Total | Sad | Neutral | Happy | Angry |
|---------|-------|-----|---------|-------|-------|
| RAVDESS | 672 | 192 | 96 | 192 | 192 |
| TESS | 1,600 | 400 | 400 | 400 | 400 |
| Hindi | 1,599 | 400 | 400 | 399 | 400 |
| **TOTAL** | **3,871** | **992** | **896** | **991** | **992** |

## Migration Guide

### If you have existing 3-emotion models:

**Option 1: Continue using 3 emotions**
```bash
python demo_transformer.py --emotions sad neutral happy
```

**Option 2: Retrain with 4 emotions**
```bash
python train_transformer.py --epochs 15
```

### If you want to evaluate existing models:

**3-emotion model:**
```python
rec = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy'])
rec.load_model('models/old_3emotion_model.pt')
```

**4-emotion model:**
```python
rec = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy', 'angry'])
rec.load_model('models/transformer_best.pt')
```

## Benefits of 4 Emotions

1. **Better Coverage**: Anger is a common emotion in real-world scenarios
2. **More Practical**: 4 emotions cover most use cases
3. **Still Balanced**: All emotions well-represented in training data
4. **Multilingual**: Hindi dataset included with same 4 emotions
5. **Higher Utility**: More useful than 3 emotions, easier than 6+

## Expected Performance

| Configuration | Expected Accuracy | F1 Score |
|---------------|------------------|----------|
| 3-emotion | 75-85% | 0.75-0.85 |
| 4-emotion | 70-80% | 0.70-0.80 |
| 6-emotion | 65-75% | 0.65-0.75 |

*More emotions = slightly lower accuracy but more practical utility*

## Files and Locations

### CSV Files (4-emotion)
```
data/csv/
├── train_ravdess_4class.csv    (535 samples)
├── test_ravdess_4class.csv     (137 samples)
├── train_tess_4class.csv       (1,280 samples)
├── test_tess_4class.csv        (320 samples)
├── train_hindi_4class.csv      (1,279 samples)
└── test_hindi_4class.csv       (320 samples)
```

### Scripts
- `train_transformer.py` - Train transformer models
- `demo_transformer.py` - Run predictions
- `generate_4class_csv.py` - Generate 4-emotion CSV files
- `run_test.py` - Quick setup test
- `grid_search.py` - Hyperparameter tuning

## Customization

To use different emotions, simply specify them:
```bash
python train_transformer.py --emotions emotion1 emotion2 emotion3
```

Available emotions from datasets:
- RAVDESS: neutral, happy, sad, angry, fear, disgust, surprised
- TESS: neutral, happy, sad, angry, fear, disgust, surprised
- Hindi: neutral, happy, sad, angry, fear, disgust, surprised, sarcastic

---

**Questions?** Check `TRANSFORMER_README.md` for detailed documentation.
