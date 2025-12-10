# Dataset Setup Guide

This guide will help you download and set up the required datasets for Speech Emotion Recognition.

## Overview

The repository includes CSV files that reference the datasets, but **you need to download the actual audio files separately** due to size and licensing constraints.

### Required Datasets

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
2. **TESS** (Toronto Emotional Speech Set)
3. **Hindi Emotion Dataset** (Optional, for multilingual support)

---

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the dataset setup helper script
python setup_datasets.py
```

This script will:
- Check which datasets are missing
- Provide download links
- Validate dataset structure
- Generate CSV files

### Option 2: Manual Setup

Follow the detailed instructions below for each dataset.

---

## Dataset Download Instructions

### 1. RAVDESS Dataset

**Download Link**: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

**Steps:**

1. Visit the Zenodo link above
2. Download **Audio-only files** (not Video+Audio)
   - File: `Audio_Speech_Actors_01-24.zip` (~500MB)
3. Extract the ZIP file
4. You should get folders named `Actor_01` through `Actor_24`
5. Move all `Actor_*` folders to `data/ravdess/` in this project

**Expected Structure:**
```
data/
└── ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   ├── 03-01-01-01-01-02-01.wav
    │   └── ...
    ├── Actor_02/
    ├── ...
    └── Actor_24/
```

**Filename Format**: `03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`

- Emotion codes: 01=neutral, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprised

---

### 2. TESS Dataset

**Download Link**: [https://doi.org/10.5683/SP2/E8H2MF](https://doi.org/10.5683/SP2/E8H2MF)

or search for "TESS Toronto emotional speech set dataverse"

**Steps:**

1. Visit the dataverse link
2. Download all emotion category folders:
   - OAF_angry.zip
   - OAF_disgust.zip
   - OAF_Fear.zip
   - OAF_happy.zip
   - OAF_neutral.zip
   - OAF_Pleasant_surprise.zip
   - OAF_Sad.zip
   - YAF_angry.zip
   - YAF_disgust.zip
   - YAF_fear.zip
   - YAF_happy.zip
   - YAF_neutral.zip
   - YAF_pleasant_surprised.zip
   - YAF_sad.zip

3. Extract all ZIP files
4. Move all folders to `data/tess/`

**Expected Structure:**
```
data/
└── tess/
    ├── OAF_angry/
    │   ├── OAF_back_angry.wav
    │   └── ...
    ├── OAF_disgust/
    ├── OAF_Fear/
    ├── OAF_happy/
    ├── OAF_neutral/
    ├── OAF_Pleasant_surprise/
    ├── OAF_Sad/
    ├── YAF_angry/
    ├── YAF_disgust/
    ├── YAF_fear/
    ├── YAF_happy/
    ├── YAF_neutral/
    ├── YAF_pleasant_surprised/
    └── YAF_sad/
```

**Note**: OAF = Older Adult Female, YAF = Younger Adult Female

---

### 3. Hindi Emotion Dataset (Optional)

**Note**: The Hindi dataset may require academic access or direct contact with researchers.

If you have access to the Hindi emotion dataset:

1. Place audio files in `data/hindi/my Dataset/`
2. Expected structure with folders: `anger/`, `fear/`, `happy/`, `neutral/`, `sad/`, etc.

**Expected Structure:**
```
data/
└── hindi/
    └── my Dataset/
        ├── 1/
        │   └── session1/
        │       ├── anger/
        │       ├── happy/
        │       ├── neutral/
        │       └── sad/
        └── ...
```

**Skip Hindi dataset** if not available - the model works fine with just RAVDESS and TESS.

---

## Verification

After downloading datasets, verify the setup:

```bash
# Check dataset structure
python validate_datasets.py

# Quick test to ensure everything works
python run_test.py
```

---

## Dataset Statistics

After proper setup, you should have:

| Dataset | Files | Emotions | Size |
|---------|-------|----------|------|
| RAVDESS | 1,440 | 8 emotions (7 + neutral) | ~500MB |
| TESS | 2,800 | 7 emotions + neutral | ~400MB |
| Hindi | ~1,600 | 8 emotions | ~300MB |

For **4-emotion classification** (sad, neutral, happy, angry), the system uses:
- RAVDESS: 672 files
- TESS: 1,600 files
- Hindi: 1,599 files (optional)
- **Total: 3,871 files**

---

## Troubleshooting

### Issue: "No dataset found" error

**Cause**: Audio files are not in the correct location.

**Solution**:
1. Run `python validate_datasets.py` to check what's missing
2. Verify folder structure matches the expected structure above
3. Ensure folder names are exact (case-sensitive on Linux/Mac)

### Issue: "FileNotFoundError" during training

**Cause**: CSV files reference paths that don't exist.

**Solution**:
1. Delete old CSV files: `rm -rf data/csv/*.csv`
2. Regenerate CSVs: `python generate_4class_csv.py`
3. Run validation: `python validate_datasets.py`

### Issue: Wrong folder names

**Common mistakes**:
- `data/RAVDESS/` instead of `data/ravdess/` (case matters on Linux/Mac)
- `data/emo-db/` instead of `data/emodb/`
- Missing intermediate folders (e.g., files directly in `data/tess/` instead of `data/tess/OAF_angry/`)

**Solution**: Carefully match the expected structure shown above.

### Issue: CSV files already exist but point to wrong paths

**Solution**:
```bash
# Backup existing CSVs
mkdir -p data/csv_backup
mv data/csv/*.csv data/csv_backup/

# Regenerate CSVs
python generate_4class_csv.py

# Verify
python validate_datasets.py
```

---

## Alternative: Use Pre-extracted Features

If downloading datasets is not possible, you can:

1. Use pre-trained models (if available in `models/` folder)
2. Train on a smaller custom dataset
3. Use the model for inference only (requires pre-trained weights)

---

## Training Without All Datasets

You can train with a subset of datasets by modifying the training script:

```python
from emotion_recognition import EmotionRecognizer

# Use only RAVDESS
rec = EmotionRecognizer(
    use_ravdess=True,
    use_tess=False,
    use_hindi=False
)

# Or only TESS
rec = EmotionRecognizer(
    use_ravdess=False,
    use_tess=True,
    use_hindi=False
)
```

Or via command line:

```bash
# Exclude specific datasets
python train_transformer.py --exclude_hindi --exclude_tess
```

---

## License and Citation

### RAVDESS
- **Citation**: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
- **License**: Creative Commons Attribution (CC BY-NC-SA 4.0)
- **Link**: https://zenodo.org/record/1188976

### TESS
- **Citation**: Pichora-Fuller, M. K., & Dupuis, K. (2020). Toronto emotional speech set (TESS). University of Toronto, Psychology Department.
- **License**: Available for research purposes
- **Link**: https://doi.org/10.5683/SP2/E8H2MF

### Important
When using these datasets:
1. Cite the original papers
2. Follow their respective licenses
3. Use for research/educational purposes only
4. Do not redistribute the datasets

---

## Next Steps

After setting up datasets:

1. **Verify installation**: `python validate_datasets.py`
2. **Quick test**: `python run_test.py`
3. **Train a model**:
   - Transformer: `python train_transformer.py --epochs 15`
   - Traditional ML: `python train_traditional_ml.py`
4. **Make predictions**: `python demo_transformer.py --mode single --audio_file your_audio.wav`

---

## Need Help?

- Check existing issues: https://github.com/yourusername/speech-emotion-recognition/issues
- Read the main README: [README.md](README.md)
- For dataset-specific questions, contact the original dataset authors

---

**Last Updated**: December 2024
