# Quick Start Guide

## 🚨 First Time Setup

If you just cloned this repository and are seeing **"No dataset found"** errors, follow these steps:

### Step 1: Validate Your Setup

```bash
python validate_datasets.py
```

This will check what datasets you have and what's missing.

### Step 2: Interactive Setup Helper

```bash
python setup_datasets.py
```

This interactive tool will:
- Check which datasets are already set up
- Provide download links
- Open download pages in your browser
- Guide you through the setup process

### Step 3: Download Required Datasets

You need to download two main datasets:

#### RAVDESS (Required)
- **Link**: https://zenodo.org/record/1188976
- **File**: `Audio_Speech_Actors_01-24.zip` (~500MB)
- **Extract to**: `data/ravdess/Actor_01/`, `data/ravdess/Actor_02/`, etc.

#### TESS (Required)
- **Link**: https://doi.org/10.5683/SP2/E8H2MF
- **Files**: Download all emotion ZIP files (OAF_*, YAF_*)
- **Extract to**: `data/tess/OAF_angry/`, `data/tess/YAF_happy/`, etc.

#### Hindi (Optional)
- Contact: Research dataset, may require academic access
- **Extract to**: `data/hindi/`

### Step 4: Verify Installation

```bash
python validate_datasets.py
```

You should see:
```
✓ VALIDATION PASSED
All datasets are properly set up!
```

### Step 5: Start Training

#### Option A: Transformer Model (Best Accuracy)
```bash
python train_transformer.py --epochs 15 --batch_size 8
```

#### Option B: Traditional ML (Fast Training)
```bash
python run_test.py
```

---

## 📁 Expected Directory Structure

After downloading datasets, your project should look like:

```
speech-emotion-recognition/
├── data/
│   ├── ravdess/
│   │   ├── Actor_01/
│   │   │   ├── 03-01-01-01-01-01-01.wav
│   │   │   └── ...
│   │   ├── Actor_02/
│   │   └── ... (up to Actor_24)
│   ├── tess/
│   │   ├── OAF_angry/
│   │   │   ├── OAF_back_angry.wav
│   │   │   └── ...
│   │   ├── OAF_happy/
│   │   ├── YAF_angry/
│   │   └── ... (14 emotion folders total)
│   └── hindi/ (optional)
│       └── my Dataset/
│           └── ...
├── validate_datasets.py
├── setup_datasets.py
└── train_transformer.py
```

---

## ❓ Common Issues

### Issue: "No dataset found"

**Cause**: Audio files not downloaded or in wrong location

**Fix**:
```bash
python validate_datasets.py    # Check what's missing
python setup_datasets.py       # Get download links
```

### Issue: "FileNotFoundError" during training

**Cause**: CSV files pointing to non-existent audio files

**Fix**:
```bash
rm -rf data/csv/               # Delete old CSVs
python generate_4class_csv.py  # Regenerate CSVs
python validate_datasets.py    # Verify
```

### Issue: Folder naming problems

**Common mistakes**:
- `data/RAVDESS/` instead of `data/ravdess/` (case matters!)
- `data/emo-db/` instead of `data/emodb/`
- Files directly in `data/tess/` instead of `data/tess/OAF_angry/`

**Fix**: Check folder names carefully and match the expected structure above

---

## 🎯 What Each Tool Does

### `validate_datasets.py`
- ✓ Checks if datasets are downloaded
- ✓ Verifies folder structure
- ✓ Counts audio files
- ✓ Checks CSV integrity
- ✓ Provides helpful error messages

### `setup_datasets.py`
- ✓ Interactive setup wizard
- ✓ Opens download pages in browser
- ✓ Guides through each step
- ✓ Checks current status

### `DATASET_SETUP_GUIDE.md`
- ✓ Detailed download instructions
- ✓ Dataset descriptions and statistics
- ✓ Licensing information
- ✓ Troubleshooting guide

---

## 🚀 Quick Commands Reference

```bash
# Check what's missing
python validate_datasets.py

# Interactive setup
python setup_datasets.py

# Quick test (uses existing datasets)
python run_test.py

# Train transformer model
python train_transformer.py --epochs 15

# Make predictions
python demo_transformer.py --mode single --audio_file audio.wav

# Train without Hindi dataset
python train_transformer.py --exclude_hindi
```

---

## 📚 Need More Help?

- **Detailed Guide**: See [DATASET_SETUP_GUIDE.md](DATASET_SETUP_GUIDE.md)
- **Full Documentation**: See [README.md](README.md)
- **Transformer Details**: See [TRANSFORMER_README.md](TRANSFORMER_README.md)
- **Issues**: https://github.com/yourusername/speech-emotion-recognition/issues

---

## ⏱️ How Long Does Setup Take?

| Task | Time |
|------|------|
| Download RAVDESS | 5-10 minutes |
| Download TESS | 10-15 minutes |
| Extract and organize files | 5 minutes |
| Validation | < 1 minute |
| **Total Setup Time** | **~30 minutes** |

After setup, training times:
- **Transformer**: 1-3 hours (GPU required)
- **Traditional ML**: 1-5 minutes (CPU okay)

---

**Ready to start?** Run `python validate_datasets.py` now!
