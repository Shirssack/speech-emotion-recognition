# Features Directory

This directory stores cached feature extractions from audio files to speed up subsequent training runs.

## Files Generated

When you run training with traditional ML models, feature extraction caching will create files like:
- `train_mfcc-chroma-mel_HNS.npz` - Cached training features
- `test_mfcc-chroma-mel_HNS.npz` - Cached test features

The filename format indicates:
- Feature types used (mfcc, chroma, mel, contrast, tonnetz)
- Emotion classes (first letters)

## Cache Benefits

- **First run**: Extracts and saves features (~2-5 minutes for full dataset)
- **Subsequent runs**: Loads cached features (~5-10 seconds)
- **Space**: Each cache file is typically 10-50 MB

## Clearing Cache

To force re-extraction (e.g., after changing audio config):
```bash
rm features/*.npz
```
