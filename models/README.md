# Models Directory

This directory stores trained model checkpoints.

## Model Types

### Traditional ML Models (.pkl files)
- `emotion_model.pkl` - Default sklearn model (MLP)
- `ravdess_tess_model.pkl` - Trained on RAVDESS + TESS
- Custom named models from training scripts

### Deep Learning Models
- `best_deep_model.keras` - Best LSTM/GRU model during training
- `deep_emotion_model.h5` - Saved Keras model
- `deep_emotion_model.json` - Model architecture

### Transformer Models (.pt files)
- `transformer_best.pt` - Best transformer checkpoint
- `multilingual_best.pt` - Multilingual (English + Hindi) model
- Custom named transformer models

## Loading Models

### Traditional ML:
```python
from emotion_recognition import EmotionRecognizer
rec = EmotionRecognizer(emotions=['sad', 'neutral', 'happy'])
rec.load_model('models/emotion_model.pkl')
```

### Deep Learning:
```python
from deep_emotion_recognition import DeepEmotionRecognizer
rec = DeepEmotionRecognizer(emotions=['sad', 'neutral', 'happy'])
rec.load_model('models/deep_emotion_model')
```

### Transformer:
```python
from transformer_emotion_recognition import TransformerEmotionRecognizer
rec = TransformerEmotionRecognizer(emotions=['sad', 'neutral', 'happy'])
rec.load_model('models/transformer_best.pt')
```

## Model Sizes

- Traditional ML: 1-10 MB
- Deep Learning: 50-200 MB
- Transformer: 300-500 MB (base), 1-2 GB (large)
