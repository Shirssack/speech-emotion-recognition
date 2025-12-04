"""
train_deep_learning.py - Train deep learning models (LSTM/GRU)
Quick training script for LSTM and GRU neural networks.
"""

from deep_emotion_recognition import DeepEmotionRecognizer
from data_extractor import load_data
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

print("="*70)
print("DEEP LEARNING MODEL TRAINING")
print("="*70)

# Configuration - Edit these parameters
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'LSTM'  # Options: LSTM, GRU

# Model architecture
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Training parameters
EPOCHS = 100
BATCH_SIZE = 64

print(f"\nConfiguration:")
print(f"  Model: {MODEL_TYPE}")
print(f"  Emotions: {EMOTIONS}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Num Layers: {NUM_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")

# Load data from 4-class CSV files
print("\n[1/4] Loading and extracting features...")
X_train, y_train, X_test, y_test = load_data(
    train_csv=['data/csv/train_ravdess_4class.csv',
               'data/csv/train_tess_4class.csv',
               'data/csv/train_hindi_4class.csv'],
    test_csv=['data/csv/test_ravdess_4class.csv',
              'data/csv/test_tess_4class.csv',
              'data/csv/test_hindi_4class.csv'],
    emotions=EMOTIONS,
    balance=True,
    verbose=1
)

print(f"\n  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Feature dimensions: {X_train.shape[1]}")

# Create recognizer (with automatic data loading disabled)
print(f"\n[2/4] Creating {MODEL_TYPE} recognizer...")
rec = DeepEmotionRecognizer(
    emotions=EMOTIONS,
    n_rnn_layers=NUM_LAYERS,
    rnn_units=HIDDEN_SIZE,
    dropout=DROPOUT,
    cell_type=MODEL_TYPE,
    use_ravdess=False,  # Disable auto-loading since we loaded manually
    use_tess=False,
    use_hindi=False,
    verbose=1
)

# Manually set the loaded data to the recognizer
# The recognizer's _prepare_data() didn't load anything because all use_ flags are False
# So we need to manually assign the data
rec.X_train = X_train
rec.y_train = y_train
rec.X_test = X_test
rec.y_test = y_test

# Scale the data
rec.X_train = rec.scaler.fit_transform(rec.X_train)
rec.X_test = rec.scaler.transform(rec.X_test)

# Rebuild the model with correct input shape
rec._build_model()

# Train
print(f"\n[3/4] Training {MODEL_TYPE} model...")
print(f"  This may take 5-15 minutes depending on your hardware...\n")

history = rec.train(epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# Evaluate
print(f"\n[4/4] Evaluating model...")
test_acc = rec.test_score()
train_acc = rec.train_score()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Training Accuracy: {train_acc:.2%}")
print(f"  Test Accuracy: {test_acc:.2%}")

# Get predictions for confusion matrix
X_test_rnn = rec._prepare_data_for_rnn(rec.X_test)
y_pred_proba = rec.model.predict(X_test_rnn, verbose=0)
y_pred = rec._decode_labels(y_pred_proba)

# Confusion matrix
print("\n[Confusion Matrix]")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rec.y_test, y_pred, labels=EMOTIONS)

print(f"\n{'':>12}", end='')
for emotion in EMOTIONS:
    print(f"{emotion:>10}", end='')
print()
print("-"*70)
for i, emotion in enumerate(EMOTIONS):
    print(f"{emotion:>12}", end='')
    for j in range(len(EMOTIONS)):
        print(f"{cm[i][j]:>10}", end='')
    print()

# Classification report
print("\n[Classification Report]")
from sklearn.metrics import classification_report
print(classification_report(rec.y_test, y_pred, target_names=EMOTIONS))

# Save model
os.makedirs('models', exist_ok=True)
model_path = f'models/{MODEL_TYPE.lower()}_4emotions'
rec.save_model(model_path)
print(f"\nModel saved to: {model_path}.keras")

print("\n" + "="*70)
print("USAGE - Making predictions:")
print("="*70)
print(f"from deep_emotion_recognition import DeepEmotionRecognizer")
print(f"")
print(f"rec = DeepEmotionRecognizer(emotions={EMOTIONS})")
print(f"rec.load_model('{model_path}')")
print(f"")
print(f"# Single prediction")
print(f"emotion = rec.predict('path/to/audio.wav')")
print(f"print(f'Emotion: {{emotion}}')")
print(f"")
print(f"# Prediction with probabilities")
print(f"probs = rec.predict_proba('path/to/audio.wav')")
print(f"for emotion, prob in probs.items():")
print(f"    print(f'{{emotion}}: {{prob:.2%}}')")
print("="*70)
