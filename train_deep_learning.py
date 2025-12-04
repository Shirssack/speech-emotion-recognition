"""
train_deep_learning.py - Train deep learning models (LSTM/GRU)
Quick training script for LSTM and GRU neural networks.
"""

from deep_emotion_recognition import DeepEmotionRecognizer
from data_extractor import load_data
import numpy as np

print("="*70)
print("DEEP LEARNING MODEL TRAINING")
print("="*70)

# Configuration
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'LSTM'  # Options: LSTM, GRU

# Model hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.2
HIDDEN_SIZE = 128
NUM_LAYERS = 2

print(f"\nConfiguration:")
print(f"  Model: {MODEL_TYPE}")
print(f"  Emotions: {EMOTIONS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Num Layers: {NUM_LAYERS}")
print(f"  Dropout: {DROPOUT}")

# Load data
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

# Reshape for LSTM/GRU (samples, timesteps, features)
# We'll treat each feature as a timestep for sequence modeling
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"  Reshaped to: {X_train.shape} (samples, timesteps, features)")

# Create recognizer
print(f"\n[2/4] Creating {MODEL_TYPE} recognizer...")
rec = DeepEmotionRecognizer(
    emotions=EMOTIONS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    dropout=DROPOUT,
    n_rnn_layers=NUM_LAYERS,
    cell=MODEL_TYPE
)

# Train
print(f"\n[3/4] Training {MODEL_TYPE} model...")
print(f"  This may take 5-15 minutes depending on your hardware...\n")

# Convert labels to numpy arrays if they aren't already
y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test

rec.train(X_train, y_train, X_test, y_test, verbose=1)

# Evaluate
print(f"\n[4/4] Evaluating model...")
train_acc = rec.test_score()  # Uses the stored test set from training
test_acc = train_acc  # Already evaluated on test set during training

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Test Accuracy: {test_acc:.2%}")

# Get predictions for confusion matrix
y_pred = rec.predict(X_test)

# Confusion matrix
print("\n[Confusion Matrix]")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

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
print(classification_report(y_test, y_pred, target_names=EMOTIONS))

# Save model
model_path = f'models/{MODEL_TYPE.lower()}_4emotions.h5'
rec.save(model_path)
print(f"\n✓ Model saved to: {model_path}")

print("\n" + "="*70)
print("To make predictions:")
print(f"  from deep_emotion_recognition import DeepEmotionRecognizer")
print(f"  rec = DeepEmotionRecognizer(emotions={EMOTIONS})")
print(f"  rec.load('{model_path}')")
print(f"  emotion = rec.predict('path/to/audio.wav')")
print("="*70)
