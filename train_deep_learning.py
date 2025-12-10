"""
train_deep_learning.py - Train deep learning models (LSTM/GRU)
Standalone training script that doesn't rely on DeepEmotionRecognizer class.
"""

from data_extractor import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import os
import pickle

print("="*70)
print("DEEP LEARNING MODEL TRAINING")
print("="*70)

# Configuration - Edit these parameters
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'LSTM'  # Options: LSTM, GRU

# Model architecture
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DENSE_LAYERS = 2
DENSE_UNITS = 128

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
print("\n[1/5] Loading and extracting features...")
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

# Scale data
print("\n[2/5] Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for RNN (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"  Reshaped to: {X_train.shape} (samples, timesteps, features)")

# Encode labels
emotion_to_int = {e: i for i, e in enumerate(EMOTIONS)}
int_to_emotion = {i: e for e, i in emotion_to_int.items()}

y_train_int = np.array([emotion_to_int[label] for label in y_train])
y_test_int = np.array([emotion_to_int[label] for label in y_test])

y_train_encoded = to_categorical(y_train_int, num_classes=len(EMOTIONS))
y_test_encoded = to_categorical(y_test_int, num_classes=len(EMOTIONS))

# Build model
print(f"\n[3/5] Building {MODEL_TYPE} model...")

# Select RNN cell type
if MODEL_TYPE == 'LSTM':
    RNN_Cell = LSTM
elif MODEL_TYPE == 'GRU':
    RNN_Cell = GRU
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model = Sequential()

# RNN Layers
for i in range(NUM_LAYERS):
    if i == 0:
        model.add(RNN_Cell(
            HIDDEN_SIZE,
            return_sequences=(i < NUM_LAYERS - 1),
            input_shape=(1, X_train.shape[2])
        ))
    else:
        model.add(RNN_Cell(
            HIDDEN_SIZE,
            return_sequences=(i < NUM_LAYERS - 1)
        ))

    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))

# Dense Layers
for i in range(DENSE_LAYERS):
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))

# Output Layer
model.add(Dense(len(EMOTIONS), activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n{MODEL_TYPE} Model Summary:")
model.summary()

# Prepare callbacks
os.makedirs('models', exist_ok=True)
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_deep_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print(f"\n[4/5] Training {MODEL_TYPE} model...")
print(f"  This may take 10-30 minutes depending on your hardware...\n")

history = model.fit(
    X_train,
    y_train_encoded,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print(f"\n[5/5] Evaluating model...")

_, train_acc = model.evaluate(X_train, y_train_encoded, verbose=0)
_, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Training Accuracy: {train_acc:.2%}")
print(f"  Test Accuracy: {test_acc:.2%}")

# Check for overfitting
print("\n[Overfitting Check]")
gap = train_acc - test_acc
if gap > 0.15:
    print(f"⚠ WARNING: Possible overfitting detected!")
    print(f"   Train-Test gap: {gap:.2%}")
    print(f"   Consider: increasing dropout, reducing model size, or training longer")
elif gap > 0.10:
    print(f"ℹ Moderate train-test gap: {gap:.2%}")
    print(f"   Model may be slightly overfitting")
else:
    print(f"✓ Good generalization (gap: {gap:.2%})")

# Get predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred_int = np.argmax(y_pred_proba, axis=1)
y_pred = [int_to_emotion[i] for i in y_pred_int]

# Confusion matrix
print("\n[Confusion Matrix]")
cm = confusion_matrix(y_test, y_pred, labels=EMOTIONS)

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
print(classification_report(y_test, y_pred, target_names=EMOTIONS))

# Save model
model_path = f'models/{MODEL_TYPE.lower()}_4emotions.keras'
config_path = f'models/{MODEL_TYPE.lower()}_4emotions_config.pkl'

model.save(model_path)

# Save configuration
with open(config_path, 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'emotions': EMOTIONS,
        'emotion_to_int': emotion_to_int,
        'int_to_emotion': int_to_emotion,
        'model_type': MODEL_TYPE
    }, f)

print(f"\nModel saved to: {model_path}")
print(f"Config saved to: {config_path}")

print("\n" + "="*70)
print("USAGE - Making predictions:")
print("="*70)
print(f"from tensorflow.keras.models import load_model")
print(f"from utils import extract_feature")
print(f"import pickle")
print(f"import numpy as np")
print(f"")
print(f"# Load model and config")
print(f"model = load_model('{model_path}')")
print(f"with open('{config_path}', 'rb') as f:")
print(f"    config = pickle.load(f)")
print(f"")
print(f"# Extract features from audio")
print(f"features = extract_feature('path/to/audio.wav',")
print(f"                          mfcc=True, chroma=True, mel=True)")
print(f"")
print(f"# Scale and reshape")
print(f"features_scaled = config['scaler'].transform([features])")
print(f"features_rnn = features_scaled.reshape((1, 1, features_scaled.shape[1]))")
print(f"")
print(f"# Predict")
print(f"proba = model.predict(features_rnn, verbose=0)[0]")
print(f"emotion_idx = np.argmax(proba)")
print(f"emotion = config['int_to_emotion'][emotion_idx]")
print(f"confidence = proba[emotion_idx]")
print(f"")
print(f"print(f'Emotion: {{emotion}} ({{confidence:.2%}} confidence)')")
print("="*70)
