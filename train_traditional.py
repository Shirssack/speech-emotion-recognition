"""
train_traditional.py - Train traditional ML models
Quick training script for SVM, MLP, Random Forest, etc.
"""

from emotion_recognition import EmotionRecognizer
from data_extractor import load_data

print("="*70)
print("TRADITIONAL ML MODEL TRAINING")
print("="*70)

# Configuration
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'MLP'  # Options: MLP, SVM, RandomForest, GradientBoosting, KNN

print(f"\nConfiguration:")
print(f"  Model: {MODEL_TYPE}")
print(f"  Emotions: {EMOTIONS}")

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

# Create recognizer
print(f"\n[2/4] Creating {MODEL_TYPE} recognizer...")
rec = EmotionRecognizer(
    model=MODEL_TYPE,
    emotions=EMOTIONS,
    balance=True
)

# Train
print(f"\n[3/4] Training {MODEL_TYPE} model...")
rec.train(X_train, y_train)

# Evaluate
print(f"\n[4/4] Evaluating model...")
train_acc = rec.score(X_train, y_train)
test_acc = rec.score(X_test, y_test)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Training Accuracy:   {train_acc:.2%}")
print(f"  Test Accuracy:       {test_acc:.2%}")

# Confusion matrix
print("\n[Confusion Matrix]")
cm = rec.confusion_matrix(y_test, X_test)
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

# Save model
model_path = f'models/{MODEL_TYPE.lower()}_4emotions.pkl'
rec.save_model(model_path)
print(f"\n✓ Model saved to: {model_path}")

print("\n" + "="*70)
print("To make predictions:")
print(f"  from emotion_recognition import EmotionRecognizer")
print(f"  rec = EmotionRecognizer(emotions={EMOTIONS})")
print(f"  rec.load_model('{model_path}')")
print(f"  emotion = rec.predict('path/to/audio.wav')")
print("="*70)
