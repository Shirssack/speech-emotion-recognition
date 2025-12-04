"""
train_traditional.py - Train traditional ML models
Quick training script for SVM, MLP, Random Forest, etc.
"""

from emotion_recognition import EmotionRecognizer
from data_extractor import load_data
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

print("="*70)
print("TRADITIONAL ML MODEL TRAINING")
print("="*70)

# Configuration - Edit these parameters
EMOTIONS = ['sad', 'neutral', 'happy', 'angry']
MODEL_TYPE = 'MLP'  # Options: MLP, SVM, RandomForest, GradientBoosting, KNN

print(f"\nConfiguration:")
print(f"  Model: {MODEL_TYPE}")
print(f"  Emotions: {EMOTIONS}")

# Create the sklearn model based on type
print(f"\n[1/5] Creating {MODEL_TYPE} model...")
if MODEL_TYPE == 'MLP':
    model = MLPClassifier(
        hidden_layer_sizes=(300,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=42,
        verbose=True
    )
elif MODEL_TYPE == 'SVM':
    model = SVC(
        kernel='rbf',
        gamma='scale',
        C=1.0,
        random_state=42,
        verbose=True
    )
elif MODEL_TYPE == 'RandomForest':
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
elif MODEL_TYPE == 'GradientBoosting':
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=1
    )
elif MODEL_TYPE == 'KNN':
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Load data from 4-class CSV files
print(f"\n[2/5] Loading and extracting features...")
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

# Create recognizer with the model
print(f"\n[3/5] Creating recognizer with {MODEL_TYPE}...")
rec = EmotionRecognizer(
    model=model,
    emotions=EMOTIONS,
    use_ravdess=False,  # Disable auto-loading since we loaded manually
    use_tess=False,
    use_hindi=False,
    balance=True,
    verbose=1
)

# Manually set the loaded data
rec.X_train = X_train
rec.y_train = y_train
rec.X_test = X_test
rec.y_test = y_test

# Scale the data
rec.X_train = rec.scaler.fit_transform(rec.X_train)
rec.X_test = rec.scaler.transform(rec.X_test)

# Train
print(f"\n[4/5] Training {MODEL_TYPE} model...")
rec.train()

# Evaluate
print(f"\n[5/5] Evaluating model...")
train_acc = rec.train_score()
test_acc = rec.test_score()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Training Accuracy: {train_acc:.2%}")
print(f"  Test Accuracy: {test_acc:.2%}")

# Confusion matrix
print("\n[Confusion Matrix]")
cm = rec.confusion_matrix()
print(cm)

# Classification report
print("\n[Classification Report]")
from sklearn.metrics import classification_report, accuracy_score
y_pred = rec.predict_batch(rec.X_test)
print(classification_report(rec.y_test, y_pred, target_names=EMOTIONS))

# Save model
os.makedirs('models', exist_ok=True)
model_path = f'models/{MODEL_TYPE.lower()}_4emotions.pkl'
rec.save_model(model_path)
print(f"\nModel saved to: {model_path}")

print("\n" + "="*70)
print("USAGE - Making predictions:")
print("="*70)
print(f"from emotion_recognition import EmotionRecognizer")
print(f"")
print(f"rec = EmotionRecognizer(emotions={EMOTIONS})")
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
