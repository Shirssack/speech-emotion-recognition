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
        alpha=0.001,  # Increased regularization (was 0.0001)
        max_iter=500,
        early_stopping=True,  # Stop when validation score stops improving
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
elif MODEL_TYPE == 'SVM':
    model = SVC(
        kernel='rbf',
        gamma='scale',
        C=1.0,  # Regularization parameter (lower = more regularization)
        probability=True,  # Enable probability predictions
        random_state=42,
        verbose=True
    )
elif MODEL_TYPE == 'RandomForest':
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,  # Limit tree depth to reduce overfitting
        min_samples_split=5,  # Increased from 2
        min_samples_leaf=2,  # Minimum samples in leaf nodes
        max_features='sqrt',  # Reduce features per split
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
elif MODEL_TYPE == 'GradientBoosting':
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,  # Increased regularization
        subsample=0.8,  # Use 80% of samples for each tree
        random_state=42,
        verbose=1
    )
elif MODEL_TYPE == 'KNN':
    model = KNeighborsClassifier(
        n_neighbors=7,  # Increased from 5 for smoother decision boundary
        weights='distance',
        metric='manhattan',  # Try manhattan distance
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
try:
    # Try with disabled flags first
    rec = EmotionRecognizer(
        model=model,
        emotions=EMOTIONS,
        use_ravdess=False,
        use_tess=False,
        use_hindi=False,
        data_path='__nonexistent__',  # Use non-existent path to skip auto-loading
        balance=True,
        verbose=0  # Suppress "No dataset found" message
    )
except:
    # If that fails, create with default settings and override data
    rec = EmotionRecognizer(
        model=model,
        emotions=EMOTIONS,
        balance=True,
        verbose=0
    )

# Manually set the loaded data
rec.X_train = X_train
rec.y_train = y_train
rec.X_test = X_test
rec.y_test = y_test

# Scale the data
rec.X_train = rec.scaler.fit_transform(rec.X_train)
rec.X_test = rec.scaler.transform(rec.X_test)

print(f"  Recognizer created successfully with {MODEL_TYPE}")

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
y_pred = rec.model.predict(rec.X_test)  # Use model.predict directly
print(classification_report(rec.y_test, y_pred, target_names=EMOTIONS))

# Check for overfitting
print("\n[Overfitting Check]")
if train_acc - test_acc > 0.15:  # More than 15% gap
    print(f"⚠ WARNING: Possible overfitting detected!")
    print(f"   Train-Test gap: {(train_acc - test_acc):.2%}")
    print(f"   Consider: reducing model complexity or adding regularization")
elif train_acc - test_acc > 0.10:  # 10-15% gap
    print(f"ℹ Moderate train-test gap: {(train_acc - test_acc):.2%}")
    print(f"   Model may be slightly overfitting")
else:
    print(f"✓ Good generalization (gap: {(train_acc - test_acc):.2%})")

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
