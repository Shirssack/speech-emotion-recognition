"""
train_traditional.py - Train traditional ML models
Standalone training script for SVM, MLP, Random Forest, etc.
"""

from data_extractor import load_data
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import pickle

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

print(f"  {MODEL_TYPE} model created successfully")

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

# Scale the data
print(f"\n[3/5] Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode labels to integers (required for early_stopping and some models)
print(f"  Encoding labels...")
emotion_to_int = {e: i for i, e in enumerate(EMOTIONS)}
int_to_emotion = {i: e for e, i in emotion_to_int.items()}

y_train_encoded = [emotion_to_int[label] for label in y_train]
y_test_encoded = [emotion_to_int[label] for label in y_test]

# Train
print(f"\n[4/5] Training {MODEL_TYPE} model...")
model.fit(X_train, y_train_encoded)

# Evaluate
print(f"\n[5/5] Evaluating model...")
train_acc = accuracy_score(y_train_encoded, model.predict(X_train))
test_acc = accuracy_score(y_test_encoded, model.predict(X_test))

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Training Accuracy: {train_acc:.2%}")
print(f"  Test Accuracy: {test_acc:.2%}")

# Check for overfitting
print("\n[Overfitting Check]")
gap = train_acc - test_acc
if gap > 0.15:  # More than 15% gap
    print(f"⚠ WARNING: Possible overfitting detected!")
    print(f"   Train-Test gap: {gap:.2%}")
    print(f"   Consider: reducing model complexity or adding regularization")
elif gap > 0.10:  # 10-15% gap
    print(f"ℹ Moderate train-test gap: {gap:.2%}")
    print(f"   Model may be slightly overfitting")
else:
    print(f"✓ Good generalization (gap: {gap:.2%})")

# Get predictions (decode integers back to emotion strings)
y_pred_encoded = model.predict(X_test)
y_pred = [int_to_emotion[pred] for pred in y_pred_encoded]
y_test_labels = [int_to_emotion[label] for label in y_test_encoded]

# Confusion matrix
print("\n[Confusion Matrix]")
cm = confusion_matrix(y_test_labels, y_pred, labels=EMOTIONS)

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
print(classification_report(y_test_labels, y_pred, target_names=EMOTIONS))

# Save model
os.makedirs('models', exist_ok=True)
model_path = f'models/{MODEL_TYPE.lower()}_4emotions.pkl'

with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'emotions': EMOTIONS,
        'emotion_to_int': emotion_to_int,
        'int_to_emotion': int_to_emotion
    }, f)

print(f"\nModel saved to: {model_path}")

print("\n" + "="*70)
print("USAGE - Making predictions:")
print("="*70)
print(f"from utils import extract_feature")
print(f"import pickle")
print(f"")
print(f"# Load model")
print(f"with open('{model_path}', 'rb') as f:")
print(f"    data = pickle.load(f)")
print(f"    model = data['model']")
print(f"    scaler = data['scaler']")
print(f"    emotions = data['emotions']")
print(f"    int_to_emotion = data['int_to_emotion']")
print(f"")
print(f"# Extract features from audio")
print(f"features = extract_feature('path/to/audio.wav',")
print(f"                          mfcc=True, chroma=True, mel=True)")
print(f"")
print(f"# Scale and predict")
print(f"features_scaled = scaler.transform([features])")
print(f"emotion_idx = model.predict(features_scaled)[0]")
print(f"emotion = int_to_emotion[emotion_idx]")
print(f"")
print(f"print(f'Emotion: {{emotion}}')")
print(f"")
print(f"# With probabilities (if model supports it)")
print(f"if hasattr(model, 'predict_proba'):")
print(f"    proba = model.predict_proba(features_scaled)[0]")
print(f"    for idx, prob in enumerate(proba):")
print(f"        print(f'{{int_to_emotion[idx]}}: {{prob:.2%}}')")
print("="*70)
