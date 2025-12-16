"""
test_predictions.py - Test model predictions against labeled test data
Validates model performance on the test set
"""

import os
import sys
from collections import Counter
import csv

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename to extract true emotion"""
    # RAVDESS format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
    # Example: 03-01-05-02-01-01-01.wav
    # Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

    try:
        parts = os.path.basename(filename).replace('.wav', '').split('-')
        if len(parts) >= 3:
            emotion_code = int(parts[2])

            emotion_map = {
                1: 'neutral',
                3: 'happy',
                4: 'sad',
                5: 'angry'
            }

            return emotion_map.get(emotion_code, None)
    except:
        return None

    return None

def test_traditional_ml(test_files, true_emotions):
    """Test Traditional ML model"""
    from utils import extract_feature
    import pickle

    with open('models/mlp_4emotions.pkl', 'rb') as f:
        data = pickle.load(f)

    predictions = []
    for audio_file in test_files:
        features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
        features_scaled = data['scaler'].transform([features])
        emotion_idx = data['model'].predict(features_scaled)[0]
        emotion = data['int_to_emotion'][emotion_idx]
        predictions.append(emotion)

    return predictions

def test_deep_learning(test_files, true_emotions):
    """Test Deep Learning model"""
    from tensorflow.keras.models import load_model
    from utils import extract_feature
    import pickle
    import numpy as np

    model = load_model('models/lstm_4emotions.keras')
    with open('models/lstm_4emotions_config.pkl', 'rb') as f:
        config = pickle.load(f)

    predictions = []
    for audio_file in test_files:
        features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
        features_scaled = config['scaler'].transform([features])
        features_rnn = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        proba = model.predict(features_rnn, verbose=0)[0]
        emotion_idx = np.argmax(proba)
        emotion = config['int_to_emotion'][emotion_idx]
        predictions.append(emotion)

    return predictions

def calculate_metrics(true_emotions, predictions, emotions):
    """Calculate accuracy and per-class metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    accuracy = accuracy_score(true_emotions, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_emotions, predictions, labels=emotions, zero_division=0
    )

    cm = confusion_matrix(true_emotions, predictions, labels=emotions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm
    }

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("MODEL PREDICTION TESTING")
    print("="*70)

    # Get test files from RAVDESS
    test_dir = 'data/ravdess/Actor_01'

    if not os.path.exists(test_dir):
        print(f"\nError: Test directory not found: {test_dir}")
        print("Please ensure RAVDESS dataset is available.")
        sys.exit(1)

    import glob
    all_files = glob.glob(os.path.join(test_dir, '*.wav'))

    # Filter to 4-class emotions only
    test_files = []
    true_emotions = []

    for f in all_files:
        emotion = parse_ravdess_filename(f)
        if emotion in ['sad', 'neutral', 'happy', 'angry']:
            test_files.append(f)
            true_emotions.append(emotion)

    if not test_files:
        print("\nError: No valid test files found")
        sys.exit(1)

    print(f"\nFound {len(test_files)} test files from RAVDESS")
    emotion_counts = Counter(true_emotions)
    print("\nTrue emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion:>8s}: {count:3d} files")

    emotions = ['sad', 'neutral', 'happy', 'angry']

    # Test available models
    results = {}

    print("\n" + "-"*70)

    # Test Traditional ML
    if os.path.exists('models/mlp_4emotions.pkl'):
        print("\n[Testing Traditional ML (MLP)]")
        print("  Processing files...")
        try:
            predictions = test_traditional_ml(test_files, true_emotions)
            metrics = calculate_metrics(true_emotions, predictions, emotions)
            results['MLP'] = metrics
            print(f"  ✓ Complete - Accuracy: {metrics['accuracy']:.2%}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Test Deep Learning
    if os.path.exists('models/lstm_4emotions.keras'):
        print("\n[Testing Deep Learning (LSTM)]")
        print("  Processing files...")
        try:
            predictions = test_deep_learning(test_files, true_emotions)
            metrics = calculate_metrics(true_emotions, predictions, emotions)
            results['LSTM'] = metrics
            print(f"  ✓ Complete - Accuracy: {metrics['accuracy']:.2%}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not results:
        print("\nNo models found to test!")
        sys.exit(1)

    # Display results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)

    # Overall accuracy
    print("\nOverall Accuracy:")
    print("-"*70)
    for model_name, metrics in results.items():
        print(f"  {model_name:20s} {metrics['accuracy']:>6.2%}")

    # Per-class metrics
    print("\nPer-Class Performance:")
    print("-"*70)
    print(f"{'Emotion':10s} {'Model':10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>8s}")
    print("-"*70)

    for emotion in emotions:
        idx = emotions.index(emotion)
        for model_name, metrics in results.items():
            print(f"{emotion:10s} {model_name:10s} "
                  f"{metrics['precision'][idx]:>9.2%} "
                  f"{metrics['recall'][idx]:>9.2%} "
                  f"{metrics['f1'][idx]:>9.2%} "
                  f"{int(metrics['support'][idx]):>8d}")
        print()

    # Confusion matrices
    for model_name, metrics in results.items():
        print(f"\nConfusion Matrix - {model_name}:")
        print("-"*70)

        cm = metrics['confusion_matrix']

        # Header
        print(f"{'':>12}", end='')
        for emotion in emotions:
            print(f"{emotion:>10s}", end='')
        print()
        print("-"*70)

        # Data
        for i, emotion in enumerate(emotions):
            print(f"{emotion:>12s}", end='')
            for j in range(len(emotions)):
                print(f"{cm[i][j]:>10d}", end='')
            print()

    # Model comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        accuracies = [(name, metrics['accuracy']) for name, metrics in results.items()]
        accuracies.sort(key=lambda x: x[1], reverse=True)

        print(f"\nRanking by Accuracy:")
        for rank, (model_name, accuracy) in enumerate(accuracies, 1):
            print(f"  {rank}. {model_name:20s} {accuracy:>6.2%}")

        best_model = accuracies[0][0]
        print(f"\n✓ Best performing model: {best_model}")

    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
