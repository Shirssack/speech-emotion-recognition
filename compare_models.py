"""
compare_models.py - Compare all available models on a single audio file
Shows side-by-side comparison of Traditional ML, Deep Learning, and Transformer
"""

import sys
import os
import time

def predict_with_mlp(audio_file):
    """Predict using MLP"""
    from utils import extract_feature
    import pickle

    with open('models/mlp_4emotions.pkl', 'rb') as f:
        data = pickle.load(f)

    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = data['scaler'].transform([features])
    emotion_idx = data['model'].predict(features_scaled)[0]
    emotion = data['int_to_emotion'][emotion_idx]

    probs = None
    if hasattr(data['model'], 'predict_proba'):
        proba = data['model'].predict_proba(features_scaled)[0]
        probs = {data['int_to_emotion'][i]: float(p) for i, p in enumerate(proba)}

    return emotion, probs

def predict_with_lstm(audio_file):
    """Predict using LSTM"""
    from tensorflow.keras.models import load_model
    from utils import extract_feature
    import pickle
    import numpy as np

    model = load_model('models/lstm_4emotions.keras')
    with open('models/lstm_4emotions_config.pkl', 'rb') as f:
        config = pickle.load(f)

    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = config['scaler'].transform([features])
    features_rnn = features_scaled.reshape((1, 1, features_scaled.shape[1]))

    proba = model.predict(features_rnn, verbose=0)[0]
    emotion_idx = np.argmax(proba)
    emotion = config['int_to_emotion'][emotion_idx]
    probs = {config['int_to_emotion'][i]: float(p) for i, p in enumerate(proba)}

    return emotion, probs

def predict_with_transformer(audio_file):
    """Predict using Transformer"""
    from transformer_emotion_recognition import TransformerEmotionRecognizer

    if not hasattr(predict_with_transformer, 'recognizer'):
        predict_with_transformer.recognizer = TransformerEmotionRecognizer(
            emotions=['sad', 'neutral', 'happy', 'angry']
        )
        predict_with_transformer.recognizer.load_model('models/transformer_best.pt')

    emotion, confidence = predict_with_transformer.recognizer.predict(audio_file)

    # Create probability dict with single value (transformer doesn't return all probs)
    probs = {emotion: confidence}

    return emotion, probs

def compare_all_models(audio_file):
    """Compare all available models"""

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Audio File: {audio_file}")
    print("="*70)

    models = []

    # Check which models exist
    if os.path.exists('models/mlp_4emotions.pkl'):
        models.append(('Traditional ML (MLP)', predict_with_mlp, 'mlp'))
    if os.path.exists('models/lstm_4emotions.keras'):
        models.append(('Deep Learning (LSTM)', predict_with_lstm, 'lstm'))
    if os.path.exists('models/transformer_best.pt'):
        models.append(('Transformer (Wav2Vec2)', predict_with_transformer, 'transformer'))

    if not models:
        print("\nError: No trained models found!")
        print("Please train at least one model first.")
        return

    results = []

    # Run predictions
    for model_name, predict_func, model_id in models:
        print(f"\n[{model_name}]")
        print("  Loading model and predicting...")

        try:
            start_time = time.time()
            emotion, probs = predict_func(audio_file)
            elapsed = time.time() - start_time

            results.append({
                'name': model_name,
                'id': model_id,
                'emotion': emotion,
                'probs': probs,
                'time': elapsed
            })

            print(f"  ✓ Prediction: {emotion.upper()}")
            print(f"  ✓ Time: {elapsed:.3f} seconds")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'name': model_name,
                'id': model_id,
                'emotion': 'ERROR',
                'probs': None,
                'time': 0
            })

    # Display comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    # Emotion predictions
    print("\nPredicted Emotions:")
    print("-"*70)
    for r in results:
        if r['emotion'] != 'ERROR':
            print(f"  {r['name']:30s} → {r['emotion'].upper():8s}")
        else:
            print(f"  {r['name']:30s} → ERROR")

    # Processing times
    print("\nProcessing Times:")
    print("-"*70)
    for r in results:
        if r['time'] > 0:
            print(f"  {r['name']:30s} → {r['time']:.3f} seconds")

    # Probability distributions
    print("\nDetailed Probabilities:")
    print("-"*70)

    # Get all unique emotions
    all_emotions = set()
    for r in results:
        if r['probs']:
            all_emotions.update(r['probs'].keys())

    all_emotions = sorted(all_emotions)

    if all_emotions:
        # Header
        print(f"{'Model':30s}", end='')
        for emo in all_emotions:
            print(f"{emo:>10s}", end='')
        print()
        print("-"*70)

        # Data
        for r in results:
            print(f"{r['name']:30s}", end='')
            if r['probs']:
                for emo in all_emotions:
                    prob = r['probs'].get(emo, 0.0)
                    print(f"{prob:>9.2%}", end=' ')
            else:
                print("  N/A", end='')
            print()

    # Agreement analysis
    print("\n" + "="*70)
    print("AGREEMENT ANALYSIS")
    print("="*70)

    valid_predictions = [r for r in results if r['emotion'] != 'ERROR']
    if len(valid_predictions) > 1:
        emotions = [r['emotion'] for r in valid_predictions]
        if len(set(emotions)) == 1:
            print(f"✓ All models agree: {emotions[0].upper()}")
        else:
            print(f"⚠ Models disagree:")
            for r in valid_predictions:
                confidence = ""
                if r['probs'] and r['emotion'] in r['probs']:
                    confidence = f" ({r['probs'][r['emotion']]:.1%})"
                print(f"    {r['name']:30s} → {r['emotion']:8s}{confidence}")

            # Find consensus
            from collections import Counter
            vote_count = Counter(emotions)
            consensus, count = vote_count.most_common(1)[0]
            if count > len(emotions) / 2:
                print(f"\n  Consensus (majority): {consensus.upper()} ({count}/{len(emotions)} models)")

    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*70)
        print("MODEL COMPARISON TOOL")
        print("="*70)
        print("\nUsage: python compare_models.py <audio_file>")
        print("\nThis script compares predictions from all available models:")
        print("  - Traditional ML (MLP)")
        print("  - Deep Learning (LSTM)")
        print("  - Transformer (Wav2Vec2)")
        print("\nExample:")
        print("  python compare_models.py data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
        print("="*70)
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    try:
        compare_all_models(audio_file)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
