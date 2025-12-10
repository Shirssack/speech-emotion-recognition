"""
predict_emotion.py - Predict emotion from audio using trained models
Usage: python predict_emotion.py <audio_file> [model_type]
"""

import sys
import os

def predict_traditional(audio_file, model_path='models/mlp_4emotions.pkl'):
    """Predict using Traditional ML model (MLP, SVM, etc.)"""
    from utils import extract_feature
    import pickle

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        int_to_emotion = data['int_to_emotion']

    print("Extracting features from audio...")
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])

    print("Predicting emotion...")
    emotion_idx = model.predict(features_scaled)[0]
    emotion = int_to_emotion[emotion_idx]

    # Get confidence if available
    confidence = None
    probs = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = proba[emotion_idx]
        probs = {int_to_emotion[i]: float(p) for i, p in enumerate(proba)}

    return emotion, confidence, probs

def predict_deep_learning(audio_file, model_path='models/lstm_4emotions'):
    """Predict using Deep Learning model (LSTM/GRU)"""
    from tensorflow.keras.models import load_model
    from utils import extract_feature
    import pickle
    import numpy as np

    print(f"Loading model from {model_path}.keras...")
    model = load_model(f'{model_path}.keras')

    with open(f'{model_path}_config.pkl', 'rb') as f:
        config = pickle.load(f)
        scaler = config['scaler']
        int_to_emotion = config['int_to_emotion']

    print("Extracting features from audio...")
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])

    # Reshape for RNN (samples, timesteps, features)
    features_rnn = features_scaled.reshape((1, 1, features_scaled.shape[1]))

    print("Predicting emotion...")
    proba = model.predict(features_rnn, verbose=0)[0]
    emotion_idx = np.argmax(proba)
    emotion = int_to_emotion[emotion_idx]
    confidence = proba[emotion_idx]
    probs = {int_to_emotion[i]: float(p) for i, p in enumerate(proba)}

    return emotion, confidence, probs

def predict_transformer(audio_file, model_path='models/transformer_best.pt'):
    """Predict using Transformer model (Wav2Vec2)"""
    from transformer_emotion_recognition import TransformerEmotionRecognizer

    print(f"Loading transformer model from {model_path}...")
    rec = TransformerEmotionRecognizer(
        emotions=['sad', 'neutral', 'happy', 'angry']
    )
    rec.load_model(model_path)

    print("Predicting emotion with transformer...")
    emotion, confidence = rec.predict(audio_file)

    # Note: Transformer returns single emotion and confidence
    # For consistency, we return probs as None
    return emotion, confidence, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*70)
        print("EMOTION PREDICTION FROM AUDIO")
        print("="*70)
        print("\nUsage: python predict_emotion.py <audio_file> [model_type]")
        print("\nArguments:")
        print("  audio_file    Path to audio file (.wav)")
        print("  model_type    Model to use (default: traditional)")
        print("\nAvailable model types:")
        print("  traditional   - Traditional ML (MLP, SVM, etc.) - Fast")
        print("  deep_learning - LSTM/GRU - Moderate speed")
        print("  transformer   - Wav2Vec2 - Best accuracy, slower")
        print("\nExamples:")
        print("  python predict_emotion.py audio.wav")
        print("  python predict_emotion.py audio.wav traditional")
        print("  python predict_emotion.py audio.wav deep_learning")
        print("  python predict_emotion.py audio.wav transformer")
        print("="*70)
        sys.exit(1)

    audio_file = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'traditional'

    # Validate inputs
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    print("\n" + "="*70)
    print("EMOTION PREDICTION")
    print("="*70)
    print(f"Audio File: {audio_file}")
    print(f"Model Type: {model_type}")
    print("="*70 + "\n")

    try:
        if model_type == 'traditional':
            emotion, confidence, probs = predict_traditional(audio_file)
        elif model_type == 'deep_learning':
            emotion, confidence, probs = predict_deep_learning(audio_file)
        elif model_type == 'transformer':
            emotion, confidence, probs = predict_transformer(audio_file)
        else:
            print(f"Error: Unknown model type: {model_type}")
            print("Use: traditional, deep_learning, or transformer")
            sys.exit(1)

        # Display results
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        print(f"Predicted Emotion: {emotion.upper()}")
        if confidence:
            print(f"Confidence: {confidence:.2%}")

        if probs:
            print("\nAll Probabilities:")
            for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"  {emo:>8}: {prob:6.2%} {bar}")

        print("="*70)

    except FileNotFoundError as e:
        print(f"Error: Model file not found. Please train the model first.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
