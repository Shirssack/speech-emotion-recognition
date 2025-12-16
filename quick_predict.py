"""
quick_predict.py - Fastest way to predict emotion from audio
Simple script with minimal output for quick testing
"""

import sys
import os

def quick_predict(audio_file):
    """Quick prediction using Traditional ML (fastest)"""
    from utils import extract_feature
    import pickle

    # Load model
    model_path = 'models/mlp_4emotions.pkl'
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        int_to_emotion = data['int_to_emotion']

    # Predict
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])
    emotion_idx = model.predict(features_scaled)[0]
    emotion = int_to_emotion[emotion_idx]

    # Get confidence if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = proba[emotion_idx]

    return emotion, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_predict.py <audio_file>")
        print("Example: python quick_predict.py audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    try:
        emotion, confidence = quick_predict(audio_file)

        if confidence:
            print(f"{emotion} ({confidence:.1%})")
        else:
            print(f"{emotion}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
