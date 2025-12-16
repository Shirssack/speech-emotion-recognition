"""
predict_interactive.py - Interactive emotion prediction tool
Provides a simple menu-driven interface for emotion prediction
"""

import os
import glob

def list_audio_files(directory='data/ravdess/Actor_01', limit=20):
    """List available audio files in a directory"""
    files = glob.glob(os.path.join(directory, '*.wav'))
    return files[:limit]

def predict_emotion(audio_file, model_type='traditional'):
    """Predict emotion from audio file"""
    from utils import extract_feature
    import pickle

    if model_type == 'traditional':
        # Load traditional ML model
        with open('models/mlp_4emotions.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data['scaler']
            int_to_emotion = data['int_to_emotion']

        features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
        features_scaled = scaler.transform([features])
        emotion_idx = model.predict(features_scaled)[0]
        emotion = int_to_emotion[emotion_idx]

        confidence = None
        probs = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = proba[emotion_idx]
            probs = {int_to_emotion[i]: float(p) for i, p in enumerate(proba)}

        return emotion, confidence, probs

    elif model_type == 'deep_learning':
        # Load deep learning model
        from tensorflow.keras.models import load_model
        import numpy as np

        model = load_model('models/lstm_4emotions.keras')
        with open('models/lstm_4emotions_config.pkl', 'rb') as f:
            config = pickle.load(f)
            scaler = config['scaler']
            int_to_emotion = config['int_to_emotion']

        features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
        features_scaled = scaler.transform([features])
        features_rnn = features_scaled.reshape((1, 1, features_scaled.shape[1]))

        proba = model.predict(features_rnn, verbose=0)[0]
        emotion_idx = np.argmax(proba)
        emotion = int_to_emotion[emotion_idx]
        confidence = proba[emotion_idx]
        probs = {int_to_emotion[i]: float(p) for i, p in enumerate(proba)}

        return emotion, confidence, probs

    elif model_type == 'transformer':
        # Load transformer model
        from transformer_emotion_recognition import TransformerEmotionRecognizer

        rec = TransformerEmotionRecognizer(
            emotions=['sad', 'neutral', 'happy', 'angry']
        )
        rec.load_model('models/transformer_best.pt')

        emotion, confidence = rec.predict(audio_file)
        return emotion, confidence, None

def main():
    """Main interactive loop"""
    print("\n" + "="*70)
    print(" "*20 + "EMOTION PREDICTION TOOL")
    print("="*70)

    # Select model
    print("\nSelect model type:")
    print("  1. Traditional ML (MLP) - Fast")
    print("  2. Deep Learning (LSTM) - Moderate")
    print("  3. Transformer (Wav2Vec2) - Best accuracy")

    model_choice = input("\nEnter choice (1-3, default: 1): ").strip() or "1"

    model_map = {
        '1': 'traditional',
        '2': 'deep_learning',
        '3': 'transformer'
    }

    model_type = model_map.get(model_choice, 'traditional')

    print(f"\nUsing: {model_type} model")

    while True:
        print("\n" + "-"*70)
        print("Options:")
        print("  1. Predict emotion from file path")
        print("  2. Select from sample files")
        print("  3. Change model type")
        print("  4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            audio_file = input("Enter audio file path: ").strip()

            if not os.path.exists(audio_file):
                print(f"Error: File not found: {audio_file}")
                continue

            try:
                print("\nPredicting...")
                emotion, confidence, probs = predict_emotion(audio_file, model_type)

                print("\n" + "="*70)
                print("RESULT")
                print("="*70)
                print(f"File: {os.path.basename(audio_file)}")
                print(f"Predicted Emotion: {emotion.upper()}")
                if confidence:
                    print(f"Confidence: {confidence:.2%}")

                if probs:
                    print("\nAll Probabilities:")
                    for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        bar = "█" * int(prob * 30)
                        print(f"  {emo:>8}: {prob:6.2%} {bar}")
                print("="*70)

            except Exception as e:
                print(f"Error during prediction: {e}")

        elif choice == '2':
            print("\nLooking for sample files...")
            sample_dir = 'data/ravdess/Actor_01'

            if not os.path.exists(sample_dir):
                print("Error: Sample directory not found. Please provide a file path.")
                continue

            files = list_audio_files(sample_dir, limit=15)

            if not files:
                print("No audio files found in sample directory.")
                continue

            print(f"\nSample audio files from {sample_dir}:")
            for i, f in enumerate(files, 1):
                print(f"  {i:2d}. {os.path.basename(f)}")

            file_choice = input(f"\nSelect file (1-{len(files)}, or 0 to cancel): ").strip()

            try:
                file_idx = int(file_choice)
                if file_idx == 0:
                    continue
                if 1 <= file_idx <= len(files):
                    audio_file = files[file_idx - 1]

                    print("\nPredicting...")
                    emotion, confidence, probs = predict_emotion(audio_file, model_type)

                    print("\n" + "="*70)
                    print("RESULT")
                    print("="*70)
                    print(f"File: {os.path.basename(audio_file)}")
                    print(f"Predicted Emotion: {emotion.upper()}")
                    if confidence:
                        print(f"Confidence: {confidence:.2%}")

                    if probs:
                        print("\nAll Probabilities:")
                        for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                            bar = "█" * int(prob * 30)
                            print(f"  {emo:>8}: {prob:6.2%} {bar}")
                    print("="*70)

                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
            except Exception as e:
                print(f"Error during prediction: {e}")

        elif choice == '3':
            print("\nSelect model type:")
            print("  1. Traditional ML (MLP)")
            print("  2. Deep Learning (LSTM)")
            print("  3. Transformer (Wav2Vec2)")

            model_choice = input("\nEnter choice (1-3): ").strip()
            model_type = model_map.get(model_choice, model_type)
            print(f"\nSwitched to: {model_type} model")

        elif choice == '4':
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
