"""
predict_batch.py - Batch emotion prediction from multiple audio files
Usage: python predict_batch.py <audio_dir> [model_type] [--limit N]
"""

import sys
import os
import glob
import time
from collections import Counter

def predict_traditional(audio_file, model_path='models/mlp_4emotions.pkl'):
    """Predict using Traditional ML model"""
    from utils import extract_feature
    import pickle

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        int_to_emotion = data['int_to_emotion']

    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])
    emotion_idx = model.predict(features_scaled)[0]
    emotion = int_to_emotion[emotion_idx]

    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = proba[emotion_idx]

    return emotion, confidence

def predict_deep_learning(audio_file, model_path='models/lstm_4emotions'):
    """Predict using Deep Learning model"""
    from tensorflow.keras.models import load_model
    from utils import extract_feature
    import pickle
    import numpy as np

    model = load_model(f'{model_path}.keras')
    with open(f'{model_path}_config.pkl', 'rb') as f:
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

    return emotion, confidence

def predict_transformer(audio_file, model_path='models/transformer_best.pt'):
    """Predict using Transformer model"""
    from transformer_emotion_recognition import TransformerEmotionRecognizer

    if not hasattr(predict_transformer, 'recognizer'):
        predict_transformer.recognizer = TransformerEmotionRecognizer(
            emotions=['sad', 'neutral', 'happy', 'angry']
        )
        predict_transformer.recognizer.load_model(model_path)

    emotion, confidence = predict_transformer.recognizer.predict(audio_file)
    return emotion, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*70)
        print("BATCH EMOTION PREDICTION")
        print("="*70)
        print("\nUsage: python predict_batch.py <audio_dir> [model_type] [--limit N]")
        print("\nArguments:")
        print("  audio_dir     Directory containing audio files (.wav)")
        print("  model_type    Model to use (default: traditional)")
        print("  --limit N     Process only first N files (default: all)")
        print("\nAvailable model types:")
        print("  traditional   - Traditional ML (MLP, SVM, etc.)")
        print("  deep_learning - LSTM/GRU")
        print("  transformer   - Wav2Vec2")
        print("\nExamples:")
        print("  python predict_batch.py data/ravdess/Actor_01")
        print("  python predict_batch.py data/ravdess/Actor_01 deep_learning")
        print("  python predict_batch.py data/ravdess/Actor_01 traditional --limit 10")
        print("="*70)
        sys.exit(1)

    audio_dir = sys.argv[1]
    model_type = 'traditional'
    limit = None

    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--limit':
            if i + 1 < len(sys.argv):
                limit = int(sys.argv[i + 1])
        elif arg in ['traditional', 'deep_learning', 'transformer']:
            model_type = arg

    # Validate inputs
    if not os.path.isdir(audio_dir):
        print(f"Error: Directory not found: {audio_dir}")
        sys.exit(1)

    # Get audio files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))

    if not audio_files:
        print(f"Error: No .wav files found in {audio_dir}")
        sys.exit(1)

    if limit:
        audio_files = audio_files[:limit]

    print("\n" + "="*70)
    print("BATCH EMOTION PREDICTION")
    print("="*70)
    print(f"Audio Directory: {audio_dir}")
    print(f"Model Type: {model_type}")
    print(f"Files to process: {len(audio_files)}")
    print("="*70 + "\n")

    # Load model once (for efficiency)
    print(f"Loading {model_type} model...")
    try:
        if model_type == 'traditional':
            predict_func = predict_traditional
        elif model_type == 'deep_learning':
            predict_func = predict_deep_learning
        elif model_type == 'transformer':
            predict_func = predict_transformer
        else:
            print(f"Error: Unknown model type: {model_type}")
            sys.exit(1)

        # Warm up model with first file
        predict_func(audio_files[0])
        print("Model loaded successfully!\n")

    except FileNotFoundError as e:
        print(f"Error: Model file not found. Please train the model first.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Process all files
    results = []
    start_time = time.time()

    print("Processing files...")
    print("-"*70)

    for i, audio_file in enumerate(audio_files, 1):
        try:
            emotion, confidence = predict_func(audio_file)
            results.append({
                'file': os.path.basename(audio_file),
                'emotion': emotion,
                'confidence': confidence
            })

            conf_str = f"({confidence:.1%})" if confidence else ""
            print(f"[{i:3d}/{len(audio_files)}] {os.path.basename(audio_file):50s} -> {emotion:8s} {conf_str}")

        except Exception as e:
            print(f"[{i:3d}/{len(audio_files)}] {os.path.basename(audio_file):50s} -> ERROR: {e}")

    elapsed_time = time.time() - start_time

    # Display summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(results)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per file: {elapsed_time/len(results):.3f} seconds")

    # Emotion distribution
    emotions = [r['emotion'] for r in results]
    emotion_counts = Counter(emotions)

    print("\nEmotion Distribution:")
    for emotion, count in emotion_counts.most_common():
        percentage = count / len(results) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {emotion:>8s}: {count:3d} ({percentage:5.1f}%) {bar}")

    # Average confidence
    confidences = [r['confidence'] for r in results if r['confidence'] is not None]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nAverage Confidence: {avg_confidence:.2%}")

    print("="*70)

    # Save results to CSV
    output_file = f"predictions_{model_type}_{int(time.time())}.csv"
    try:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'emotion', 'confidence'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to CSV: {e}")
