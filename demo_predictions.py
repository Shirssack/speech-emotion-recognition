"""
demo_predictions.py - Comprehensive demo of all prediction capabilities
Shows examples of using all three model types with different prediction methods
"""

import os
import glob

def demo_traditional_ml():
    """Demo: Traditional ML prediction"""
    from utils import extract_feature
    import pickle

    print("\n" + "="*70)
    print("DEMO 1: TRADITIONAL ML PREDICTION (MLP)")
    print("="*70)

    # Load model
    print("\n[Step 1] Loading MLP model...")
    with open('models/mlp_4emotions.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        int_to_emotion = data['int_to_emotion']
    print("✓ Model loaded successfully")

    # Get sample file
    sample_files = glob.glob('data/ravdess/Actor_01/*.wav')
    if not sample_files:
        print("⚠ No sample files found. Skipping demo.")
        return

    audio_file = sample_files[0]
    print(f"\n[Step 2] Analyzing: {os.path.basename(audio_file)}")

    # Extract features
    print("  - Extracting MFCC, Chroma, and Mel features...")
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    print(f"  - Feature vector shape: {features.shape}")

    # Scale
    print("  - Scaling features...")
    features_scaled = scaler.transform([features])

    # Predict
    print("  - Making prediction...")
    emotion_idx = model.predict(features_scaled)[0]
    emotion = int_to_emotion[emotion_idx]

    # Get probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = proba[emotion_idx]

        print(f"\n[Result]")
        print(f"  Predicted Emotion: {emotion.upper()}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"\n  All Probabilities:")
        for idx, prob in enumerate(proba):
            emo = int_to_emotion[idx]
            bar = "█" * int(prob * 40)
            print(f"    {emo:>8}: {prob:6.2%} {bar}")

def demo_deep_learning():
    """Demo: Deep Learning prediction"""
    from tensorflow.keras.models import load_model
    from utils import extract_feature
    import pickle
    import numpy as np

    print("\n" + "="*70)
    print("DEMO 2: DEEP LEARNING PREDICTION (LSTM)")
    print("="*70)

    # Check if model exists
    if not os.path.exists('models/lstm_4emotions.keras'):
        print("\n⚠ LSTM model not found. Train it first with:")
        print("  python train_deep_learning.py")
        return

    # Load model
    print("\n[Step 1] Loading LSTM model...")
    model = load_model('models/lstm_4emotions.keras')
    with open('models/lstm_4emotions_config.pkl', 'rb') as f:
        config = pickle.load(f)
        scaler = config['scaler']
        int_to_emotion = config['int_to_emotion']
    print("✓ Model loaded successfully")

    # Get sample file
    sample_files = glob.glob('data/ravdess/Actor_01/*.wav')
    if not sample_files:
        print("⚠ No sample files found. Skipping demo.")
        return

    audio_file = sample_files[1] if len(sample_files) > 1 else sample_files[0]
    print(f"\n[Step 2] Analyzing: {os.path.basename(audio_file)}")

    # Extract and preprocess
    print("  - Extracting features...")
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])

    print("  - Reshaping for RNN (samples, timesteps, features)...")
    features_rnn = features_scaled.reshape((1, 1, features_scaled.shape[1]))
    print(f"  - Input shape: {features_rnn.shape}")

    # Predict
    print("  - Making prediction with LSTM...")
    proba = model.predict(features_rnn, verbose=0)[0]
    emotion_idx = np.argmax(proba)
    emotion = int_to_emotion[emotion_idx]
    confidence = proba[emotion_idx]

    print(f"\n[Result]")
    print(f"  Predicted Emotion: {emotion.upper()}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"\n  All Probabilities:")
    for idx, prob in enumerate(proba):
        emo = int_to_emotion[idx]
        bar = "█" * int(prob * 40)
        print(f"    {emo:>8}: {prob:6.2%} {bar}")

def demo_transformer():
    """Demo: Transformer prediction"""
    from transformer_emotion_recognition import TransformerEmotionRecognizer

    print("\n" + "="*70)
    print("DEMO 3: TRANSFORMER PREDICTION (Wav2Vec2)")
    print("="*70)

    # Check if model exists
    if not os.path.exists('models/transformer_best.pt'):
        print("\n⚠ Transformer model not found. Train it first with:")
        print("  python train_transformer.py --epochs 15")
        return

    # Load model
    print("\n[Step 1] Loading Wav2Vec2 transformer model...")
    print("  (This may take a moment...)")
    rec = TransformerEmotionRecognizer(
        emotions=['sad', 'neutral', 'happy', 'angry']
    )
    rec.load_model('models/transformer_best.pt')
    print("✓ Model loaded successfully")

    # Get sample file
    sample_files = glob.glob('data/ravdess/Actor_01/*.wav')
    if not sample_files:
        print("⚠ No sample files found. Skipping demo.")
        return

    audio_file = sample_files[2] if len(sample_files) > 2 else sample_files[0]
    print(f"\n[Step 2] Analyzing: {os.path.basename(audio_file)}")

    # Predict
    print("  - Loading audio and extracting features...")
    print("  - Processing with transformer model...")
    emotion, confidence = rec.predict(audio_file)

    print(f"\n[Result]")
    print(f"  Predicted Emotion: {emotion.upper()}")
    print(f"  Confidence: {confidence:.2%}")

def demo_batch_comparison():
    """Demo: Compare all models on same files"""
    from utils import extract_feature
    import pickle

    print("\n" + "="*70)
    print("DEMO 4: MODEL COMPARISON")
    print("="*70)

    # Get sample files
    sample_files = glob.glob('data/ravdess/Actor_01/*.wav')[:5]
    if not sample_files:
        print("⚠ No sample files found. Skipping demo.")
        return

    print(f"\nComparing predictions on {len(sample_files)} audio files:")

    # Check which models are available
    has_mlp = os.path.exists('models/mlp_4emotions.pkl')
    has_lstm = os.path.exists('models/lstm_4emotions.keras')
    has_transformer = os.path.exists('models/transformer_best.pt')

    if not has_mlp:
        print("\n⚠ MLP model not found. Train it first.")
        return

    # Load MLP model
    with open('models/mlp_4emotions.pkl', 'rb') as f:
        mlp_data = pickle.load(f)

    # Process files
    print("\n" + "-"*70)
    print(f"{'File':40s} {'MLP':12s} {'LSTM':12s} {'Transformer':12s}")
    print("-"*70)

    for audio_file in sample_files:
        filename = os.path.basename(audio_file)[:40]
        results = [filename]

        # MLP prediction
        try:
            features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
            features_scaled = mlp_data['scaler'].transform([features])
            emotion_idx = mlp_data['model'].predict(features_scaled)[0]
            emotion = mlp_data['int_to_emotion'][emotion_idx]
            results.append(emotion)
        except:
            results.append("ERROR")

        # LSTM prediction
        if has_lstm:
            try:
                from tensorflow.keras.models import load_model
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
                results.append(emotion)
            except:
                results.append("ERROR")
        else:
            results.append("N/A")

        # Transformer prediction
        if has_transformer:
            try:
                from transformer_emotion_recognition import TransformerEmotionRecognizer
                if not hasattr(demo_batch_comparison, 'transformer'):
                    demo_batch_comparison.transformer = TransformerEmotionRecognizer(
                        emotions=['sad', 'neutral', 'happy', 'angry']
                    )
                    demo_batch_comparison.transformer.load_model('models/transformer_best.pt')

                emotion, _ = demo_batch_comparison.transformer.predict(audio_file)
                results.append(emotion)
            except:
                results.append("ERROR")
        else:
            results.append("N/A")

        print(f"{results[0]:40s} {results[1]:12s} {results[2]:12s} {results[3]:12s}")

    print("-"*70)

def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*15 + "EMOTION PREDICTION DEMO SUITE")
    print("="*70)
    print("\nThis demo showcases all prediction capabilities:")
    print("  1. Traditional ML (MLP) - Fast, CPU-friendly")
    print("  2. Deep Learning (LSTM) - Balanced speed/accuracy")
    print("  3. Transformer (Wav2Vec2) - Best accuracy")
    print("  4. Model Comparison - Compare all models")

    input("\nPress Enter to start the demos...")

    # Run demos
    try:
        demo_traditional_ml()
        input("\nPress Enter to continue to next demo...")

        demo_deep_learning()
        input("\nPress Enter to continue to next demo...")

        demo_transformer()
        input("\nPress Enter to continue to next demo...")

        demo_batch_comparison()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return

    # Final summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nTo use these models in your own code:")
    print("  - Single file: python predict_emotion.py <file> [model_type]")
    print("  - Batch files: python predict_batch.py <directory> [model_type]")
    print("  - Interactive: python predict_interactive.py")
    print("  - Quick test:  python quick_predict.py <file>")
    print("\nFor more details, see PREDICTION_GUIDE.md")
    print("="*70)

if __name__ == "__main__":
    main()
