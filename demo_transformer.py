"""
demo_transformer.py - Quick demo for transformer-based emotion recognition
Author: Shirssack

This script demonstrates:
1. Loading a trained transformer model
2. Making predictions on audio files
3. Batch prediction on multiple files

QUICK START - Just modify these settings and run: python demo_transformer.py
"""

import os
import argparse
from transformer_emotion_recognition import TransformerEmotionRecognizer
from glob import glob
from collections import Counter


# ============================================================================
# CONFIGURATION - CHANGE THESE SETTINGS TO YOUR NEEDS
# ============================================================================

# Path to your trained model (change this to your model location)
DEFAULT_MODEL_PATH = 'models/transformer_best.pt'

# Emotions your model was trained on (MUST match training!)
DEFAULT_EMOTIONS = ['sad', 'neutral', 'happy', 'angry']

# For single prediction mode
DEFAULT_AUDIO_FILE = 'data/ravdess/Actor_01/03-01-01-01-01-01-01.wav'

# For batch prediction mode
DEFAULT_AUDIO_DIR = 'data/ravdess'
DEFAULT_BATCH_LIMIT = 10  # Maximum number of files to process

# Demo mode: 'info', 'single', or 'batch'
DEFAULT_MODE = 'info'

# ============================================================================
# END CONFIGURATION
# ============================================================================


def demo_single_prediction(model_path, audio_path, emotions):
    """
    Predict emotion for a single audio file.

    Args:
        model_path: Path to trained model (.pt file)
        audio_path: Path to audio file (.wav)
        emotions: List of emotion labels (must match training)
    """
    print("\n" + "="*70)
    print("SINGLE FILE PREDICTION")
    print("="*70)

    try:
        # Initialize recognizer
        print(f"\n[1/3] Initializing recognizer...")
        print(f"      Emotions: {emotions}")
        recognizer = TransformerEmotionRecognizer(emotions=emotions)

        # Load model
        print(f"\n[2/3] Loading model from: {model_path}")
        recognizer.load_model(model_path)
        print(f"      ✓ Model loaded successfully")
        print(f"      ✓ Device: {recognizer.device}")

        # Predict
        print(f"\n[3/3] Predicting emotion...")
        print(f"      Audio: {audio_path}")
        emotion, confidence = recognizer.predict(audio_path)

        # Display result
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        print(f"  🎭 Emotion:    {emotion.upper()}")
        print(f"  📊 Confidence: {confidence:.1%}")
        print("="*70)

    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_batch_prediction(model_path, audio_dir, emotions, limit=10):
    """
    Predict emotions for multiple audio files.

    Args:
        model_path: Path to trained model (.pt file)
        audio_dir: Directory containing audio files
        emotions: List of emotion labels (must match training)
        limit: Maximum number of files to process
    """
    print("\n" + "="*70)
    print("BATCH PREDICTION")
    print("="*70)

    try:
        # Initialize recognizer
        print(f"\n[1/4] Initializing recognizer...")
        recognizer = TransformerEmotionRecognizer(emotions=emotions)

        # Load model
        print(f"\n[2/4] Loading model from: {model_path}")
        recognizer.load_model(model_path)
        print(f"      ✓ Model loaded successfully")

        # Get audio files
        print(f"\n[3/4] Scanning for audio files in: {audio_dir}")
        audio_files = glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)

        if not audio_files:
            print(f"      ⚠ No .wav files found in {audio_dir}")
            return

        # Limit files
        if len(audio_files) > limit:
            print(f"      Found {len(audio_files)} files, processing first {limit}")
            audio_files = audio_files[:limit]
        else:
            print(f"      Found {len(audio_files)} files")

        # Predict
        print(f"\n[4/4] Predicting emotions...")
        print("\n" + "-"*70)
        print(f"{'#':<4} {'File':<45} {'Emotion':<12} {'Confidence'}")
        print("-"*70)

        results = []
        errors = 0

        for i, audio_path in enumerate(audio_files, 1):
            try:
                emotion, confidence = recognizer.predict(audio_path)
                filename = os.path.basename(audio_path)
                results.append((filename, emotion, confidence))

                # Format output
                print(f"{i:<4} {filename:<45} {emotion:<12} {confidence:.1%}")

            except Exception as e:
                filename = os.path.basename(audio_path)
                print(f"{i:<4} {filename:<45} {'ERROR':<12} {str(e)[:20]}")
                errors += 1

        # Summary
        if results:
            print("-"*70)
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)

            emotion_counts = Counter([r[1] for r in results])
            total = len(results)

            print(f"\n✓ Successfully processed: {total} files")
            if errors > 0:
                print(f"✗ Errors: {errors} files")

            print(f"\nEmotion Distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {emotion:12} {count:3} files ({percentage:5.1f}%) {bar}")

            # Confidence statistics
            avg_confidence = sum(r[2] for r in results) / len(results)
            max_confidence = max(r[2] for r in results)
            min_confidence = min(r[2] for r in results)

            print(f"\nConfidence Statistics:")
            print(f"  Average: {avg_confidence:.1%}")
            print(f"  Maximum: {max_confidence:.1%}")
            print(f"  Minimum: {min_confidence:.1%}")
            print("="*70)

    except FileNotFoundError as e:
        print(f"\n❌ Error: Directory not found - {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_model_info(model_path, emotions):
    """
    Display model information and configuration.

    Args:
        model_path: Path to trained model (.pt file)
        emotions: List of emotion labels (must match training)
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)

    try:
        # Initialize recognizer
        print(f"\nInitializing recognizer...")
        recognizer = TransformerEmotionRecognizer(emotions=emotions)

        # Load model
        print(f"Loading model from: {model_path}\n")
        recognizer.load_model(model_path)

        print("="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        print(f"  Base Model:      {recognizer.model_name}")
        print(f"  Emotions:        {', '.join(recognizer.emotions)}")
        print(f"  Num Classes:     {recognizer.num_labels}")
        print(f"  Sample Rate:     {recognizer.sample_rate} Hz")
        print(f"  Max Duration:    {recognizer.max_duration} seconds")
        print(f"  Device:          {recognizer.device}")

        # Model parameters
        if recognizer.model:
            total_params = sum(p.numel() for p in recognizer.model.parameters())
            trainable_params = sum(p.numel() for p in recognizer.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            print(f"\n" + "="*70)
            print("MODEL PARAMETERS")
            print("="*70)
            print(f"  Total Parameters:      {total_params:,}")
            print(f"  Trainable Parameters:  {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            print(f"  Frozen Parameters:     {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

            # Estimate model size
            model_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
            print(f"  Estimated Size:        {model_size_mb:.1f} MB")

        print("="*70)
        print("\n✓ Model loaded and ready for predictions!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Model file not found - {e}")
        print("\nTrain a model first with:")
        print("  python train_transformer.py --epochs 3")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def run_quick_demo():
    """
    Run demo with default settings from configuration section.
    This is called when you run: python demo_transformer.py
    """
    print("\n" + "="*70)
    print("TRANSFORMER EMOTION RECOGNITION - QUICK DEMO")
    print("="*70)
    print("\nUsing default settings from configuration section:")
    print(f"  Model Path:  {DEFAULT_MODEL_PATH}")
    print(f"  Emotions:    {DEFAULT_EMOTIONS}")
    print(f"  Mode:        {DEFAULT_MODE}")

    # Check if model exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"\n❌ Error: Model not found at {DEFAULT_MODEL_PATH}")
        print("\nTo train a model, run:")
        print("  python train_transformer.py --epochs 3")
        print("\nOr modify DEFAULT_MODEL_PATH in this script to point to your model.")
        return

    # Run appropriate demo
    if DEFAULT_MODE == 'info':
        demo_model_info(DEFAULT_MODEL_PATH, DEFAULT_EMOTIONS)

    elif DEFAULT_MODE == 'single':
        if not os.path.exists(DEFAULT_AUDIO_FILE):
            print(f"\n❌ Error: Audio file not found at {DEFAULT_AUDIO_FILE}")
            print("\nModify DEFAULT_AUDIO_FILE in this script to point to your audio file.")
            return
        demo_single_prediction(DEFAULT_MODEL_PATH, DEFAULT_AUDIO_FILE, DEFAULT_EMOTIONS)

    elif DEFAULT_MODE == 'batch':
        if not os.path.exists(DEFAULT_AUDIO_DIR):
            print(f"\n❌ Error: Directory not found at {DEFAULT_AUDIO_DIR}")
            print("\nModify DEFAULT_AUDIO_DIR in this script to point to your audio directory.")
            return
        demo_batch_prediction(DEFAULT_MODEL_PATH, DEFAULT_AUDIO_DIR, DEFAULT_EMOTIONS, DEFAULT_BATCH_LIMIT)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Transformer-based Emotion Recognition Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model information
  python demo_transformer.py --mode info

  # Predict single file
  python demo_transformer.py --mode single --audio_file data/ravdess/Actor_01/03-01-01-01-01-01-01.wav

  # Batch prediction
  python demo_transformer.py --mode batch --audio_dir data/ravdess --limit 20

  # Use different model and emotions
  python demo_transformer.py --mode single --model_path models/multilingual_best.pt --emotions angry happy sad neutral fear disgust --audio_file my_audio.wav
        """
    )

    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to trained model (default: {DEFAULT_MODEL_PATH})')

    parser.add_argument('--emotions', nargs='+', default=DEFAULT_EMOTIONS,
                        help=f'Emotion labels - must match training (default: {" ".join(DEFAULT_EMOTIONS)})')

    parser.add_argument('--mode', type=str, choices=['info', 'single', 'batch'],
                        default=None, help='Demo mode (default: use quick demo)')

    # Single prediction options
    parser.add_argument('--audio_file', type=str, default=DEFAULT_AUDIO_FILE,
                        help='Audio file for single prediction')

    # Batch prediction options
    parser.add_argument('--audio_dir', type=str, default=DEFAULT_AUDIO_DIR,
                        help='Directory with audio files for batch prediction')

    parser.add_argument('--limit', type=int, default=DEFAULT_BATCH_LIMIT,
                        help='Max number of files to process in batch mode')

    args = parser.parse_args()

    # If no mode specified, run quick demo with defaults
    if args.mode is None:
        run_quick_demo()
        return

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"\n❌ Error: Model not found at {args.model_path}")
        print("\nTrain a model first with:")
        print("  python train_transformer.py --epochs 3")
        return

    # Run selected demo mode
    if args.mode == 'info':
        demo_model_info(args.model_path, args.emotions)

    elif args.mode == 'single':
        if not os.path.exists(args.audio_file):
            print(f"\n❌ Error: Audio file not found at {args.audio_file}")
            return
        demo_single_prediction(args.model_path, args.audio_file, args.emotions)

    elif args.mode == 'batch':
        if not os.path.exists(args.audio_dir):
            print(f"\n❌ Error: Directory not found at {args.audio_dir}")
            return
        demo_batch_prediction(args.model_path, args.audio_dir, args.emotions, args.limit)


if __name__ == "__main__":
    main()
