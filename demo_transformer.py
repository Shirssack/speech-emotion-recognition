"""
demo_transformer.py - Quick demo for transformer-based emotion recognition
Author: Shirssack

This script demonstrates:
1. Loading a trained transformer model
2. Making predictions on audio files
3. Batch prediction on multiple files
"""

import os
import argparse
from transformer_emotion_recognition import TransformerEmotionRecognizer
from glob import glob


def demo_single_prediction(model_path, audio_path, emotions):
    """Demo: Predict emotion for a single audio file."""
    print("\n" + "="*60)
    print("SINGLE FILE PREDICTION DEMO")
    print("="*60)

    # Initialize recognizer
    recognizer = TransformerEmotionRecognizer(emotions=emotions)

    # Load model
    print(f"\nLoading model from: {model_path}")
    recognizer.load_model(model_path)

    # Predict
    print(f"\nPredicting emotion for: {audio_path}")
    emotion, confidence = recognizer.predict(audio_path)

    print(f"\n[Result]")
    print(f"  Emotion: {emotion}")
    print(f"  Confidence: {confidence:.2%}")


def demo_batch_prediction(model_path, audio_dir, emotions, limit=10):
    """Demo: Predict emotions for multiple audio files."""
    print("\n" + "="*60)
    print("BATCH PREDICTION DEMO")
    print("="*60)

    # Initialize recognizer
    recognizer = TransformerEmotionRecognizer(emotions=emotions)

    # Load model
    print(f"\nLoading model from: {model_path}")
    recognizer.load_model(model_path)

    # Get audio files
    audio_files = glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)[:limit]

    if not audio_files:
        print(f"\nNo audio files found in {audio_dir}")
        return

    print(f"\nPredicting emotions for {len(audio_files)} files...")
    print("\n" + "-"*60)

    # Predict for each file
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        try:
            emotion, confidence = recognizer.predict(audio_path)
            results.append((os.path.basename(audio_path), emotion, confidence))
            print(f"{i:2d}. {os.path.basename(audio_path):40s} -> {emotion:10s} ({confidence:.2%})")
        except Exception as e:
            print(f"{i:2d}. {os.path.basename(audio_path):40s} -> ERROR: {e}")

    # Summary
    if results:
        print("\n" + "-"*60)
        print("\n[Summary]")
        from collections import Counter
        emotion_counts = Counter([r[1] for r in results])
        for emotion, count in emotion_counts.most_common():
            print(f"  {emotion}: {count} files ({count/len(results)*100:.1f}%)")


def demo_model_info(model_path, emotions):
    """Demo: Display model information."""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)

    # Initialize recognizer
    recognizer = TransformerEmotionRecognizer(emotions=emotions)

    # Load model
    print(f"\nLoading model from: {model_path}")
    recognizer.load_model(model_path)

    print(f"\n[Model Configuration]")
    print(f"  Model name: {recognizer.model_name}")
    print(f"  Emotions: {recognizer.emotions}")
    print(f"  Number of classes: {recognizer.num_labels}")
    print(f"  Sample rate: {recognizer.sample_rate} Hz")
    print(f"  Max duration: {recognizer.max_duration} seconds")
    print(f"  Device: {recognizer.device}")

    # Model parameters
    if recognizer.model:
        total_params = sum(p.numel() for p in recognizer.model.parameters())
        trainable_params = sum(p.numel() for p in recognizer.model.parameters() if p.requires_grad)
        print(f"\n[Model Parameters]")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")


def main():
    parser = argparse.ArgumentParser(description='Transformer Emotion Recognition Demo')

    parser.add_argument('--model_path', type=str, default='models/transformer_best.pt',
                        help='Path to trained model')
    parser.add_argument('--emotions', nargs='+', default=['sad', 'neutral', 'happy'],
                        help='Emotion labels')

    # Demo modes
    parser.add_argument('--mode', type=str, choices=['info', 'single', 'batch'],
                        default='info', help='Demo mode')

    # For single prediction
    parser.add_argument('--audio_file', type=str,
                        help='Audio file for single prediction')

    # For batch prediction
    parser.add_argument('--audio_dir', type=str, default='data/ravdess',
                        help='Directory with audio files for batch prediction')
    parser.add_argument('--limit', type=int, default=10,
                        help='Max number of files to process in batch mode')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"\n[ERROR] Model not found: {args.model_path}")
        print("\nPlease train a model first:")
        print("  python train_transformer.py --epochs 3")
        return

    # Run demo based on mode
    if args.mode == 'info':
        demo_model_info(args.model_path, args.emotions)

    elif args.mode == 'single':
        if not args.audio_file:
            print("\n[ERROR] --audio_file required for single mode")
            return
        if not os.path.exists(args.audio_file):
            print(f"\n[ERROR] Audio file not found: {args.audio_file}")
            return
        demo_single_prediction(args.model_path, args.audio_file, args.emotions)

    elif args.mode == 'batch':
        if not os.path.exists(args.audio_dir):
            print(f"\n[ERROR] Audio directory not found: {args.audio_dir}")
            return
        demo_batch_prediction(args.model_path, args.audio_dir, args.emotions, args.limit)


if __name__ == "__main__":
    # Example usage:
    print("\n" + "="*60)
    print("TRANSFORMER EMOTION RECOGNITION - DEMO")
    print("="*60)
    print("\nUsage examples:")
    print("\n1. Show model info:")
    print("   python demo_transformer.py --mode info")
    print("\n2. Predict single file:")
    print("   python demo_transformer.py --mode single --audio_file path/to/audio.wav")
    print("\n3. Batch predict:")
    print("   python demo_transformer.py --mode batch --audio_dir data/ravdess --limit 10")
    print("\n" + "="*60)

    main()
