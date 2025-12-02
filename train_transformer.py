"""
train_transformer.py - Training script for transformer-based emotion recognition
Author: Shirssack

This script trains a Wav2Vec2-based emotion recognition model with:
- Memory-efficient training for 6GB GPU
- Mixed precision training (FP16)
- Support for multiple datasets (RAVDESS, TESS, Hindi)
- Automatic model checkpointing
"""

import os
import argparse
import pandas as pd
import numpy as np
from transformer_emotion_recognition import TransformerEmotionRecognizer


def load_csv_data(csv_files):
    """Load data from CSV files."""
    all_paths = []
    all_labels = []

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_paths.extend(df['path'].tolist())
            all_labels.extend(df['emotion'].tolist())
            print(f"Loaded {len(df)} samples from {csv_file}")
        else:
            print(f"Warning: {csv_file} not found, skipping...")

    return all_paths, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train Transformer-based Emotion Recognition')

    # Data arguments
    parser.add_argument('--train_csv', nargs='+',
                        default=['data/csv/train_ravdess.csv', 'data/csv/train_tess.csv'],
                        help='Training CSV files')
    parser.add_argument('--test_csv', nargs='+',
                        default=['data/csv/test_ravdess.csv', 'data/csv/test_tess.csv'],
                        help='Testing CSV files')
    parser.add_argument('--include_hindi', action='store_true',
                        help='Include Hindi dataset for training')
    parser.add_argument('--emotions', nargs='+',
                        default=['sad', 'neutral', 'happy'],
                        help='Emotions to train on')

    # Model arguments
    parser.add_argument('--model_name', type=str,
                        default='facebook/wav2vec2-base',
                        help='Pretrained model name')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze Wav2Vec2 feature extractor')
    parser.add_argument('--weighted_layers', action='store_true',
                        help='Use weighted layer sum')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (reduce if OOM on 6GB GPU)')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for scheduler')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--model_name_suffix', type=str, default='transformer',
                        help='Model name suffix')

    args = parser.parse_args()

    # Add Hindi dataset if requested
    if args.include_hindi:
        if 'data/csv/train_hindi.csv' not in args.train_csv:
            args.train_csv.append('data/csv/train_hindi.csv')
        if 'data/csv/test_hindi.csv' not in args.test_csv:
            args.test_csv.append('data/csv/test_hindi.csv')
        print("\n[Dataset] Including Hindi dataset for multilingual training")

    print("\n" + "="*70)
    print("TRANSFORMER-BASED EMOTION RECOGNITION TRAINING")
    print("="*70)

    # Load data
    print("\n[1/5] Loading data...")
    train_paths, train_labels = load_csv_data(args.train_csv)
    test_paths, test_labels = load_csv_data(args.test_csv)

    # Filter by selected emotions
    train_data = [(p, l) for p, l in zip(train_paths, train_labels) if l in args.emotions]
    test_data = [(p, l) for p, l in zip(test_paths, test_labels) if l in args.emotions]

    train_paths, train_labels = zip(*train_data) if train_data else ([], [])
    test_paths, test_labels = zip(*test_data) if test_data else ([], [])

    train_paths = list(train_paths)
    train_labels = list(train_labels)
    test_paths = list(test_paths)
    test_labels = list(test_labels)

    print(f"\n[Data] Training samples: {len(train_paths)}")
    print(f"[Data] Testing samples: {len(test_paths)}")
    print(f"[Data] Emotions: {args.emotions}")

    # Distribution
    print("\n[Data] Emotion distribution:")
    from collections import Counter
    train_dist = Counter(train_labels)
    for emotion in args.emotions:
        count = train_dist.get(emotion, 0)
        print(f"  {emotion}: {count} samples")

    # Initialize recognizer
    print("\n[2/5] Initializing transformer model...")
    recognizer = TransformerEmotionRecognizer(
        emotions=args.emotions,
        model_name=args.model_name,
        max_duration=args.max_duration,
        sample_rate=16000
    )

    # Build model
    print("\n[3/5] Building model architecture...")
    recognizer.build_model(
        freeze_feature_extractor=args.freeze_encoder,
        use_weighted_layer_sum=args.weighted_layers
    )

    # Prepare data
    print("\n[4/5] Preparing data loaders...")
    recognizer.prepare_data(
        train_paths, train_labels,
        test_paths, test_labels,
        batch_size=args.batch_size,
        num_workers=2
    )

    # Memory tips
    print("\n[Memory] Tips for 6GB GPU:")
    print("  - Gradient checkpointing: ENABLED")
    print("  - Mixed precision: " + ("ENABLED" if not args.no_mixed_precision else "DISABLED"))
    print("  - Recommended batch size: 4-8")
    print("  - If OOM, reduce --batch_size or --max_duration")

    # Train
    print("\n[5/5] Starting training...")
    model_path = os.path.join(
        args.output_dir,
        f"{args.model_name_suffix}_best.pt"
    )

    recognizer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        use_mixed_precision=not args.no_mixed_precision,
        save_best_model=True,
        model_path=model_path
    )

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    recognizer.load_model(model_path)
    accuracy, f1, loss = recognizer.evaluate()

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = recognizer.get_confusion_matrix()
    print(cm)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {model_path}")
    print("="*70)


if __name__ == "__main__":
    main()
