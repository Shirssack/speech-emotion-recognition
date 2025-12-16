#!/usr/bin/env python3
"""
train_adversarial.py - Training Script for Adversarial Cross-Lingual SER

This script trains the AdversarialHuBERT model for cross-lingual speech emotion
recognition using Hindi and English datasets.

Usage:
    # Standard training with both languages
    python train_adversarial.py

    # Custom configuration
    python train_adversarial.py --epochs 30 --max_lambda 0.5 --adversarial_layer 6

    # Cross-lingual evaluation (train English, test Hindi)
    python train_adversarial.py --cross_lingual_eval

Author: Research Implementation
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from adversarial_emotion_recognition import AdversarialEmotionTrainer


def load_csv_data(csv_path, language):
    """
    Load data from CSV file and add language labels.

    Args:
        csv_path: Path to CSV file with 'path' and 'emotion' columns
        language: Language label ('english' or 'hindi')

    Returns:
        paths: List of audio file paths
        emotions: List of emotion labels
        languages: List of language labels
    """
    df = pd.read_csv(csv_path)

    paths = df['path'].tolist()
    emotions = df['emotion'].tolist()
    languages = [language] * len(paths)

    return paths, emotions, languages


def combine_datasets(datasets):
    """
    Combine multiple datasets.

    Args:
        datasets: List of (paths, emotions, languages) tuples

    Returns:
        Combined paths, emotions, languages
    """
    all_paths = []
    all_emotions = []
    all_languages = []

    for paths, emotions, languages in datasets:
        all_paths.extend(paths)
        all_emotions.extend(emotions)
        all_languages.extend(languages)

    return all_paths, all_emotions, all_languages


def filter_emotions(paths, emotions, languages, target_emotions):
    """Filter data to include only specified emotions."""
    filtered_paths = []
    filtered_emotions = []
    filtered_languages = []

    for p, e, l in zip(paths, emotions, languages):
        if e in target_emotions:
            filtered_paths.append(p)
            filtered_emotions.append(e)
            filtered_languages.append(l)

    return filtered_paths, filtered_emotions, filtered_languages


def main():
    parser = argparse.ArgumentParser(
        description='Train Adversarial HuBERT for Cross-Lingual Speech Emotion Recognition'
    )

    # Data arguments
    parser.add_argument(
        '--train_english_csv',
        nargs='+',
        default=['data/csv/train_ravdess_4class.csv', 'data/csv/train_tess_4class.csv'],
        help='Training CSV files for English data'
    )
    parser.add_argument(
        '--test_english_csv',
        nargs='+',
        default=['data/csv/test_ravdess_4class.csv', 'data/csv/test_tess_4class.csv'],
        help='Test CSV files for English data'
    )
    parser.add_argument(
        '--train_hindi_csv',
        default='data/csv/train_hindi_4class.csv',
        help='Training CSV file for Hindi data'
    )
    parser.add_argument(
        '--test_hindi_csv',
        default='data/csv/test_hindi_4class.csv',
        help='Test CSV file for Hindi data'
    )
    parser.add_argument(
        '--emotions',
        nargs='+',
        default=['sad', 'neutral', 'happy', 'angry'],
        help='Emotions to train on'
    )

    # Model arguments
    parser.add_argument(
        '--model_name',
        default='facebook/hubert-base-ls960',
        help='Pretrained model (hubert-base or wav2vec2-base)'
    )
    parser.add_argument(
        '--adversarial_layer',
        type=int,
        default=4,
        help='Layer for adversarial language discrimination (1-12)'
    )
    parser.add_argument(
        '--emotion_layer',
        type=int,
        default=12,
        help='Layer for emotion classification (1-12)'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        default=True,
        help='Freeze HuBERT feature extractor'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Learning rate (lower for adversarial stability)'
    )
    parser.add_argument(
        '--max_lambda',
        type=float,
        default=1.0,
        help='Maximum adversarial weight'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio for learning rate scheduler'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=5.0,
        help='Maximum audio duration in seconds'
    )
    parser.add_argument(
        '--no_mixed_precision',
        action='store_true',
        help='Disable mixed precision training'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        default='models/',
        help='Directory to save models'
    )
    parser.add_argument(
        '--model_suffix',
        default='adversarial_hubert',
        help='Model filename suffix'
    )

    # Evaluation arguments
    parser.add_argument(
        '--cross_lingual_eval',
        action='store_true',
        help='Run cross-lingual evaluation after training'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ADVERSARIAL CROSS-LINGUAL SPEECH EMOTION RECOGNITION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Adversarial layer: {args.adversarial_layer}")
    print(f"  Emotion layer: {args.emotion_layer}")
    print(f"  Max lambda: {args.max_lambda}")
    print(f"  Emotions: {args.emotions}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print()

    # =========================================================================
    # Load Data
    # =========================================================================
    print("Loading datasets...")

    # Load English data
    train_english_data = []
    for csv_path in args.train_english_csv:
        if os.path.exists(csv_path):
            data = load_csv_data(csv_path, 'english')
            train_english_data.append(data)
            print(f"  Loaded {len(data[0])} samples from {csv_path}")
        else:
            print(f"  Warning: {csv_path} not found, skipping")

    test_english_data = []
    for csv_path in args.test_english_csv:
        if os.path.exists(csv_path):
            data = load_csv_data(csv_path, 'english')
            test_english_data.append(data)
            print(f"  Loaded {len(data[0])} samples from {csv_path}")
        else:
            print(f"  Warning: {csv_path} not found, skipping")

    # Load Hindi data
    train_hindi_data = []
    if os.path.exists(args.train_hindi_csv):
        data = load_csv_data(args.train_hindi_csv, 'hindi')
        train_hindi_data.append(data)
        print(f"  Loaded {len(data[0])} samples from {args.train_hindi_csv}")
    else:
        print(f"  Warning: {args.train_hindi_csv} not found, skipping")

    test_hindi_data = []
    if os.path.exists(args.test_hindi_csv):
        data = load_csv_data(args.test_hindi_csv, 'hindi')
        test_hindi_data.append(data)
        print(f"  Loaded {len(data[0])} samples from {args.test_hindi_csv}")
    else:
        print(f"  Warning: {args.test_hindi_csv} not found, skipping")

    # Combine datasets
    train_paths, train_emotions, train_languages = combine_datasets(
        train_english_data + train_hindi_data
    )
    test_paths, test_emotions, test_languages = combine_datasets(
        test_english_data + test_hindi_data
    )

    # Filter to target emotions
    train_paths, train_emotions, train_languages = filter_emotions(
        train_paths, train_emotions, train_languages, args.emotions
    )
    test_paths, test_emotions, test_languages = filter_emotions(
        test_paths, test_emotions, test_languages, args.emotions
    )

    print(f"\nDataset summary:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"    English: {train_languages.count('english')}")
    print(f"    Hindi: {train_languages.count('hindi')}")
    print(f"  Test samples: {len(test_paths)}")
    print(f"    English: {test_languages.count('english')}")
    print(f"    Hindi: {test_languages.count('hindi')}")

    if len(train_paths) == 0:
        print("Error: No training data found!")
        sys.exit(1)

    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    print("\nInitializing trainer...")

    trainer = AdversarialEmotionTrainer(
        emotions=args.emotions,
        languages=['english', 'hindi'],
        model_name=args.model_name,
        max_duration=args.max_duration
    )

    # =========================================================================
    # Build Model
    # =========================================================================
    print("\nBuilding model...")

    trainer.build_model(
        freeze_feature_extractor=args.freeze_encoder,
        adversarial_layer=args.adversarial_layer,
        emotion_layer=args.emotion_layer
    )

    # =========================================================================
    # Prepare Data
    # =========================================================================
    print("\nPreparing data loaders...")

    trainer.prepare_data(
        train_paths=train_paths,
        train_emotions=train_emotions,
        train_languages=train_languages,
        test_paths=test_paths,
        test_emotions=test_emotions,
        test_languages=test_languages,
        batch_size=args.batch_size
    )

    # =========================================================================
    # Train
    # =========================================================================
    print("\nStarting training...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        args.output_dir,
        f"{args.model_suffix}_{timestamp}.pt"
    )

    history = trainer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_lambda=args.max_lambda,
        use_mixed_precision=not args.no_mixed_precision,
        save_best_model=True,
        model_path=model_path
    )

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    trainer.load_model(model_path)

    # Overall evaluation
    results = trainer.evaluate()
    print(f"\nOverall Results:")
    print(f"  Emotion Accuracy: {results['emotion_acc']:.4f}")
    print(f"  Emotion F1 Score: {results['emotion_f1']:.4f}")
    print(f"  Language Accuracy: {results['language_acc']:.4f}")
    print(f"  (Lower language accuracy = better disentanglement)")

    # Confusion matrices
    cms = trainer.get_confusion_matrices()
    print(f"\nEmotion Confusion Matrix:")
    print(f"  Labels: {args.emotions}")
    print(cms['emotion'])

    # =========================================================================
    # Cross-Lingual Evaluation
    # =========================================================================
    if args.cross_lingual_eval:
        print("\n" + "=" * 70)
        print("CROSS-LINGUAL EVALUATION")
        print("=" * 70)

        # Separate test data by language
        english_test_paths = [p for p, l in zip(test_paths, test_languages) if l == 'english']
        english_test_emotions = [e for e, l in zip(test_emotions, test_languages) if l == 'english']
        english_test_languages = ['english'] * len(english_test_paths)

        hindi_test_paths = [p for p, l in zip(test_paths, test_languages) if l == 'hindi']
        hindi_test_emotions = [e for e, l in zip(test_emotions, test_languages) if l == 'hindi']
        hindi_test_languages = ['hindi'] * len(hindi_test_paths)

        if english_test_paths:
            english_results = trainer.evaluate_cross_lingual(
                english_test_paths, english_test_emotions, english_test_languages
            )
            print(f"\nEnglish Test Set:")
            print(f"  Samples: {len(english_test_paths)}")
            print(f"  Emotion Accuracy: {english_results['emotion_acc']:.4f}")
            print(f"  Emotion F1 Score: {english_results['emotion_f1']:.4f}")

        if hindi_test_paths:
            hindi_results = trainer.evaluate_cross_lingual(
                hindi_test_paths, hindi_test_emotions, hindi_test_languages
            )
            print(f"\nHindi Test Set:")
            print(f"  Samples: {len(hindi_test_paths)}")
            print(f"  Emotion Accuracy: {hindi_results['emotion_acc']:.4f}")
            print(f"  Emotion F1 Score: {hindi_results['emotion_f1']:.4f}")

        # Cross-lingual gap analysis
        if english_test_paths and hindi_test_paths:
            gap = abs(english_results['emotion_acc'] - hindi_results['emotion_acc'])
            print(f"\nCross-Lingual Gap: {gap:.4f}")
            print(f"  (Lower gap = better language invariance)")

    # =========================================================================
    # Save Training History
    # =========================================================================
    history_path = model_path.replace('.pt', '_history.npz')
    np.savez(
        history_path,
        emotion_loss=history['emotion_loss'],
        language_loss=history['language_loss'],
        lambda_values=history['lambda'],
        train_acc=history['train_acc'],
        val_acc=history['val_acc']
    )
    print(f"\nTraining history saved to: {history_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
