"""
evaluate_crosslingual.py - Cross-Lingual Evaluation

Evaluate trained model on cross-lingual scenarios:
1. English→English (in-domain)
2. Hindi→Hindi (in-domain)
3. English→Hindi (cross-lingual)
4. Hindi→English (cross-lingual)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hubert_crosslingual_emotion import CrossLingualEmotionRecognizer, CrossLingualEmotionDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import argparse


def evaluate_dataset(recognizer, test_csv, max_duration=5.0, batch_size=8, language_filter=None):
    """
    Evaluate model on a test dataset

    Args:
        recognizer: Trained recognizer
        test_csv: List of test CSV files
        max_duration: Max audio duration
        batch_size: Batch size for evaluation
        language_filter: Filter by language ('english', 'hindi', or None for all)
    """
    # Create test dataset
    test_dataset = CrossLingualEmotionDataset(
        test_csv, recognizer.processor, recognizer.emotions,
        max_duration=max_duration, augment=False
    )

    # Filter by language if specified
    if language_filter:
        test_dataset.data = [
            item for item in test_dataset.data
            if item['language'] == language_filter
        ]
        print(f"  Filtered to {len(test_dataset.data)} {language_filter} samples")

    if len(test_dataset.data) == 0:
        print(f"  No samples found for {language_filter}")
        return None, None, None

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    # Evaluation
    recognizer.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch['input_values'].to(recognizer.device)
            emotion_labels = batch['emotion_label'].to(recognizer.device)

            emotion_logits, _, _ = recognizer.model(
                input_values,
                language_labels=None
            )

            probs = F.softmax(emotion_logits, dim=1)
            _, predicted = torch.max(emotion_logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(emotion_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description='Cross-Lingual Evaluation')

    parser.add_argument('--model_path', type=str,
                        default='models/hubert_crosslingual_best.pt',
                        help='Path to trained model')
    parser.add_argument('--test_csv', nargs='+',
                        default=['data/csv/test_ravdess_4class.csv',
                                'data/csv/test_tess_4class.csv',
                                'data/csv/test_hindi_4class.csv'],
                        help='Test CSV files')
    parser.add_argument('--emotions', nargs='+',
                        default=['sad', 'neutral', 'happy', 'angry'],
                        help='Emotions to recognize')
    parser.add_argument('--adversarial_layers', nargs='+', type=int,
                        default=[6, 9, 12],
                        help='Adversarial layers used in training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CROSS-LINGUAL EVALUATION")
    print("="*70)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    recognizer = CrossLingualEmotionRecognizer(
        emotions=args.emotions,
        adversarial_layers=args.adversarial_layers
    )
    recognizer.load_model(args.model_path)
    print("✓ Model loaded successfully")

    # Evaluation scenarios
    scenarios = [
        ('English (RAVDESS+TESS)', 'english', 'In-domain English'),
        ('Hindi', 'hindi', 'In-domain Hindi'),
        ('All Languages', None, 'Combined'),
    ]

    results = {}

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    for scenario_name, lang_filter, description in scenarios:
        print(f"\n[{description}]")
        print(f"Evaluating on: {scenario_name}")

        preds, labels, probs = evaluate_dataset(
            recognizer, args.test_csv,
            max_duration=args.max_duration,
            batch_size=args.batch_size,
            language_filter=lang_filter
        )

        if preds is None:
            continue

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        results[scenario_name] = accuracy

        print(f"\nAccuracy: {accuracy:.2%}")

        # Per-class metrics
        print("\nPer-Class Performance:")
        pred_emotions = [args.emotions[p] for p in preds]
        true_emotions = [args.emotions[l] for l in labels]

        print(classification_report(
            true_emotions, pred_emotions,
            target_names=args.emotions,
            digits=3
        ))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(labels, preds)

        # Print header
        print(f"\n{'':>12}", end='')
        for emotion in args.emotions:
            print(f"{emotion:>10}", end='')
        print()
        print("-"*70)

        # Print matrix
        for i, emotion in enumerate(args.emotions):
            print(f"{emotion:>12}", end='')
            for j in range(len(args.emotions)):
                print(f"{cm[i][j]:>10}", end='')
            print()

    # Cross-lingual analysis
    print("\n" + "="*70)
    print("CROSS-LINGUAL ANALYSIS")
    print("="*70)

    if 'English (RAVDESS+TESS)' in results and 'Hindi' in results:
        english_acc = results['English (RAVDESS+TESS)']
        hindi_acc = results['Hindi']
        combined_acc = results.get('All Languages', 0)

        print(f"\nIn-Domain Performance:")
        print(f"  English: {english_acc:.2%}")
        print(f"  Hindi:   {hindi_acc:.2%}")
        print(f"  Average: {(english_acc + hindi_acc)/2:.2%}")

        print(f"\nCombined Performance:")
        print(f"  All Languages: {combined_acc:.2%}")

        # Cross-lingual generalization gap
        avg_in_domain = (english_acc + hindi_acc) / 2
        gap = avg_in_domain - combined_acc

        print(f"\nGeneralization Analysis:")
        print(f"  Average In-Domain: {avg_in_domain:.2%}")
        print(f"  Cross-Lingual:     {combined_acc:.2%}")
        print(f"  Gap:               {gap:.2%}")

        if gap < 0.05:
            print("  ✓ Excellent cross-lingual generalization!")
        elif gap < 0.10:
            print("  ✓ Good cross-lingual generalization")
        else:
            print("  ⚠ Moderate cross-lingual generalization")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nKey Results:")
    for scenario, acc in results.items():
        print(f"  {scenario:30s}: {acc:.2%}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
