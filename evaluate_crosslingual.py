"""
Cross-Lingual Evaluation Script for Adversarial HuBERT

This script evaluates cross-lingual transfer performance for emotion recognition:
- Hindi → English: Train on Hindi, test on English
- English → Hindi: Train on English, test on Hindi
- Baseline comparison: With and without adversarial training

Key Metrics:
- Within-language accuracy
- Cross-lingual transfer accuracy
- Transfer gap (difference between within-language and cross-lingual)
- Language discriminator confusion (measure of language invariance)

Author: Research Implementation
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2FeatureExtractor

from adversarial_hubert_emotion import LayerWiseAdversarialHuBERT
from train_adversarial_hubert import (
    EmotionDataset,
    load_data_with_language_labels,
    evaluate
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Path
):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cm, cm_normalized


def evaluate_cross_lingual_transfer(
    model: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    train_paths: List[str],
    train_labels: List[int],
    test_paths_same: List[str],
    test_labels_same: List[int],
    test_paths_cross: List[str],
    test_labels_cross: List[int],
    train_lang: str,
    test_lang: str,
    emotion_labels: List[str],
    batch_size: int,
    max_duration: float,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    Evaluate cross-lingual transfer

    Args:
        train_lang: "english" or "hindi"
        test_lang: "english" or "hindi"
    """
    print(f"\n{'='*80}")
    print(f"Cross-Lingual Transfer: Train on {train_lang.upper()}, Test on {test_lang.upper()}")
    print(f"{'='*80}")

    # Create language IDs
    train_lang_id = 0 if train_lang == "english" else 1
    test_same_lang_id = train_lang_id
    test_cross_lang_id = 1 - train_lang_id

    # Create dataloaders
    print("\nCreating dataloaders...")

    # Within-language test (same as training language)
    test_dataset_same = EmotionDataset(
        test_paths_same,
        test_labels_same,
        [test_same_lang_id] * len(test_paths_same),
        processor,
        max_duration
    )
    test_loader_same = DataLoader(
        test_dataset_same,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Cross-lingual test (different from training language)
    test_dataset_cross = EmotionDataset(
        test_paths_cross,
        test_labels_cross,
        [test_cross_lang_id] * len(test_paths_cross),
        processor,
        max_duration
    )
    test_loader_cross = DataLoader(
        test_dataset_cross,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate within-language (source)
    print(f"\nEvaluating within-language ({train_lang.upper()})...")
    metrics_same = evaluate(model, test_loader_same, device, f"Within-{train_lang}")

    # Evaluate cross-lingual (target)
    print(f"\nEvaluating cross-lingual ({test_lang.upper()})...")
    metrics_cross = evaluate(model, test_loader_cross, device, f"Cross-{test_lang}")

    # Calculate transfer gap
    transfer_gap = metrics_same['emotion_f1'] - metrics_cross['emotion_f1']

    # Print results
    print(f"\n{'='*80}")
    print(f"Results: {train_lang.upper()} → {test_lang.upper()}")
    print(f"{'='*80}")
    print(f"Within-language ({train_lang.upper()}):")
    print(f"  Accuracy: {metrics_same['emotion_acc']:.4f}")
    print(f"  F1 Score: {metrics_same['emotion_f1']:.4f}")
    print(f"\nCross-lingual ({test_lang.upper()}):")
    print(f"  Accuracy: {metrics_cross['emotion_acc']:.4f}")
    print(f"  F1 Score: {metrics_cross['emotion_f1']:.4f}")
    print(f"\nTransfer Gap: {transfer_gap:.4f} (Lower is better)")
    print(f"Transfer Rate: {(metrics_cross['emotion_f1'] / metrics_same['emotion_f1'] * 100):.2f}%")
    print(f"{'='*80}")

    # Plot confusion matrices
    cm_same, _ = plot_confusion_matrix(
        metrics_same['emotion_labels'],
        metrics_same['emotion_preds'],
        emotion_labels,
        f"Within-Language: {train_lang.upper()}",
        output_dir / f"confusion_within_{train_lang}.png"
    )

    cm_cross, _ = plot_confusion_matrix(
        metrics_cross['emotion_labels'],
        metrics_cross['emotion_preds'],
        emotion_labels,
        f"Cross-Lingual: {train_lang.upper()} → {test_lang.upper()}",
        output_dir / f"confusion_cross_{train_lang}_to_{test_lang}.png"
    )

    # Classification reports
    print(f"\nClassification Report - Within-Language ({train_lang.upper()}):")
    print(classification_report(
        metrics_same['emotion_labels'],
        metrics_same['emotion_preds'],
        target_names=emotion_labels,
        digits=4
    ))

    print(f"\nClassification Report - Cross-Lingual ({test_lang.upper()}):")
    print(classification_report(
        metrics_cross['emotion_labels'],
        metrics_cross['emotion_preds'],
        target_names=emotion_labels,
        digits=4
    ))

    return {
        "train_language": train_lang,
        "test_language": test_lang,
        "within_language": {
            "accuracy": float(metrics_same['emotion_acc']),
            "f1": float(metrics_same['emotion_f1'])
        },
        "cross_lingual": {
            "accuracy": float(metrics_cross['emotion_acc']),
            "f1": float(metrics_cross['emotion_f1'])
        },
        "transfer_gap": float(transfer_gap),
        "transfer_rate": float(metrics_cross['emotion_f1'] / metrics_same['emotion_f1'] * 100)
    }


def analyze_language_invariance(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Analyze language invariance by checking language discriminator performance

    If the model successfully learned language-invariant representations,
    the language discriminator should perform poorly (close to random chance).
    """
    model.eval()

    all_language_preds_list = [[] for _ in range(len(model.adversarial_layers))]
    all_language_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            language_labels = batch["language_label"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_adversarial=True
            )

            # Collect predictions from all discriminators
            for i, language_logits in enumerate(outputs["language_logits_list"]):
                language_preds = torch.argmax(language_logits, dim=1)
                all_language_preds_list[i].extend(language_preds.cpu().numpy())

            all_language_labels.extend(language_labels.cpu().numpy())

    # Calculate accuracy for each discriminator
    results = {}
    for i, (layer_idx, preds) in enumerate(zip(model.adversarial_layers, all_language_preds_list)):
        acc = accuracy_score(all_language_labels, preds)
        results[f"layer_{layer_idx}"] = {
            "accuracy": float(acc),
            "confusion": float(0.5 - abs(0.5 - acc))  # How close to random (0.5)
        }

    avg_acc = np.mean([r["accuracy"] for r in results.values()])
    avg_confusion = np.mean([r["confusion"] for r in results.values()])

    results["average"] = {
        "accuracy": float(avg_acc),
        "confusion": float(avg_confusion)
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Cross-Lingual Transfer for Adversarial HuBERT'
    )

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/crosslingual',
                        help='Output directory for results')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/csv',
                        help='Directory containing CSV files')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')

    # Evaluation modes
    parser.add_argument('--evaluate_both_directions', action='store_true',
                        help='Evaluate both Hindi→English and English→Hindi')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Cross-Lingual Evaluation for Adversarial HuBERT")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")

    # Load checkpoint
    print(f"\nLoading model checkpoint...")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    emotion_to_id = checkpoint['emotion_to_id']
    id_to_emotion = checkpoint['id_to_emotion']

    # Create model
    print(f"Creating model...")
    model = LayerWiseAdversarialHuBERT(
        model_name=config['model_name'],
        num_emotions=len(emotion_to_id),
        num_languages=2,
        adversarial_layers=config['adversarial_layers'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        freeze_feature_extractor=config.get('freeze_feature_extractor', True),
        gradient_checkpointing=False  # Disable for evaluation
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Emotions: {list(emotion_to_id.keys())}")

    # Load processor (feature extractor only; HuBERT has no tokenizer vocab)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config['model_name'])

    # Load data
    print(f"\n{'='*80}")
    print("Loading Data")
    print(f"{'='*80}")

    # English data
    train_csv_en = [
        os.path.join(args.data_dir, 'train_ravdess_4class.csv'),
        os.path.join(args.data_dir, 'train_tess_4class.csv')
    ]
    test_csv_en = [
        os.path.join(args.data_dir, 'test_ravdess_4class.csv'),
        os.path.join(args.data_dir, 'test_tess_4class.csv')
    ]

    print("\nLoading English data...")
    train_paths_en, train_labels_en, _ = load_data_with_language_labels(
        train_csv_en, emotion_to_id, "english"
    )
    test_paths_en, test_labels_en, _ = load_data_with_language_labels(
        test_csv_en, emotion_to_id, "english"
    )

    # Hindi data
    train_csv_hi = [os.path.join(args.data_dir, 'train_hindi_4class.csv')]
    test_csv_hi = [os.path.join(args.data_dir, 'test_hindi_4class.csv')]

    print("\nLoading Hindi data...")
    train_paths_hi, train_labels_hi, _ = load_data_with_language_labels(
        train_csv_hi, emotion_to_id, "hindi"
    )
    test_paths_hi, test_labels_hi, _ = load_data_with_language_labels(
        test_csv_hi, emotion_to_id, "hindi"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emotion_labels = [id_to_emotion[i] for i in range(len(id_to_emotion))]

    # Evaluate cross-lingual transfer
    results = {}

    # Hindi → English
    print(f"\n{'#'*80}")
    print("Scenario 1: Train on HINDI, Test on ENGLISH")
    print(f"{'#'*80}")
    results['hindi_to_english'] = evaluate_cross_lingual_transfer(
        model, processor,
        train_paths_hi, train_labels_hi,
        test_paths_hi, test_labels_hi,
        test_paths_en, test_labels_en,
        "hindi", "english",
        emotion_labels,
        args.batch_size, args.max_duration,
        device, output_dir
    )

    # English → Hindi (if requested)
    if args.evaluate_both_directions:
        print(f"\n{'#'*80}")
        print("Scenario 2: Train on ENGLISH, Test on HINDI")
        print(f"{'#'*80}")
        results['english_to_hindi'] = evaluate_cross_lingual_transfer(
            model, processor,
            train_paths_en, train_labels_en,
            test_paths_en, test_labels_en,
            test_paths_hi, test_labels_hi,
            "english", "hindi",
            emotion_labels,
            args.batch_size, args.max_duration,
            device, output_dir
        )

    # Analyze language invariance
    print(f"\n{'='*80}")
    print("Language Invariance Analysis")
    print(f"{'='*80}")
    print("\nCreating combined test set...")

    # Create combined test dataset
    test_paths_combined = test_paths_en + test_paths_hi
    test_labels_combined = test_labels_en + test_labels_hi
    test_lang_labels_combined = [0] * len(test_paths_en) + [1] * len(test_paths_hi)

    test_dataset_combined = EmotionDataset(
        test_paths_combined,
        test_labels_combined,
        test_lang_labels_combined,
        processor,
        args.max_duration
    )
    test_loader_combined = DataLoader(
        test_dataset_combined,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Analyzing language discriminator performance...")
    print("(Lower accuracy = better language invariance)")
    language_invariance = analyze_language_invariance(
        model, test_loader_combined, device
    )

    print(f"\nLanguage Discriminator Performance:")
    for layer_name, metrics in language_invariance.items():
        if layer_name != "average":
            print(f"  {layer_name}: Accuracy = {metrics['accuracy']:.4f}, "
                  f"Confusion = {metrics['confusion']:.4f}")

    print(f"\n  Average: Accuracy = {language_invariance['average']['accuracy']:.4f}, "
          f"Confusion = {language_invariance['average']['confusion']:.4f}")

    if language_invariance['average']['accuracy'] < 0.6:
        print("\n  ✓ Good language invariance! Discriminator performs poorly.")
    elif language_invariance['average']['accuracy'] < 0.7:
        print("\n  ~ Moderate language invariance.")
    else:
        print("\n  ✗ Weak language invariance. Discriminator can still identify language.")

    results['language_invariance'] = language_invariance

    # Save results
    results_file = output_dir / "crosslingual_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"Confusion matrices saved to: {output_dir}")
    print(f"{'='*80}\n")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for direction, result in results.items():
        if direction == 'language_invariance':
            continue

        print(f"\n{direction.upper().replace('_', ' ')}:")
        print(f"  Within-language F1: {result['within_language']['f1']:.4f}")
        print(f"  Cross-lingual F1: {result['cross_lingual']['f1']:.4f}")
        print(f"  Transfer Gap: {result['transfer_gap']:.4f}")
        print(f"  Transfer Rate: {result['transfer_rate']:.2f}%")

    print(f"\nLanguage Invariance:")
    print(f"  Average Discriminator Accuracy: {language_invariance['average']['accuracy']:.4f}")
    print(f"  (Closer to 0.5 = better language invariance)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
