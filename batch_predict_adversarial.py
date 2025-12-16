"""
Batch prediction script for adversarial HuBERT emotion recognition

Processes multiple audio files from a CSV and generates predictions with accuracy metrics.

Usage:
    python batch_predict_adversarial.py \
        --model_path models/adversarial_hubert/best_model.pth \
        --csv_path data/csv/test_hindi_4class.csv \
        --output_csv results/predictions.csv

Author: Research Implementation
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2Processor

from predict_adversarial import load_model, predict_emotion


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
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
    print(f"  ✓ Confusion matrix saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch predict emotions from CSV file'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with audio paths')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save predictions CSV')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--plot_confusion', action='store_true',
                        help='Generate confusion matrix plot')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n{'='*70}")
    print(f"Batch Emotion Prediction")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"CSV: {args.csv_path}")
    print(f"Output: {args.output_csv}\n")

    # Load model
    model, id_to_emotion, config = load_model(args.model_path, device)

    # Load processor
    print("Loading HuBERT processor...")
    processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
    print("✓ Processor loaded!\n")

    # Load CSV
    print(f"Loading CSV file...")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    print(f"✓ Loaded {len(df)} samples\n")

    # Predict for each file
    print("Making predictions...")
    predictions = []
    confidences = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        audio_path = row['path']

        try:
            emotion, confidence, _ = predict_emotion(
                audio_path,
                model,
                processor,
                id_to_emotion,
                device,
                args.max_duration
            )
            predictions.append(emotion)
            confidences.append(confidence)
            errors.append(None)
        except Exception as e:
            print(f"\n  ⚠ Error processing {audio_path}: {e}")
            predictions.append("error")
            confidences.append(0.0)
            errors.append(str(e))

    # Add predictions to dataframe
    df['predicted_emotion'] = predictions
    df['confidence'] = confidences
    if any(errors):
        df['error'] = errors

    # Calculate accuracy if ground truth exists
    has_ground_truth = 'emotion' in df.columns

    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")

    if has_ground_truth:
        # Remove error rows for accuracy calculation
        valid_df = df[df['predicted_emotion'] != 'error'].copy()

        # Ensure emotion labels match
        valid_df['correct'] = valid_df['emotion'].str.lower() == valid_df['predicted_emotion'].str.lower()

        accuracy = valid_df['correct'].mean()
        f1 = f1_score(
            valid_df['emotion'].str.lower(),
            valid_df['predicted_emotion'].str.lower(),
            average='weighted',
            zero_division=0
        )

        print(f"Total Samples: {len(df)}")
        print(f"Valid Predictions: {len(valid_df)}")
        print(f"Errors: {len(df) - len(valid_df)}")
        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(f"Average Confidence: {valid_df['confidence'].mean():.2%}")

        # Per-emotion accuracy
        print(f"\nPer-Emotion Performance:")
        print(f"{'-'*70}")
        for emotion in sorted(valid_df['emotion'].str.lower().unique()):
            emotion_df = valid_df[valid_df['emotion'].str.lower() == emotion]
            emotion_acc = emotion_df['correct'].mean()
            emotion_conf = emotion_df['confidence'].mean()
            print(f"  {emotion.capitalize():10s}: Accuracy = {emotion_acc:6.2%}, "
                  f"Avg Confidence = {emotion_conf:6.2%}, "
                  f"Samples = {len(emotion_df):4d}")

        # Classification report
        print(f"\n{'-'*70}")
        print("Classification Report:")
        print(f"{'-'*70}")
        print(classification_report(
            valid_df['emotion'].str.lower(),
            valid_df['predicted_emotion'].str.lower(),
            zero_division=0
        ))

        # Confusion matrix
        if args.plot_confusion:
            output_dir = os.path.dirname(args.output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            cm_path = args.output_csv.replace('.csv', '_confusion_matrix.png')
            emotion_labels = sorted(valid_df['emotion'].str.lower().unique())

            print("\nGenerating confusion matrix...")
            import numpy as np
            plot_confusion_matrix(
                valid_df['emotion'].str.lower().values,
                valid_df['predicted_emotion'].str.lower().values,
                emotion_labels,
                "Confusion Matrix - Batch Predictions",
                cm_path
            )

    else:
        print(f"Total Samples: {len(df)}")
        print(f"Valid Predictions: {sum(df['predicted_emotion'] != 'error')}")
        print(f"Errors: {sum(df['predicted_emotion'] == 'error')}")
        print(f"Average Confidence: {df[df['predicted_emotion'] != 'error']['confidence'].mean():.2%}")

        # Prediction distribution
        print(f"\nPrediction Distribution:")
        print(f"{'-'*70}")
        pred_counts = df[df['predicted_emotion'] != 'error']['predicted_emotion'].value_counts()
        for emotion, count in pred_counts.items():
            percentage = count / len(df) * 100
            print(f"  {emotion.capitalize():10s}: {count:4d} samples ({percentage:5.1f}%)")

    print(f"{'='*70}\n")

    # Save results
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.output_csv, index=False)
    print(f"✓ Predictions saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
