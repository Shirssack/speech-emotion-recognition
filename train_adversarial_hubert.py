"""
Train Adversarial HuBERT for Cross-Lingual Emotion Recognition

This script implements training for the novel layer-wise adversarial disentanglement
approach using HuBERT for Hindi-English cross-lingual emotion recognition.

Key Features:
- Adversarial language disentanglement at multiple HuBERT layers
- Separate tracking of language labels for Hindi and English
- Cross-lingual evaluation (Hindi→English, English→Hindi)
- Lambda scheduling for gradient reversal
- Comprehensive metrics and checkpointing

Author: Research Implementation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor

from adversarial_hubert_emotion import (
    LayerWiseAdversarialHuBERT,
    compute_adversarial_loss,
    get_lambda_schedule
)


class EmotionDataset(Dataset):
    """
    Dataset for emotion recognition with language labels

    Args:
        audio_paths: List of audio file paths
        emotion_labels: List of emotion labels
        language_labels: List of language labels (0=English, 1=Hindi)
        processor: HuBERT processor for audio preprocessing
        max_duration: Maximum audio duration in seconds
        sampling_rate: Target sampling rate (default: 16000)
    """

    def __init__(
        self,
        audio_paths: List[str],
        emotion_labels: List[int],
        language_labels: List[int],
        processor: Wav2Vec2FeatureExtractor,
        max_duration: float = 5.0,
        sampling_rate: int = 16000
    ):
        self.audio_paths = audio_paths
        self.emotion_labels = emotion_labels
        self.language_labels = language_labels
        self.processor = processor
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.max_length = int(max_duration * sampling_rate)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """Load and preprocess audio sample"""
        try:
            # Load audio
            audio_path = self.audio_paths[idx]
            speech, sr = sf.read(audio_path)

            # Resample if needed
            if sr != self.sampling_rate:
                # Simple resampling (for production, use librosa.resample)
                import librosa
                speech = librosa.resample(speech, orig_sr=sr, target_sr=self.sampling_rate)

            # Truncate or pad to max_length
            if len(speech) > self.max_length:
                speech = speech[:self.max_length]
            else:
                speech = np.pad(speech, (0, self.max_length - len(speech)))

            # Process with HuBERT processor
            inputs = self.processor(
                speech,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )

            return {
                "input_values": inputs.input_values.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "emotion_label": torch.tensor(self.emotion_labels[idx], dtype=torch.long),
                "language_label": torch.tensor(self.language_labels[idx], dtype=torch.long)
            }

        except Exception as e:
            print(f"Error loading {self.audio_paths[idx]}: {e}")
            # Return dummy data on error
            return {
                "input_values": torch.zeros(self.max_length),
                "attention_mask": torch.ones(self.max_length),
                "emotion_label": torch.tensor(0, dtype=torch.long),
                "language_label": torch.tensor(0, dtype=torch.long)
            }


def load_data_with_language_labels(
    csv_files: List[str],
    emotion_to_id: Dict[str, int],
    language_name: str
) -> Tuple[List[str], List[int], List[int]]:
    """
    Load data from CSV files with language labels

    Args:
        csv_files: List of CSV file paths
        emotion_to_id: Mapping from emotion name to ID
        language_name: Language name ("english" or "hindi")

    Returns:
        audio_paths: List of audio file paths
        emotion_labels: List of emotion IDs
        language_labels: List of language IDs
    """
    language_id = 0 if language_name.lower() == "english" else 1

    audio_paths = []
    emotion_labels = []
    language_labels = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            emotion = row['emotion'].lower()
            if emotion in emotion_to_id:
                audio_paths.append(row['path'])
                emotion_labels.append(emotion_to_id[emotion])
                language_labels.append(language_id)

        print(f"  Loaded {len(df)} samples from {csv_file} ({language_name})")

    return audio_paths, emotion_labels, language_labels


def create_dataloaders(
    train_paths_en: List[str],
    train_labels_en: List[int],
    train_paths_hi: List[str],
    train_labels_hi: List[int],
    test_paths_en: List[str],
    test_labels_en: List[int],
    test_paths_hi: List[str],
    test_labels_hi: List[int],
    processor: Wav2Vec2FeatureExtractor,
    batch_size: int,
    max_duration: float,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training and testing (both combined and language-specific)

    Returns:
        train_loader: Combined training data (English + Hindi)
        test_loader_combined: Combined test data (English + Hindi)
        test_loader_en: English-only test data
        test_loader_hi: Hindi-only test data
    """
    # Combined training data
    train_paths = train_paths_en + train_paths_hi
    train_emotion_labels = train_labels_en + train_labels_hi
    train_language_labels = [0] * len(train_paths_en) + [1] * len(train_paths_hi)

    train_dataset = EmotionDataset(
        train_paths,
        train_emotion_labels,
        train_language_labels,
        processor,
        max_duration
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Combined test data
    test_paths_combined = test_paths_en + test_paths_hi
    test_emotion_labels_combined = test_labels_en + test_labels_hi
    test_language_labels_combined = [0] * len(test_paths_en) + [1] * len(test_paths_hi)

    test_dataset_combined = EmotionDataset(
        test_paths_combined,
        test_emotion_labels_combined,
        test_language_labels_combined,
        processor,
        max_duration
    )

    test_loader_combined = DataLoader(
        test_dataset_combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # English-only test data
    test_dataset_en = EmotionDataset(
        test_paths_en,
        test_labels_en,
        [0] * len(test_paths_en),
        processor,
        max_duration
    )

    test_loader_en = DataLoader(
        test_dataset_en,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Hindi-only test data
    test_dataset_hi = EmotionDataset(
        test_paths_hi,
        test_labels_hi,
        [1] * len(test_paths_hi),
        processor,
        max_duration
    )

    test_loader_hi = DataLoader(
        test_dataset_hi,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader_combined, test_loader_en, test_loader_hi


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    emotion_weight: float,
    language_weight: float,
    lambda_schedule: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_emotion_loss = 0.0
    total_language_loss = 0.0
    all_emotion_preds = []
    all_emotion_labels = []
    all_language_preds = []
    all_language_labels = []

    # Calculate total steps for lambda scheduling
    total_steps = total_epochs * len(train_loader)
    current_step_offset = epoch * len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["emotion_label"].to(device)
        language_labels = batch["language_label"].to(device)

        # Update lambda for GRL
        current_step = current_step_offset + batch_idx
        lambda_param = get_lambda_schedule(current_step, total_steps, lambda_schedule)
        model.set_lambda(lambda_param)

        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    return_adversarial=True
                )

                # Compute loss
                loss_dict = compute_adversarial_loss(
                    emotion_logits=outputs["emotion_logits"],
                    language_logits_list=outputs["language_logits_list"],
                    emotion_labels=emotion_labels,
                    language_labels=language_labels,
                    emotion_weight=emotion_weight,
                    language_weight=language_weight
                )

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss_dict["total_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_adversarial=True
            )

            # Compute loss
            loss_dict = compute_adversarial_loss(
                emotion_logits=outputs["emotion_logits"],
                language_logits_list=outputs["language_logits_list"],
                emotion_labels=emotion_labels,
                language_labels=language_labels,
                emotion_weight=emotion_weight,
                language_weight=language_weight
            )

            # Backward pass
            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

        # Track metrics
        total_loss += loss_dict["total_loss"].item()
        total_emotion_loss += loss_dict["emotion_loss"].item()
        total_language_loss += loss_dict["language_loss"].item()

        # Predictions
        emotion_preds = torch.argmax(outputs["emotion_logits"], dim=1)
        language_preds = torch.argmax(outputs["language_logits_list"][0], dim=1)  # Use first discriminator

        all_emotion_preds.extend(emotion_preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_language_preds.extend(language_preds.cpu().numpy())
        all_language_labels.extend(language_labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss'].item():.4f}",
            "λ": f"{lambda_param:.3f}"
        })

    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    avg_emotion_loss = total_emotion_loss / len(train_loader)
    avg_language_loss = total_language_loss / len(train_loader)

    emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    language_acc = accuracy_score(all_language_labels, all_language_preds)

    return {
        "loss": avg_loss,
        "emotion_loss": avg_emotion_loss,
        "language_loss": avg_language_loss,
        "emotion_acc": emotion_acc,
        "emotion_f1": emotion_f1,
        "language_acc": language_acc
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    split_name: str = "Test"
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    all_emotion_preds = []
    all_emotion_labels = []
    all_language_preds = []
    all_language_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {split_name}"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            language_labels = batch["language_label"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_adversarial=True
            )

            emotion_preds = torch.argmax(outputs["emotion_logits"], dim=1)
            language_preds = torch.argmax(outputs["language_logits_list"][0], dim=1)

            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_language_preds.extend(language_preds.cpu().numpy())
            all_language_labels.extend(language_labels.cpu().numpy())

    # Calculate metrics
    emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    language_acc = accuracy_score(all_language_labels, all_language_preds)

    return {
        "emotion_acc": emotion_acc,
        "emotion_f1": emotion_f1,
        "language_acc": language_acc,
        "emotion_preds": all_emotion_preds,
        "emotion_labels": all_emotion_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train Adversarial HuBERT for Cross-Lingual Emotion Recognition'
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/csv',
                        help='Directory containing CSV files')
    parser.add_argument('--emotions', nargs='+',
                        default=['sad', 'neutral', 'happy', 'angry'],
                        help='Emotions to train on')

    # Model arguments
    parser.add_argument('--model_name', type=str,
                        default='facebook/hubert-base-ls960',
                        help='HuBERT model name (feature extractor only; no tokenizer needed)')
    parser.add_argument('--adversarial_layers', nargs='+', type=int,
                        default=[3, 6, 9, 12],
                        help='Layers for gradient reversal')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for classifiers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--freeze_feature_extractor', action='store_true',
                        help='Freeze HuBERT feature extractor')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--emotion_weight', type=float, default=1.0,
                        help='Weight for emotion loss')
    parser.add_argument('--language_weight', type=float, default=0.1,
                        help='Weight for language loss')
    parser.add_argument('--lambda_schedule', type=str, default='progressive',
                        choices=['constant', 'progressive', 'linear'],
                        help='Lambda schedule for GRL')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str,
                        default='adversarial_hubert',
                        help='Experiment name')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Training Adversarial HuBERT for Cross-Lingual Emotion Recognition")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create emotion mapping
    emotion_to_id = {emotion.lower(): i for i, emotion in enumerate(args.emotions)}
    id_to_emotion = {i: emotion for emotion, i in emotion_to_id.items()}
    num_emotions = len(args.emotions)

    print(f"\nEmotions: {args.emotions}")
    print(f"Number of emotions: {num_emotions}")

    # Load data
    print(f"\n{'='*80}")
    print("Loading Data")
    print(f"{'='*80}")

    # English data (RAVDESS + TESS)
    train_csv_en = [
        os.path.join(args.data_dir, 'train_ravdess_4class.csv'),
        os.path.join(args.data_dir, 'train_tess_4class.csv')
    ]
    test_csv_en = [
        os.path.join(args.data_dir, 'test_ravdess_4class.csv'),
        os.path.join(args.data_dir, 'test_tess_4class.csv')
    ]

    print("\nEnglish Training Data:")
    train_paths_en, train_labels_en, _ = load_data_with_language_labels(
        train_csv_en, emotion_to_id, "english"
    )

    print("\nEnglish Test Data:")
    test_paths_en, test_labels_en, _ = load_data_with_language_labels(
        test_csv_en, emotion_to_id, "english"
    )

    # Hindi data
    train_csv_hi = [os.path.join(args.data_dir, 'train_hindi_4class.csv')]
    test_csv_hi = [os.path.join(args.data_dir, 'test_hindi_4class.csv')]

    print("\nHindi Training Data:")
    train_paths_hi, train_labels_hi, _ = load_data_with_language_labels(
        train_csv_hi, emotion_to_id, "hindi"
    )

    print("\nHindi Test Data:")
    test_paths_hi, test_labels_hi, _ = load_data_with_language_labels(
        test_csv_hi, emotion_to_id, "hindi"
    )

    print(f"\n{'='*80}")
    print("Dataset Summary")
    print(f"{'='*80}")
    print(f"English Train: {len(train_paths_en)} samples")
    print(f"English Test: {len(test_paths_en)} samples")
    print(f"Hindi Train: {len(train_paths_hi)} samples")
    print(f"Hindi Test: {len(test_paths_hi)} samples")
    print(f"Total Train: {len(train_paths_en) + len(train_paths_hi)} samples")
    print(f"Total Test: {len(test_paths_en) + len(test_paths_hi)} samples")

    # Load processor
    print(f"\n{'='*80}")
    print("Loading Processor")
    print(f"{'='*80}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)

    # Create dataloaders
    print(f"\n{'='*80}")
    print("Creating DataLoaders")
    print(f"{'='*80}")
    train_loader, test_loader_combined, test_loader_en, test_loader_hi = create_dataloaders(
        train_paths_en, train_labels_en,
        train_paths_hi, train_labels_hi,
        test_paths_en, test_labels_en,
        test_paths_hi, test_labels_hi,
        processor,
        args.batch_size,
        args.max_duration,
        args.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches (combined): {len(test_loader_combined)}")
    print(f"Test batches (English): {len(test_loader_en)}")
    print(f"Test batches (Hindi): {len(test_loader_hi)}")

    # Create model
    print(f"\n{'='*80}")
    print("Creating Model")
    print(f"{'='*80}")
    model = LayerWiseAdversarialHuBERT(
        model_name=args.model_name,
        num_emotions=num_emotions,
        num_languages=2,  # English and Hindi
        adversarial_layers=args.adversarial_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_feature_extractor=args.freeze_feature_extractor,
        gradient_checkpointing=args.gradient_checkpointing
    )
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Mixed precision scaler
    use_amp = not args.no_mixed_precision and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if use_amp:
        print("Mixed precision training: ENABLED")
    else:
        print("Mixed precision training: DISABLED")

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        "model_name": args.model_name,
        "emotions": args.emotions,
        "adversarial_layers": args.adversarial_layers,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "freeze_feature_extractor": args.freeze_feature_extractor,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "emotion_weight": args.emotion_weight,
        "language_weight": args.language_weight,
        "lambda_schedule": args.lambda_schedule,
        "max_duration": args.max_duration
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}\n")

    best_combined_f1 = 0.0
    training_history = []

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch, args.epochs,
            args.emotion_weight, args.language_weight,
            args.lambda_schedule, scaler, use_amp
        )

        # Evaluate on all splits
        eval_combined = evaluate(model, test_loader_combined, device, "Combined")
        eval_en = evaluate(model, test_loader_en, device, "English")
        eval_hi = evaluate(model, test_loader_hi, device, "Hindi")

        # Print results
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Emotion Loss: {train_metrics['emotion_loss']:.4f} | "
              f"Language Loss: {train_metrics['language_loss']:.4f}")
        print(f"Train Emotion Acc: {train_metrics['emotion_acc']:.4f} | "
              f"Train Emotion F1: {train_metrics['emotion_f1']:.4f}")
        print(f"Train Language Acc: {train_metrics['language_acc']:.4f} "
              f"(Lower is better - indicates language invariance)")
        print(f"\nTest Results:")
        print(f"  Combined  - Emotion Acc: {eval_combined['emotion_acc']:.4f} | "
              f"F1: {eval_combined['emotion_f1']:.4f} | "
              f"Language Acc: {eval_combined['language_acc']:.4f}")
        print(f"  English   - Emotion Acc: {eval_en['emotion_acc']:.4f} | "
              f"F1: {eval_en['emotion_f1']:.4f}")
        print(f"  Hindi     - Emotion Acc: {eval_hi['emotion_acc']:.4f} | "
              f"F1: {eval_hi['emotion_f1']:.4f}")

        # Cross-lingual transfer gap
        transfer_gap = abs(eval_en['emotion_f1'] - eval_hi['emotion_f1'])
        print(f"  Cross-lingual Transfer Gap: {transfer_gap:.4f} "
              f"(Lower is better)")

        # Save history
        epoch_history = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "test_combined": eval_combined,
            "test_english": eval_en,
            "test_hindi": eval_hi,
            "transfer_gap": transfer_gap
        }
        training_history.append(epoch_history)

        # Save best model
        if eval_combined['emotion_f1'] > best_combined_f1:
            best_combined_f1 = eval_combined['emotion_f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'emotion_to_id': emotion_to_id,
                'id_to_emotion': id_to_emotion,
                'best_f1': best_combined_f1
            }, output_dir / 'best_model.pth')
            print(f"  *** New best model saved! F1: {best_combined_f1:.4f} ***")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'emotion_to_id': emotion_to_id,
            'id_to_emotion': id_to_emotion
        }, output_dir / f'checkpoint_epoch{epoch+1}.pth')

        print(f"{'='*80}\n")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best Combined F1: {best_combined_f1:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
