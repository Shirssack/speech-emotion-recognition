"""
transformer_emotion_recognition.py - Transformer-based Speech Emotion Recognition
Author: Shirssack

This module implements a Wav2Vec2-based emotion recognition system with
memory-efficient training for 6GB GPU constraint.

Features:
- Wav2Vec2 backbone for powerful audio feature extraction
- Gradient checkpointing for reduced memory usage
- Mixed precision training (FP16)
- Support for multilingual datasets (English + Hindi)
- Fine-tuning capabilities
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import librosa
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


class Wav2Vec2EmotionDataset(Dataset):
    """Dataset for loading audio files directly for transformer models."""

    def __init__(self, audio_paths, labels, processor, max_duration=5.0, sample_rate=16000):
        """
        Args:
            audio_paths: List of paths to audio files
            labels: List of emotion labels
            processor: Wav2Vec2 processor/feature extractor
            max_duration: Maximum duration in seconds
            sample_rate: Target sample rate
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        try:
            # Load audio
            speech, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Pad or truncate to max_length
            if len(speech) > self.max_length:
                speech = speech[:self.max_length]
            else:
                speech = np.pad(speech, (0, self.max_length - len(speech)))

            # Process with Wav2Vec2 processor
            inputs = self.processor(
                speech,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            return {
                'input_values': inputs.input_values.squeeze(0),
                'label': label
            }

        except Exception as e:
            # Return zeros if file fails to load
            return {
                'input_values': torch.zeros(self.max_length),
                'label': label
            }


class Wav2Vec2ForEmotionClassification(nn.Module):
    """Wav2Vec2 model with classification head for emotion recognition."""

    def __init__(self, num_labels, model_name="facebook/wav2vec2-base",
                 freeze_feature_extractor=True, use_weighted_layer_sum=False):
        """
        Args:
            num_labels: Number of emotion classes
            model_name: Pretrained Wav2Vec2 model name
            freeze_feature_extractor: Whether to freeze the CNN feature extractor
            use_weighted_layer_sum: Use weighted sum of all hidden layers
        """
        super().__init__()

        self.num_labels = num_labels
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # Load pretrained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            gradient_checkpointing=True  # Enable gradient checkpointing
        )

        # Freeze feature extractor (CNN layers) if specified
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # Hidden size from the model config
        hidden_size = self.wav2vec2.config.hidden_size

        # Weighted layer sum (if enabled)
        if use_weighted_layer_sum:
            num_layers = self.wav2vec2.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: Raw audio waveform
            attention_mask: Attention mask for padding
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.use_weighted_layer_sum
        )

        # Get hidden states
        if self.use_weighted_layer_sum:
            # Weighted sum of all layers
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack(hidden_states, dim=0)
            norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1)
            weighted_sum = (stacked_hidden_states * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)
            hidden_state = weighted_sum
        else:
            # Use last layer
            hidden_state = outputs.last_hidden_state

        # Mean pooling over time dimension
        pooled_output = hidden_state.mean(dim=1)

        # Classification
        logits = self.classifier(pooled_output)

        return logits


class TransformerEmotionRecognizer:
    """Main class for transformer-based emotion recognition."""

    def __init__(self, emotions, model_name="facebook/wav2vec2-base",
                 max_duration=5.0, sample_rate=16000, device=None):
        """
        Args:
            emotions: List of emotion labels
            model_name: Pretrained model to use
            max_duration: Maximum audio duration in seconds
            sample_rate: Audio sample rate
            device: Device to use (cuda/cpu)
        """
        self.emotions = emotions
        self.num_labels = len(emotions)
        self.model_name = model_name
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Emotion mappings
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotions)}
        self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}

        # Initialize processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None

        print(f"[Transformer] Initialized with {self.num_labels} emotions")
        print(f"[Transformer] Device: {self.device}")
        print(f"[Transformer] Model: {model_name}")

    def build_model(self, freeze_feature_extractor=True, use_weighted_layer_sum=False):
        """Build the model architecture."""
        self.model = Wav2Vec2ForEmotionClassification(
            num_labels=self.num_labels,
            model_name=self.model_name,
            freeze_feature_extractor=freeze_feature_extractor,
            use_weighted_layer_sum=use_weighted_layer_sum
        )
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"[Model] Total parameters: {total_params:,}")
        print(f"[Model] Trainable parameters: {trainable_params:,}")

    def prepare_data(self, train_paths, train_labels, test_paths, test_labels,
                     batch_size=8, num_workers=2):
        """Prepare data loaders."""
        # Convert labels to integers
        train_label_ids = [self.emotion_to_id[label] for label in train_labels]
        test_label_ids = [self.emotion_to_id[label] for label in test_labels]

        # Create datasets
        train_dataset = Wav2Vec2EmotionDataset(
            train_paths, train_label_ids, self.processor,
            max_duration=self.max_duration, sample_rate=self.sample_rate
        )

        test_dataset = Wav2Vec2EmotionDataset(
            test_paths, test_label_ids, self.processor,
            max_duration=self.max_duration, sample_rate=self.sample_rate
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True if self.device.type == 'cuda' else False
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"[Data] Train samples: {len(train_dataset)}")
        print(f"[Data] Test samples: {len(test_dataset)}")
        print(f"[Data] Batch size: {batch_size}")

    def train(self, epochs=10, learning_rate=3e-5, warmup_ratio=0.1,
              use_mixed_precision=True, save_best_model=True, model_path='models/transformer_best.pt'):
        """
        Train the model with memory-efficient techniques.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for learning rate scheduler
            use_mixed_precision: Use FP16 mixed precision training
            save_best_model: Save best model based on validation accuracy
            model_path: Path to save the best model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device.type == 'cuda' else None

        # Training loop
        best_accuracy = 0.0
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                input_values = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_values)
                        loss = criterion(logits, labels)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    logits = self.model(input_values)
                    loss = criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()

                # Record metrics
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Calculate training metrics
            train_loss /= len(self.train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')

            print(f"\nTraining   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

            # Validation phase
            val_acc, val_f1, val_loss = self.evaluate()
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Save best model
            if save_best_model and val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_model(model_path)
                print(f"[BEST] Model saved with accuracy: {best_accuracy:.4f}")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        print(f"{'='*60}")

    def evaluate(self):
        """Evaluate the model on test set."""
        if self.model is None:
            raise ValueError("Model not built.")

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_values = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_values)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return accuracy, f1, avg_loss

    def predict(self, audio_path):
        """Predict emotion for a single audio file."""
        if self.model is None:
            raise ValueError("Model not built.")

        self.model.eval()

        # Load and process audio
        speech, sr = librosa.load(audio_path, sr=self.sample_rate)
        max_length = int(self.sample_rate * self.max_duration)

        if len(speech) > max_length:
            speech = speech[:max_length]
        else:
            speech = np.pad(speech, (0, max_length - len(speech)))

        inputs = self.processor(
            speech,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()

        emotion = self.id_to_emotion[pred_id]
        confidence = probs[0][pred_id].item()

        return emotion, confidence

    def get_confusion_matrix(self):
        """Get confusion matrix on test set."""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_values = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_values)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return confusion_matrix(all_labels, all_preds)

    def save_model(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'emotions': self.emotions,
            'model_name': self.model_name,
            'max_duration': self.max_duration,
            'sample_rate': self.sample_rate,
            'emotion_to_id': self.emotion_to_id,
            'id_to_emotion': self.id_to_emotion
        }

        torch.save(checkpoint, path)
        print(f"[Save] Model saved to {path}")

    def load_model(self, path, map_location=None):
        """Load model checkpoint."""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        self.emotions = checkpoint['emotions']
        self.model_name = checkpoint['model_name']
        self.max_duration = checkpoint['max_duration']
        self.sample_rate = checkpoint['sample_rate']
        self.emotion_to_id = checkpoint['emotion_to_id']
        self.id_to_emotion = checkpoint['id_to_emotion']
        self.num_labels = len(self.emotions)

        # Rebuild model
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"[Load] Model loaded from {path}")


if __name__ == "__main__":
    print("Transformer Emotion Recognition module loaded successfully.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
