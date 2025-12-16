"""
adversarial_emotion_recognition.py - Adversarial Language Disentanglement for Cross-Lingual SER
Author: Research Implementation

This module implements a novel approach for cross-lingual speech emotion recognition using:
- HuBERT backbone for audio feature extraction
- Layer-wise Gradient Reversal for language disentanglement
- Adversarial training to learn language-invariant emotion representations

Key Innovation:
- First application of adversarial language disentanglement on HuBERT
- Layer-specific gradient reversal (not just final embedding)
- Hindi-English cross-lingual evaluation

Reference Architecture:
    Audio → HuBERT → [Layer 4: Language GRL] → Language Discriminator
                   → [Layer 12: Emotion]    → Emotion Classifier
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import librosa
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for adversarial training.

    Forward pass: Identity function (x → x)
    Backward pass: Reverses gradients (grad → -λ * grad)

    This forces the encoder to learn features that CONFUSE the discriminator,
    resulting in language-invariant representations.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by lambda
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        """Update lambda value during training."""
        self.lambda_ = lambda_


# =============================================================================
# LAMBDA SCHEDULER
# =============================================================================

class LambdaScheduler:
    """
    Scheduler for adversarial weight (lambda).

    Gradually increases lambda during training:
    - Early epochs: Focus on emotion classification (λ ≈ 0)
    - Later epochs: Add language disentanglement (λ → max)

    Uses sigmoid schedule for smooth transition.
    """

    def __init__(self, max_lambda=1.0, gamma=10.0):
        self.max_lambda = max_lambda
        self.gamma = gamma

    def get_lambda(self, epoch, max_epochs):
        """
        Compute lambda based on training progress.

        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total number of epochs

        Returns:
            Lambda value in range [0, max_lambda]
        """
        p = epoch / max_epochs
        lambda_ = self.max_lambda * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)
        return lambda_


# =============================================================================
# LANGUAGE DISCRIMINATOR
# =============================================================================

class LanguageDiscriminator(nn.Module):
    """
    Discriminator network to predict language from features.

    The goal is for HuBERT to learn features that this discriminator
    CANNOT use to distinguish languages (via gradient reversal).
    """

    def __init__(self, input_dim, hidden_dim=256, num_languages=2, dropout=0.1):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_languages)
        )

    def forward(self, x):
        return self.discriminator(x)


# =============================================================================
# EMOTION CLASSIFIER
# =============================================================================

class EmotionClassifier(nn.Module):
    """Classification head for emotion prediction."""

    def __init__(self, input_dim, hidden_dim=256, num_emotions=4, dropout=0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )

    def forward(self, x):
        return self.classifier(x)


# =============================================================================
# MAIN MODEL: ADVERSARIAL HUBERT FOR EMOTION RECOGNITION
# =============================================================================

class AdversarialHuBERTForEmotion(nn.Module):
    """
    HuBERT model with adversarial language disentanglement for emotion recognition.

    Architecture:
        Input Audio
            ↓
        HuBERT Encoder (12 transformer layers)
            ↓
        ┌───────────────────────────────────────┐
        │                                       │
        Layer 4 output                    Layer 12 output
        (language info)                   (emotion info)
            ↓                                   ↓
        Mean Pooling                      Mean Pooling
            ↓                                   ↓
        Gradient Reversal                 Emotion Classifier
            ↓                                   ↓
        Language Discriminator            Emotion Prediction
            ↓
        Language Prediction
        (to be confused)

    Training Objective:
        - Minimize emotion classification loss
        - Maximize language discriminator loss (via GRL)
        - Result: Language-invariant emotion features
    """

    def __init__(
        self,
        num_emotions,
        num_languages=2,
        model_name="facebook/hubert-base-ls960",
        freeze_feature_extractor=True,
        adversarial_layer=4,
        emotion_layer=12,
        hidden_dim=256,
        dropout=0.1,
        initial_lambda=0.0
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            num_languages: Number of languages (default: 2 for Hindi/English)
            model_name: HuBERT model identifier
            freeze_feature_extractor: Freeze CNN feature extractor
            adversarial_layer: Layer to extract features for language disc (1-12)
            emotion_layer: Layer to extract features for emotion (1-12)
            hidden_dim: Hidden dimension for classifiers
            dropout: Dropout rate
            initial_lambda: Initial adversarial weight
        """
        super().__init__()

        self.num_emotions = num_emotions
        self.num_languages = num_languages
        self.adversarial_layer = adversarial_layer
        self.emotion_layer = emotion_layer

        # Load HuBERT model
        if "hubert" in model_name.lower():
            self.encoder = HubertModel.from_pretrained(
                model_name,
                output_hidden_states=True
            )
        else:
            # Fall back to Wav2Vec2 for compatibility
            self.encoder = Wav2Vec2Model.from_pretrained(
                model_name,
                output_hidden_states=True
            )

        # Freeze feature extractor (CNN layers)
        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()

        # Get hidden size from model config
        hidden_size = self.encoder.config.hidden_size  # 768 for base models

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_=initial_lambda)

        # Language Discriminator (applied after GRL)
        self.language_discriminator = LanguageDiscriminator(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_languages=num_languages,
            dropout=dropout
        )

        # Emotion Classifier
        self.emotion_classifier = EmotionClassifier(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            dropout=dropout
        )

        print(f"[Model] AdversarialHuBERT initialized")
        print(f"[Model] Adversarial layer: {adversarial_layer}, Emotion layer: {emotion_layer}")
        print(f"[Model] Emotions: {num_emotions}, Languages: {num_languages}")

    def set_lambda(self, lambda_):
        """Update adversarial weight."""
        self.grl.set_lambda(lambda_)

    def forward(self, input_values, attention_mask=None):
        """
        Forward pass.

        Args:
            input_values: Raw audio waveform [batch, seq_len]
            attention_mask: Optional attention mask

        Returns:
            emotion_logits: Emotion predictions [batch, num_emotions]
            language_logits: Language predictions [batch, num_languages]
        """
        # Get all hidden states from HuBERT
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # hidden_states is a tuple: (embedding, layer1, layer2, ..., layer12)
        hidden_states = outputs.hidden_states

        # Extract features from specified layers
        # Note: hidden_states[0] is embedding, hidden_states[1] is layer 1, etc.
        adversarial_features = hidden_states[self.adversarial_layer]  # For language disc
        emotion_features = hidden_states[self.emotion_layer]          # For emotion class

        # Mean pooling over time dimension
        adversarial_pooled = adversarial_features.mean(dim=1)  # [batch, hidden_size]
        emotion_pooled = emotion_features.mean(dim=1)          # [batch, hidden_size]

        # Apply Gradient Reversal before language discriminator
        adversarial_reversed = self.grl(adversarial_pooled)

        # Predictions
        language_logits = self.language_discriminator(adversarial_reversed)
        emotion_logits = self.emotion_classifier(emotion_pooled)

        return emotion_logits, language_logits

    def predict_emotion(self, input_values, attention_mask=None):
        """Predict only emotion (for inference)."""
        emotion_logits, _ = self.forward(input_values, attention_mask)
        return emotion_logits


# =============================================================================
# DATASET
# =============================================================================

class CrossLingualEmotionDataset(Dataset):
    """
    Dataset for cross-lingual emotion recognition.

    Each sample includes:
    - Audio waveform
    - Emotion label
    - Language label
    """

    def __init__(
        self,
        audio_paths,
        emotion_labels,
        language_labels,
        processor,
        max_duration=5.0,
        sample_rate=16000
    ):
        """
        Args:
            audio_paths: List of paths to audio files
            emotion_labels: List of emotion label indices
            language_labels: List of language label indices (0=English, 1=Hindi)
            processor: Wav2Vec2/HuBERT feature extractor
            max_duration: Maximum audio duration in seconds
            sample_rate: Target sample rate
        """
        self.audio_paths = audio_paths
        self.emotion_labels = emotion_labels
        self.language_labels = language_labels
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        emotion_label = self.emotion_labels[idx]
        language_label = self.language_labels[idx]

        try:
            # Load audio
            speech, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Pad or truncate
            if len(speech) > self.max_length:
                speech = speech[:self.max_length]
            else:
                speech = np.pad(speech, (0, self.max_length - len(speech)))

            # Process with feature extractor
            inputs = self.processor(
                speech,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            return {
                'input_values': inputs.input_values.squeeze(0),
                'emotion_label': emotion_label,
                'language_label': language_label
            }

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return {
                'input_values': torch.zeros(self.max_length),
                'emotion_label': emotion_label,
                'language_label': language_label
            }


# =============================================================================
# TRAINER
# =============================================================================

class AdversarialEmotionTrainer:
    """
    Trainer for adversarial cross-lingual emotion recognition.

    Handles:
    - Mixed-language batch training
    - Lambda scheduling
    - Adversarial loss computation
    - Evaluation across languages
    """

    def __init__(
        self,
        emotions,
        languages=['english', 'hindi'],
        model_name="facebook/hubert-base-ls960",
        max_duration=5.0,
        sample_rate=16000,
        device=None
    ):
        """
        Args:
            emotions: List of emotion labels
            languages: List of language names
            model_name: Pretrained model to use
            max_duration: Maximum audio duration
            sample_rate: Audio sample rate
            device: Device to use
        """
        self.emotions = emotions
        self.languages = languages
        self.num_emotions = len(emotions)
        self.num_languages = len(languages)
        self.model_name = model_name
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Mappings
        self.emotion_to_id = {e: i for i, e in enumerate(emotions)}
        self.id_to_emotion = {i: e for e, i in self.emotion_to_id.items()}
        self.language_to_id = {l: i for i, l in enumerate(languages)}
        self.id_to_language = {i: l for l, i in self.language_to_id.items()}

        # Initialize processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Model will be built later
        self.model = None
        self.lambda_scheduler = LambdaScheduler(max_lambda=1.0)

        print(f"[Trainer] Initialized for {self.num_emotions} emotions, {self.num_languages} languages")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Model: {model_name}")

    def build_model(
        self,
        freeze_feature_extractor=True,
        adversarial_layer=4,
        emotion_layer=12,
        hidden_dim=256,
        dropout=0.1
    ):
        """Build the adversarial model."""
        self.model = AdversarialHuBERTForEmotion(
            num_emotions=self.num_emotions,
            num_languages=self.num_languages,
            model_name=self.model_name,
            freeze_feature_extractor=freeze_feature_extractor,
            adversarial_layer=adversarial_layer,
            emotion_layer=emotion_layer,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"[Model] Total parameters: {total_params:,}")
        print(f"[Model] Trainable parameters: {trainable_params:,}")

    def prepare_data(
        self,
        train_paths,
        train_emotions,
        train_languages,
        test_paths,
        test_emotions,
        test_languages,
        batch_size=8,
        num_workers=2
    ):
        """Prepare data loaders with language labels."""
        # Convert labels to indices
        train_emotion_ids = [self.emotion_to_id[e] for e in train_emotions]
        train_language_ids = [self.language_to_id[l] for l in train_languages]
        test_emotion_ids = [self.emotion_to_id[e] for e in test_emotions]
        test_language_ids = [self.language_to_id[l] for l in test_languages]

        # Create datasets
        train_dataset = CrossLingualEmotionDataset(
            train_paths, train_emotion_ids, train_language_ids,
            self.processor, self.max_duration, self.sample_rate
        )

        test_dataset = CrossLingualEmotionDataset(
            test_paths, test_emotion_ids, test_language_ids,
            self.processor, self.max_duration, self.sample_rate
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

    def train(
        self,
        epochs=20,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        max_lambda=1.0,
        use_mixed_precision=True,
        save_best_model=True,
        model_path='models/adversarial_hubert_best.pt'
    ):
        """
        Train with adversarial language disentanglement.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate (lower for adversarial stability)
            warmup_ratio: Warmup ratio for LR scheduler
            max_lambda: Maximum adversarial weight
            use_mixed_precision: Use FP16 training
            save_best_model: Save best model checkpoint
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Update lambda scheduler
        self.lambda_scheduler = LambdaScheduler(max_lambda=max_lambda)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Setup LR scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Loss functions
        emotion_criterion = nn.CrossEntropyLoss()
        language_criterion = nn.CrossEntropyLoss()

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device.type == 'cuda' else None

        # Training loop
        best_accuracy = 0.0
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

        training_history = {
            'emotion_loss': [],
            'language_loss': [],
            'lambda': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            # Get current lambda
            current_lambda = self.lambda_scheduler.get_lambda(epoch, epochs)
            self.model.set_lambda(current_lambda)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs} | Lambda: {current_lambda:.4f}")
            print(f"{'='*60}")

            # Training phase
            self.model.train()
            train_emotion_loss = 0.0
            train_language_loss = 0.0
            train_emotion_preds = []
            train_emotion_labels = []
            train_language_preds = []
            train_language_labels = []

            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                input_values = batch['input_values'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                language_labels = batch['language_label'].to(self.device)

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        emotion_logits, language_logits = self.model(input_values)

                        # Compute losses
                        e_loss = emotion_criterion(emotion_logits, emotion_labels)
                        l_loss = language_criterion(language_logits, language_labels)

                        # Total loss (GRL handles the adversarial aspect internally)
                        total_loss = e_loss + l_loss

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    emotion_logits, language_logits = self.model(input_values)

                    e_loss = emotion_criterion(emotion_logits, emotion_labels)
                    l_loss = language_criterion(language_logits, language_labels)

                    total_loss = e_loss + l_loss
                    total_loss.backward()
                    optimizer.step()

                scheduler.step()

                # Record metrics
                train_emotion_loss += e_loss.item()
                train_language_loss += l_loss.item()

                e_preds = torch.argmax(emotion_logits, dim=1)
                l_preds = torch.argmax(language_logits, dim=1)

                train_emotion_preds.extend(e_preds.cpu().numpy())
                train_emotion_labels.extend(emotion_labels.cpu().numpy())
                train_language_preds.extend(l_preds.cpu().numpy())
                train_language_labels.extend(language_labels.cpu().numpy())

                pbar.set_postfix({
                    'e_loss': f'{e_loss.item():.4f}',
                    'l_loss': f'{l_loss.item():.4f}'
                })

            # Calculate training metrics
            train_emotion_loss /= len(self.train_loader)
            train_language_loss /= len(self.train_loader)
            train_emotion_acc = accuracy_score(train_emotion_labels, train_emotion_preds)
            train_language_acc = accuracy_score(train_language_labels, train_language_preds)

            print(f"\nTraining:")
            print(f"  Emotion   - Loss: {train_emotion_loss:.4f}, Acc: {train_emotion_acc:.4f}")
            print(f"  Language  - Loss: {train_language_loss:.4f}, Acc: {train_language_acc:.4f}")
            print(f"  (Lower language acc = better disentanglement)")

            # Validation phase
            val_results = self.evaluate()
            print(f"\nValidation:")
            print(f"  Emotion   - Acc: {val_results['emotion_acc']:.4f}, F1: {val_results['emotion_f1']:.4f}")
            print(f"  Language  - Acc: {val_results['language_acc']:.4f}")

            # Record history
            training_history['emotion_loss'].append(train_emotion_loss)
            training_history['language_loss'].append(train_language_loss)
            training_history['lambda'].append(current_lambda)
            training_history['train_acc'].append(train_emotion_acc)
            training_history['val_acc'].append(val_results['emotion_acc'])

            # Save best model
            if save_best_model and val_results['emotion_acc'] > best_accuracy:
                best_accuracy = val_results['emotion_acc']
                self.save_model(model_path)
                print(f"[BEST] Model saved with emotion accuracy: {best_accuracy:.4f}")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Emotion Accuracy: {best_accuracy:.4f}")
        print(f"{'='*60}")

        return training_history

    def evaluate(self, data_loader=None):
        """Evaluate model on test set."""
        if self.model is None:
            raise ValueError("Model not built.")

        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()

        all_emotion_preds = []
        all_emotion_labels = []
        all_language_preds = []
        all_language_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_values = batch['input_values'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                language_labels = batch['language_label'].to(self.device)

                emotion_logits, language_logits = self.model(input_values)

                e_preds = torch.argmax(emotion_logits, dim=1)
                l_preds = torch.argmax(language_logits, dim=1)

                all_emotion_preds.extend(e_preds.cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_language_preds.extend(l_preds.cpu().numpy())
                all_language_labels.extend(language_labels.cpu().numpy())

        results = {
            'emotion_acc': accuracy_score(all_emotion_labels, all_emotion_preds),
            'emotion_f1': f1_score(all_emotion_labels, all_emotion_preds, average='weighted'),
            'language_acc': accuracy_score(all_language_labels, all_language_preds),
            'emotion_preds': all_emotion_preds,
            'emotion_labels': all_emotion_labels,
            'language_preds': all_language_preds,
            'language_labels': all_language_labels
        }

        return results

    def evaluate_cross_lingual(self, test_paths, test_emotions, test_languages, batch_size=8):
        """
        Evaluate cross-lingual performance.

        Useful for testing:
        - Train on English, test on Hindi
        - Train on Hindi, test on English
        """
        test_emotion_ids = [self.emotion_to_id[e] for e in test_emotions]
        test_language_ids = [self.language_to_id[l] for l in test_languages]

        test_dataset = CrossLingualEmotionDataset(
            test_paths, test_emotion_ids, test_language_ids,
            self.processor, self.max_duration, self.sample_rate
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return self.evaluate(test_loader)

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
            emotion_logits = self.model.predict_emotion(input_values)
            probs = torch.nn.functional.softmax(emotion_logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()

        emotion = self.id_to_emotion[pred_id]
        confidence = probs[0][pred_id].item()

        return emotion, confidence

    def get_confusion_matrices(self):
        """Get confusion matrices for both emotion and language."""
        results = self.evaluate()

        emotion_cm = confusion_matrix(results['emotion_labels'], results['emotion_preds'])
        language_cm = confusion_matrix(results['language_labels'], results['language_preds'])

        return {
            'emotion': emotion_cm,
            'language': language_cm
        }

    def save_model(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'emotions': self.emotions,
            'languages': self.languages,
            'model_name': self.model_name,
            'max_duration': self.max_duration,
            'sample_rate': self.sample_rate,
            'emotion_to_id': self.emotion_to_id,
            'id_to_emotion': self.id_to_emotion,
            'language_to_id': self.language_to_id,
            'id_to_language': self.id_to_language,
            'adversarial_layer': self.model.adversarial_layer,
            'emotion_layer': self.model.emotion_layer
        }

        torch.save(checkpoint, path)
        print(f"[Save] Model saved to {path}")

    def load_model(self, path, map_location=None):
        """Load model checkpoint."""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # Restore configuration
        self.emotions = checkpoint['emotions']
        self.languages = checkpoint['languages']
        self.model_name = checkpoint['model_name']
        self.max_duration = checkpoint['max_duration']
        self.sample_rate = checkpoint['sample_rate']
        self.emotion_to_id = checkpoint['emotion_to_id']
        self.id_to_emotion = checkpoint['id_to_emotion']
        self.language_to_id = checkpoint['language_to_id']
        self.id_to_language = checkpoint['id_to_language']
        self.num_emotions = len(self.emotions)
        self.num_languages = len(self.languages)

        # Rebuild and load model
        self.build_model(
            adversarial_layer=checkpoint.get('adversarial_layer', 4),
            emotion_layer=checkpoint.get('emotion_layer', 12)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"[Load] Model loaded from {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Adversarial Emotion Recognition module loaded successfully.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
