"""
hubert_crosslingual_emotion.py - Layer-Wise Adversarial Disentanglement for Cross-Lingual SER

Implementation of:
"Layer-Wise Adversarial Disentanglement for Cross-Lingual Speech Emotion Recognition using HuBERT"

Key Features:
- HuBERT-based emotion recognition
- Layer-wise gradient reversal for language disentanglement
- Cross-lingual training (Hindi-English)
- Adversarial language discriminator
- No synthetic data - real corpora only

Author: Research Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2Processor
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forwards input unchanged, but reverses gradient during backprop.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversal(nn.Module):
    """Gradient Reversal Layer wrapper"""
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class LanguageDiscriminator(nn.Module):
    """
    Language discriminator for adversarial training.
    Tries to predict language from layer representations.
    """
    def __init__(self, input_dim, hidden_dim=256, num_languages=2):
        super(LanguageDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_languages)
        )

    def forward(self, x):
        return self.discriminator(x)


class LayerWiseAdversarialHuBERT(nn.Module):
    """
    HuBERT with Layer-Wise Adversarial Disentanglement

    Architecture:
    - HuBERT base model (12 transformer layers)
    - Extract representations from multiple layers
    - Apply gradient reversal to each layer
    - Language discriminators for each layer
    - Emotion classifier on top layer
    """
    def __init__(self, num_emotions=4, num_languages=2,
                 adversarial_layers=[6, 9, 12],
                 freeze_feature_extractor=True,
                 alpha=1.0):
        super(LayerWiseAdversarialHuBERT, self).__init__()

        self.num_emotions = num_emotions
        self.num_languages = num_languages
        self.adversarial_layers = adversarial_layers
        self.alpha = alpha

        # Load HuBERT model
        print("Loading HuBERT model...")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False

        # Get hidden dimension
        hidden_dim = self.hubert.config.hidden_size  # 768 for base

        # Gradient reversal layers for each adversarial layer
        self.gradient_reversals = nn.ModuleDict({
            str(layer): GradientReversal(alpha=alpha)
            for layer in adversarial_layers
        })

        # Language discriminators for each adversarial layer
        self.language_discriminators = nn.ModuleDict({
            str(layer): LanguageDiscriminator(hidden_dim, hidden_dim//2, num_languages)
            for layer in adversarial_layers
        })

        # Emotion classifier (on final representation)
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_values, output_hidden_states=True, language_labels=None):
        """
        Forward pass with layer-wise adversarial training

        Args:
            input_values: Audio features (batch_size, sequence_length)
            output_hidden_states: Whether to output hidden states from all layers
            language_labels: Language labels for adversarial training

        Returns:
            emotion_logits: Emotion predictions
            language_losses: Dict of language discrimination losses per layer
            layer_outputs: Dict of layer representations (for analysis)
        """
        # Get HuBERT hidden states from all layers
        outputs = self.hubert(
            input_values,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract hidden states from all layers
        all_hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

        # Use final layer for emotion classification
        final_hidden = all_hidden_states[-1]  # Last layer

        # Pool sequence dimension (mean pooling)
        pooled_final = torch.mean(final_hidden, dim=1)  # (batch, hidden)
        pooled_final = self.layer_norm(pooled_final)

        # Emotion classification
        emotion_logits = self.emotion_classifier(pooled_final)

        # Adversarial language discrimination on specified layers
        language_losses = {}
        layer_outputs = {}

        if language_labels is not None:
            for layer_idx in self.adversarial_layers:
                # Get hidden state from this layer
                layer_hidden = all_hidden_states[layer_idx]

                # Pool sequence dimension
                pooled_layer = torch.mean(layer_hidden, dim=1)

                # Apply gradient reversal
                reversed_features = self.gradient_reversals[str(layer_idx)](pooled_layer)

                # Language discrimination
                lang_logits = self.language_discriminators[str(layer_idx)](reversed_features)

                # Compute language discrimination loss
                lang_loss = F.cross_entropy(lang_logits, language_labels)
                language_losses[f'layer_{layer_idx}'] = lang_loss

                # Store layer output for analysis
                layer_outputs[f'layer_{layer_idx}'] = pooled_layer.detach()

        return emotion_logits, language_losses, layer_outputs

    def set_alpha(self, alpha):
        """Update alpha for gradient reversal (for curriculum learning)"""
        self.alpha = alpha
        for grl in self.gradient_reversals.values():
            grl.alpha = alpha


class CrossLingualEmotionDataset(Dataset):
    """
    Dataset for cross-lingual emotion recognition
    Loads Hindi and English audio with emotion and language labels
    """
    def __init__(self, csv_files, processor, emotions, max_duration=5.0,
                 sr=16000, augment=False):
        self.processor = processor
        self.emotions = emotions
        self.emotion_to_int = {e: i for i, e in enumerate(emotions)}
        self.max_duration = max_duration
        self.sr = sr
        self.augment = augment

        # Language mapping
        self.language_to_int = {'english': 0, 'hindi': 1}

        # Load data from CSV files
        self.data = []
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"Warning: {csv_file} not found, skipping...")
                continue

            df = pd.read_csv(csv_file)

            # Determine language from filename
            if 'hindi' in csv_file.lower():
                language = 'hindi'
            else:
                language = 'english'  # RAVDESS and TESS are English

            for _, row in df.iterrows():
                audio_path = row['path']
                emotion = row['emotion']

                if emotion in self.emotions and os.path.exists(audio_path):
                    self.data.append({
                        'path': audio_path,
                        'emotion': emotion,
                        'language': language
                    })

        print(f"Loaded {len(self.data)} samples")

        # Print language distribution
        lang_dist = {}
        for item in self.data:
            lang = item['language']
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        print(f"Language distribution: {lang_dist}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load audio
        speech, sr = librosa.load(item['path'], sr=self.sr)

        # Trim or pad to max_duration
        max_len = int(self.max_duration * self.sr)
        if len(speech) > max_len:
            speech = speech[:max_len]
        else:
            speech = np.pad(speech, (0, max_len - len(speech)))

        # Simple augmentation (optional)
        if self.augment:
            # Random gain
            if np.random.random() < 0.5:
                speech = speech * np.random.uniform(0.8, 1.2)

        # Process with HuBERT processor
        inputs = self.processor(speech, sampling_rate=self.sr, return_tensors="pt")

        return {
            'input_values': inputs.input_values.squeeze(0),
            'emotion_label': self.emotion_to_int[item['emotion']],
            'language_label': self.language_to_int[item['language']]
        }


class CrossLingualEmotionRecognizer:
    """
    Main class for cross-lingual emotion recognition training and inference
    """
    def __init__(self, emotions=['sad', 'neutral', 'happy', 'angry'],
                 adversarial_layers=[6, 9, 12],
                 device=None):
        self.emotions = emotions
        self.adversarial_layers = adversarial_layers

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")

        # Initialize model
        self.model = LayerWiseAdversarialHuBERT(
            num_emotions=len(emotions),
            num_languages=2,
            adversarial_layers=adversarial_layers,
            freeze_feature_extractor=True,
            alpha=1.0
        ).to(self.device)

        self.emotion_to_int = {e: i for i, e in enumerate(emotions)}
        self.int_to_emotion = {i: e for e, i in self.emotion_to_int.items()}

    def train(self, train_csv, val_csv=None, epochs=15, batch_size=8,
              lr=1e-4, max_duration=5.0, lambda_adv=1.0,
              alpha_schedule='constant', save_dir='models'):
        """
        Train the cross-lingual emotion recognition model

        Args:
            train_csv: List of CSV files for training
            val_csv: List of CSV files for validation
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            max_duration: Maximum audio duration in seconds
            lambda_adv: Weight for adversarial loss
            alpha_schedule: Alpha scheduling ('constant', 'progressive')
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create datasets
        print("\nCreating datasets...")
        train_dataset = CrossLingualEmotionDataset(
            train_csv, self.processor, self.emotions,
            max_duration=max_duration, augment=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0
        )

        val_loader = None
        if val_csv:
            val_dataset = CrossLingualEmotionDataset(
                val_csv, self.processor, self.emotions,
                max_duration=max_duration, augment=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=0
            )

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        best_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*70}")

            # Update alpha for gradient reversal (progressive schedule)
            if alpha_schedule == 'progressive':
                alpha = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
                self.model.set_alpha(alpha)
                print(f"Alpha: {alpha:.3f}")

            # Training
            self.model.train()
            train_loss = 0
            train_emotion_loss = 0
            train_lang_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                input_values = batch['input_values'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                language_labels = batch['language_label'].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                emotion_logits, language_losses, _ = self.model(
                    input_values,
                    language_labels=language_labels
                )

                # Emotion classification loss
                emotion_loss = F.cross_entropy(emotion_logits, emotion_labels)

                # Adversarial language losses (averaged across layers)
                avg_lang_loss = torch.mean(torch.stack(list(language_losses.values())))

                # Total loss
                total_loss = emotion_loss + lambda_adv * avg_lang_loss

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # Statistics
                train_loss += total_loss.item()
                train_emotion_loss += emotion_loss.item()
                train_lang_loss += avg_lang_loss.item()

                _, predicted = torch.max(emotion_logits, 1)
                train_correct += (predicted == emotion_labels).sum().item()
                train_total += emotion_labels.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'acc': f'{100*train_correct/train_total:.2f}%'
                })

            # Average training metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_emotion_loss = train_emotion_loss / len(train_loader)
            avg_lang_loss = train_lang_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total

            print(f"\nTraining Results:")
            print(f"  Total Loss: {avg_train_loss:.4f}")
            print(f"  Emotion Loss: {avg_emotion_loss:.4f}")
            print(f"  Language Loss (Adv): {avg_lang_loss:.4f}")
            print(f"  Accuracy: {train_acc:.2f}%")

            # Validation
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                print(f"\nValidation Results:")
                print(f"  Loss: {val_loss:.4f}")
                print(f"  Accuracy: {val_acc:.2f}%")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(save_dir, 'hubert_crosslingual_best.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'emotions': self.emotions,
                        'adversarial_layers': self.adversarial_layers
                    }, save_path)
                    print(f"  ✓ Saved best model to {save_path}")

                # Update learning rate
                scheduler.step(val_loss)

        # Save final model
        final_path = os.path.join(save_dir, 'hubert_crosslingual_final.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'emotions': self.emotions,
            'adversarial_layers': self.adversarial_layers
        }, final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_values = batch['input_values'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                language_labels = batch['language_label'].to(self.device)

                emotion_logits, language_losses, _ = self.model(
                    input_values,
                    language_labels=language_labels
                )

                loss = F.cross_entropy(emotion_logits, emotion_labels)
                val_loss += loss.item()

                _, predicted = torch.max(emotion_logits, 1)
                correct += (predicted == emotion_labels).sum().item()
                total += emotion_labels.size(0)

        return val_loss / len(val_loader), 100 * correct / total

    def predict(self, audio_path):
        """Predict emotion from audio file"""
        self.model.eval()

        # Load and process audio
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            input_values = inputs.input_values.to(self.device)
            emotion_logits, _, _ = self.model(input_values, language_labels=None)

            probs = F.softmax(emotion_logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        return self.int_to_emotion[pred_idx], confidence

    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
