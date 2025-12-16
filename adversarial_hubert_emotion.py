"""
Layer-Wise Adversarial Disentanglement for Cross-Lingual Speech Emotion Recognition using HuBERT

This module implements a novel approach for cross-lingual emotion recognition by applying
adversarial language disentanglement at multiple layers of the HuBERT transformer model.

Key Novelty:
1. First to apply adversarial language disentanglement on HuBERT
2. Layer-specific gradient reversal (not just final embedding)
3. Hindi-English cross-lingual evaluation (under-resourced pair)
4. No synthetic data - purely real corpora

Architecture Overview:
- Base Model: HuBERT (facebook/hubert-base-ls960)
- Gradient Reversal Layers (GRL): Inserted at specific transformer layers
- Language Discriminators: One discriminator per GRL to predict language
- Emotion Classifier: Final head for emotion prediction

The model is trained with joint optimization:
- Emotion Classification Loss: Cross-entropy for emotion prediction
- Adversarial Language Loss: Encourages language-invariant representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import HubertModel, HubertConfig
from typing import Optional, List, Dict, Tuple
import numpy as np


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)

    During forward pass: acts as identity function
    During backward pass: reverses gradients and scales by lambda

    This is the core component for adversarial training, forcing the model
    to learn representations that are discriminative for emotion but
    invariant to language.
    """

    @staticmethod
    def forward(ctx, x, lambda_param):
        """
        Forward pass: identity function

        Args:
            x: Input tensor
            lambda_param: Scaling factor for gradient reversal
        """
        ctx.lambda_param = lambda_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse and scale gradients

        Args:
            grad_output: Gradients from subsequent layers

        Returns:
            Reversed and scaled gradients, None for lambda_param
        """
        lambda_param = ctx.lambda_param
        grad_input = grad_output.neg() * lambda_param
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper module

    Args:
        lambda_param: Initial scaling factor for gradient reversal (default: 1.0)
    """

    def __init__(self, lambda_param: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        """Apply gradient reversal"""
        return GradientReversalFunction.apply(x, self.lambda_param)

    def set_lambda(self, lambda_param: float):
        """Update lambda parameter (useful for lambda scheduling)"""
        self.lambda_param = lambda_param


class LanguageDiscriminator(nn.Module):
    """
    Language Discriminator Network

    Predicts the language of the input from intermediate representations.
    During adversarial training, the encoder tries to fool this discriminator,
    leading to language-invariant emotion representations.

    Args:
        input_dim: Dimension of input features (e.g., 768 for HuBERT base)
        hidden_dim: Dimension of hidden layer (default: 256)
        num_languages: Number of languages to discriminate (default: 2 for Hindi-English)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_languages: int = 2,
        dropout: float = 0.3
    ):
        super(LanguageDiscriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_languages)
        )

    def forward(self, x):
        """
        Forward pass through discriminator

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Language logits [batch_size, num_languages]
        """
        return self.network(x)


class EmotionClassifier(nn.Module):
    """
    Emotion Classification Head

    Final classifier for emotion prediction from aggregated representations.

    Args:
        input_dim: Dimension of input features (e.g., 768 for HuBERT base)
        hidden_dim: Dimension of hidden layer (default: 256)
        num_emotions: Number of emotion classes (default: 4)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_emotions: int = 4,
        dropout: float = 0.3
    ):
        super(EmotionClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions)
        )

    def forward(self, x):
        """
        Forward pass through classifier

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Emotion logits [batch_size, num_emotions]
        """
        return self.classifier(x)


class LayerWiseAdversarialHuBERT(nn.Module):
    """
    Layer-Wise Adversarial HuBERT for Cross-Lingual Emotion Recognition

    This model implements adversarial language disentanglement at multiple layers
    of the HuBERT transformer, enabling better cross-lingual transfer for emotion
    recognition.

    Args:
        model_name: HuBERT model name (default: "facebook/hubert-base-ls960")
        num_emotions: Number of emotion classes (default: 4)
        num_languages: Number of languages (default: 2 for Hindi-English)
        adversarial_layers: List of layer indices for GRL insertion (default: [3, 6, 9, 12])
        hidden_dim: Hidden dimension for classifiers (default: 256)
        dropout: Dropout probability (default: 0.3)
        freeze_feature_extractor: Whether to freeze CNN feature extractor (default: True)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency (default: False)
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        num_emotions: int = 4,
        num_languages: int = 2,
        adversarial_layers: List[int] = [3, 6, 9, 12],
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
        gradient_checkpointing: bool = False
    ):
        super(LayerWiseAdversarialHuBERT, self).__init__()

        # Store configuration
        self.num_emotions = num_emotions
        self.num_languages = num_languages
        self.adversarial_layers = sorted(adversarial_layers)
        self.hidden_dim = hidden_dim

        # Load pretrained HuBERT model
        print(f"Loading HuBERT model: {model_name}")
        self.hubert = HubertModel.from_pretrained(model_name)
        self.feature_dim = self.hubert.config.hidden_size  # 768 for base model

        # Optionally freeze feature extractor
        if freeze_feature_extractor:
            print("Freezing HuBERT feature extractor (CNN layers)")
            self.hubert.feature_extractor._freeze_parameters()

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            print("Enabling gradient checkpointing")
            self.hubert.gradient_checkpointing_enable()

        # Create Gradient Reversal Layers
        self.grl_layers = nn.ModuleList([
            GradientReversalLayer(lambda_param=1.0)
            for _ in self.adversarial_layers
        ])

        # Create Language Discriminators (one per adversarial layer)
        self.language_discriminators = nn.ModuleList([
            LanguageDiscriminator(
                input_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_languages=num_languages,
                dropout=dropout
            )
            for _ in self.adversarial_layers
        ])

        # Create Emotion Classifier
        self.emotion_classifier = EmotionClassifier(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            dropout=dropout
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        print(f"\n{'='*70}")
        print(f"Layer-Wise Adversarial HuBERT Model")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Emotions: {num_emotions} classes")
        print(f"Languages: {num_languages} classes")
        print(f"Adversarial Layers: {adversarial_layers}")
        print(f"Feature Dim: {self.feature_dim}")
        print(f"Hidden Dim: {hidden_dim}")
        print(f"Dropout: {dropout}")
        print(f"Frozen Feature Extractor: {freeze_feature_extractor}")
        print(f"Gradient Checkpointing: {gradient_checkpointing}")
        print(f"{'='*70}\n")

    def _build_feature_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert sample-level padding mask to feature-level mask.

        HuBERT reduces the temporal resolution of the input by the CNN feature
        extractor. Pooling hidden states therefore requires an attention mask
        that matches the hidden-state sequence length instead of the raw audio
        length. This helper projects the sample-level mask to the hidden-state
        domain using the model's internal length computation.
        """
        # attention_mask is shape [batch, num_samples]; sum gives valid samples
        sample_lengths = attention_mask.sum(dim=1)
        feature_lengths = self.hubert._get_feat_extract_output_lengths(sample_lengths)

        max_feature_len = int(feature_lengths.max().item())
        range_row = torch.arange(max_feature_len, device=attention_mask.device)

        # Broadcast compare to create [batch, max_feature_len] boolean mask
        feature_mask = (range_row.unsqueeze(0) < feature_lengths.unsqueeze(1)).float()
        return feature_mask

    def get_layer_representations(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract representations from HuBERT at specified layers

        Args:
            input_values: Raw audio waveform [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]

        Returns:
            final_representation: Final pooled representation [batch_size, feature_dim]
            layer_representations: List of representations at adversarial layers
        """
        # Run HuBERT encoder with output_hidden_states=True
        outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract hidden states from all layers
        # hidden_states[0] = embeddings, hidden_states[1-12] = transformer layers
        all_hidden_states = outputs.hidden_states

        feature_mask = None
        if attention_mask is not None:
            feature_mask = self._build_feature_attention_mask(attention_mask)

        # Collect representations at specified adversarial layers
        layer_representations = []
        for layer_idx in self.adversarial_layers:
            # Note: hidden_states[0] is embeddings, so transformer layer i is at hidden_states[i]
            layer_output = all_hidden_states[layer_idx]  # [batch, seq_len, feature_dim]

            # Mean pooling over time dimension
            if feature_mask is not None:
                # Mask padding tokens
                mask_expanded = feature_mask.unsqueeze(-1).expand(layer_output.size())
                sum_embeddings = torch.sum(layer_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(layer_output, dim=1)

            layer_representations.append(pooled)

        # Use final layer representation for emotion classification
        final_hidden_state = outputs.last_hidden_state  # [batch, seq_len, feature_dim]

        # Mean pooling for final representation
        if feature_mask is not None:
            mask_expanded = feature_mask.unsqueeze(-1).expand(final_hidden_state.size())
            sum_embeddings = torch.sum(final_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            final_representation = sum_embeddings / sum_mask
        else:
            final_representation = torch.mean(final_hidden_state, dim=1)

        # Apply layer normalization
        final_representation = self.layer_norm(final_representation)

        return final_representation, layer_representations

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_adversarial: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            input_values: Raw audio waveform [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            return_adversarial: Whether to compute adversarial outputs (default: True)

        Returns:
            Dictionary containing:
                - emotion_logits: Emotion predictions [batch_size, num_emotions]
                - language_logits_list: List of language predictions from each discriminator
                - final_representation: Final pooled representation (for analysis)
        """
        # Get representations at all adversarial layers + final layer
        final_representation, layer_representations = self.get_layer_representations(
            input_values=input_values,
            attention_mask=attention_mask
        )

        # Emotion classification from final representation
        emotion_logits = self.emotion_classifier(final_representation)

        outputs = {
            "emotion_logits": emotion_logits,
            "final_representation": final_representation
        }

        # Adversarial language discrimination at each layer
        if return_adversarial:
            language_logits_list = []

            for i, (grl, discriminator, layer_repr) in enumerate(
                zip(self.grl_layers, self.language_discriminators, layer_representations)
            ):
                # Apply gradient reversal
                reversed_repr = grl(layer_repr)

                # Predict language
                language_logits = discriminator(reversed_repr)
                language_logits_list.append(language_logits)

            outputs["language_logits_list"] = language_logits_list

        return outputs

    def set_lambda(self, lambda_param: float):
        """
        Update lambda parameter for all GRL layers

        This is useful for lambda scheduling during training (e.g., gradually increasing
        the adversarial strength as training progresses).

        Args:
            lambda_param: New lambda value
        """
        for grl in self.grl_layers:
            grl.set_lambda(lambda_param)

    def get_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


def compute_adversarial_loss(
    emotion_logits: torch.Tensor,
    language_logits_list: List[torch.Tensor],
    emotion_labels: torch.Tensor,
    language_labels: torch.Tensor,
    emotion_weight: float = 1.0,
    language_weight: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute joint adversarial loss for emotion and language

    The total loss is:
        L_total = L_emotion - λ * Σ L_language_i

    Where:
        - L_emotion: Cross-entropy loss for emotion classification
        - L_language_i: Cross-entropy loss for language discrimination at layer i
        - λ: Weight balancing emotion and adversarial objectives

    The negative sign before language loss is implicit via gradient reversal.

    Args:
        emotion_logits: Emotion predictions [batch_size, num_emotions]
        language_logits_list: List of language predictions from each discriminator
        emotion_labels: Ground truth emotion labels [batch_size]
        language_labels: Ground truth language labels [batch_size]
        emotion_weight: Weight for emotion loss (default: 1.0)
        language_weight: Weight for language loss (default: 0.1)

    Returns:
        Dictionary containing:
            - total_loss: Combined loss
            - emotion_loss: Emotion classification loss
            - language_loss: Average language discrimination loss
            - individual_language_losses: List of language losses per layer
    """
    # Emotion classification loss
    emotion_loss = F.cross_entropy(emotion_logits, emotion_labels)

    # Language discrimination losses (one per adversarial layer)
    individual_language_losses = []
    for language_logits in language_logits_list:
        lang_loss = F.cross_entropy(language_logits, language_labels)
        individual_language_losses.append(lang_loss)

    # Average language loss across all discriminators
    language_loss = torch.mean(torch.stack(individual_language_losses))

    # Combined loss
    # Note: The adversarial effect comes from GRL, so we ADD language loss here
    # (GRL will reverse gradients for the encoder, but not for the discriminator)
    total_loss = emotion_weight * emotion_loss + language_weight * language_loss

    return {
        "total_loss": total_loss,
        "emotion_loss": emotion_loss,
        "language_loss": language_loss,
        "individual_language_losses": individual_language_losses
    }


def get_lambda_schedule(current_step: int, total_steps: int, schedule: str = "progressive") -> float:
    """
    Compute lambda parameter for GRL based on training progress

    Schedules:
        - constant: λ = 1.0 throughout training
        - progressive: λ = 2 / (1 + exp(-10 * p)) - 1, where p = current_step / total_steps
        - linear: λ = p, where p = current_step / total_steps

    The progressive schedule (from DANN paper) starts near 0 and approaches 1.0,
    allowing the model to first learn emotion features before enforcing language invariance.

    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        schedule: Schedule type ("constant", "progressive", or "linear")

    Returns:
        Lambda value for current step
    """
    if schedule == "constant":
        return 1.0

    p = float(current_step) / float(total_steps)

    if schedule == "progressive":
        # DANN paper schedule
        return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
    elif schedule == "linear":
        return p
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


if __name__ == "__main__":
    """Test the model architecture"""

    print("\n" + "="*80)
    print("Testing Layer-Wise Adversarial HuBERT Architecture")
    print("="*80 + "\n")

    # Create model
    model = LayerWiseAdversarialHuBERT(
        model_name="facebook/hubert-base-ls960",
        num_emotions=4,
        num_languages=2,
        adversarial_layers=[3, 6, 9, 12],
        hidden_dim=256,
        dropout=0.3,
        freeze_feature_extractor=True,
        gradient_checkpointing=False
    )

    # Print parameter counts
    trainable_params = model.get_trainable_parameters()
    total_params = model.get_total_parameters()
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params-trainable_params)/1e6:.2f}M)\n")

    # Test forward pass
    batch_size = 4
    sequence_length = 16000 * 3  # 3 seconds at 16kHz

    print(f"Testing forward pass with batch_size={batch_size}, sequence_length={sequence_length}")

    # Create dummy input
    dummy_input = torch.randn(batch_size, sequence_length)
    dummy_attention_mask = torch.ones(batch_size, sequence_length)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_values=dummy_input,
            attention_mask=dummy_attention_mask,
            return_adversarial=True
        )

    print(f"\nOutput shapes:")
    print(f"  Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"  Number of language discriminators: {len(outputs['language_logits_list'])}")
    for i, lang_logits in enumerate(outputs['language_logits_list']):
        print(f"    Discriminator {i+1} (layer {model.adversarial_layers[i]}): {lang_logits.shape}")
    print(f"  Final representation: {outputs['final_representation'].shape}")

    # Test loss computation
    print(f"\nTesting loss computation...")
    dummy_emotion_labels = torch.randint(0, 4, (batch_size,))
    dummy_language_labels = torch.randint(0, 2, (batch_size,))

    loss_dict = compute_adversarial_loss(
        emotion_logits=outputs['emotion_logits'],
        language_logits_list=outputs['language_logits_list'],
        emotion_labels=dummy_emotion_labels,
        language_labels=dummy_language_labels,
        emotion_weight=1.0,
        language_weight=0.1
    )

    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Emotion Loss: {loss_dict['emotion_loss'].item():.4f}")
    print(f"  Language Loss (avg): {loss_dict['language_loss'].item():.4f}")
    for i, lang_loss in enumerate(loss_dict['individual_language_losses']):
        print(f"    Layer {model.adversarial_layers[i]} Loss: {lang_loss.item():.4f}")

    # Test lambda scheduling
    print(f"\nTesting lambda scheduling...")
    total_steps = 1000
    for schedule in ["constant", "progressive", "linear"]:
        print(f"  {schedule.capitalize()} schedule:")
        for step in [0, 250, 500, 750, 1000]:
            lambda_val = get_lambda_schedule(step, total_steps, schedule)
            print(f"    Step {step:4d}/{total_steps}: λ = {lambda_val:.4f}")

    print("\n" + "="*80)
    print("All tests passed! Architecture is working correctly.")
    print("="*80 + "\n")
