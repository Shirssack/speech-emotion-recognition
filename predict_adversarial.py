"""
Simple prediction script for adversarial HuBERT emotion recognition

Usage:
    python predict_adversarial.py \
        --model_path models/adversarial_hubert/best_model.pth \
        --audio_path path/to/audio.wav

Author: Research Implementation
"""
import torch
import soundfile as sf
import librosa
import argparse
import os
import numpy as np
from transformers import Wav2Vec2Processor
from adversarial_hubert_emotion import LayerWiseAdversarialHuBERT


def load_model(model_path, device='cuda'):
    """
    Load trained adversarial HuBERT model

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model
        id_to_emotion: Mapping from emotion ID to emotion name
        config: Model configuration dictionary
    """
    print(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract configuration
    config = checkpoint['config']
    id_to_emotion = checkpoint['id_to_emotion']

    print(f"  Model: {config['model_name']}")
    print(f"  Emotions: {list(id_to_emotion.values())}")
    print(f"  Adversarial Layers: {config['adversarial_layers']}")

    # Create model
    model = LayerWiseAdversarialHuBERT(
        model_name=config['model_name'],
        num_emotions=len(id_to_emotion),
        num_languages=2,
        adversarial_layers=config['adversarial_layers'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        freeze_feature_extractor=config.get('freeze_feature_extractor', True),
        gradient_checkpointing=False  # Disable for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully!\n")
    return model, id_to_emotion, config


def predict_emotion(
    audio_path,
    model,
    processor,
    id_to_emotion,
    device='cuda',
    max_duration=5.0,
    sampling_rate=16000
):
    """
    Predict emotion for a single audio file

    Args:
        audio_path: Path to audio file
        model: Loaded model
        processor: HuBERT processor
        id_to_emotion: Mapping from ID to emotion name
        device: Device ('cuda' or 'cpu')
        max_duration: Maximum audio duration in seconds
        sampling_rate: Target sampling rate

    Returns:
        emotion: Predicted emotion name
        confidence: Confidence score (0-1)
        all_probs: Dictionary of all emotion probabilities
    """
    # Load audio
    speech, sr = sf.read(audio_path)

    # Convert stereo to mono if needed
    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=1)

    # Resample if needed
    if sr != sampling_rate:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=sampling_rate)

    # Truncate or pad to max_duration
    max_length = int(max_duration * sampling_rate)
    if len(speech) > max_length:
        speech = speech[:max_length]
    else:
        # Pad with zeros
        speech = np.pad(speech, (0, max_length - len(speech)), mode='constant')

    # Process with HuBERT processor
    inputs = processor(
        speech,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            return_adversarial=False  # Don't need adversarial outputs for inference
        )

        logits = outputs['emotion_logits']
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # Get emotion and confidence
    emotion = id_to_emotion[pred]
    confidence = probs[0][pred].item()

    # Get all probabilities
    all_probs = {id_to_emotion[i]: probs[0][i].item() for i in range(len(id_to_emotion))}

    return emotion, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(
        description='Predict emotion from audio file using adversarial HuBERT'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to audio file (.wav, .mp3, etc.)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model, id_to_emotion, config = load_model(args.model_path, device)

    # Load processor
    print("Loading HuBERT processor...")
    processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
    print("✓ Processor loaded!\n")

    # Predict
    print(f"{'='*70}")
    print(f"Predicting emotion for: {args.audio_path}")
    print(f"{'='*70}\n")

    emotion, confidence, all_probs = predict_emotion(
        args.audio_path,
        model,
        processor,
        id_to_emotion,
        device,
        args.max_duration
    )

    # Display results
    print(f"{'='*70}")
    print(f"PREDICTED EMOTION: {emotion.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'='*70}")
    print(f"\nDetailed Probabilities:")
    print(f"{'-'*70}")

    # Sort by probability
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    for emo, prob in sorted_probs:
        bar_length = int(prob * 50)
        bar = '█' * bar_length
        indicator = '◄' if emo == emotion else ' '
        print(f"  {emo:10s}: {prob:6.2%}  {bar:50s} {indicator}")

    print(f"{'-'*70}\n")


if __name__ == "__main__":
    main()
