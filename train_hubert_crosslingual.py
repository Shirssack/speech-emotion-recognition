"""
train_hubert_crosslingual.py - Train HuBERT with Layer-Wise Adversarial Disentanglement

Training script for cross-lingual speech emotion recognition
Implements adversarial language disentanglement at multiple layers
"""

import argparse
from hubert_crosslingual_emotion import CrossLingualEmotionRecognizer

def main():
    parser = argparse.ArgumentParser(description='Train HuBERT Cross-Lingual Emotion Recognition')

    # Data arguments
    parser.add_argument('--train_csv', nargs='+',
                        default=['data/csv/train_ravdess_4class.csv',
                                'data/csv/train_tess_4class.csv',
                                'data/csv/train_hindi_4class.csv'],
                        help='Training CSV files')
    parser.add_argument('--val_csv', nargs='+',
                        default=['data/csv/test_ravdess_4class.csv',
                                'data/csv/test_tess_4class.csv',
                                'data/csv/test_hindi_4class.csv'],
                        help='Validation CSV files')
    parser.add_argument('--emotions', nargs='+',
                        default=['sad', 'neutral', 'happy', 'angry'],
                        help='Emotions to recognize')

    # Model arguments
    parser.add_argument('--adversarial_layers', nargs='+', type=int,
                        default=[6, 9, 12],
                        help='HuBERT layers to apply adversarial training (1-12)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='Weight for adversarial loss')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--alpha_schedule', choices=['constant', 'progressive'],
                        default='progressive',
                        help='Alpha scheduling for gradient reversal')

    # Save arguments
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("HUBERT CROSS-LINGUAL EMOTION RECOGNITION TRAINING")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Emotions: {args.emotions}")
    print(f"  Adversarial Layers: {args.adversarial_layers}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Lambda (Adversarial): {args.lambda_adv}")
    print(f"  Max Duration: {args.max_duration}s")
    print(f"  Alpha Schedule: {args.alpha_schedule}")
    print(f"  Training CSVs: {args.train_csv}")
    print(f"  Validation CSVs: {args.val_csv}")
    print("="*70)

    # Create recognizer
    recognizer = CrossLingualEmotionRecognizer(
        emotions=args.emotions,
        adversarial_layers=args.adversarial_layers
    )

    # Train
    recognizer.train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_duration=args.max_duration,
        lambda_adv=args.lambda_adv,
        alpha_schedule=args.alpha_schedule,
        save_dir=args.save_dir
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved in: {args.save_dir}/")
    print(f"  - hubert_crosslingual_best.pt (best validation)")
    print(f"  - hubert_crosslingual_final.pt (final epoch)")
    print("\nFor cross-lingual evaluation, run:")
    print(f"  python evaluate_crosslingual.py")
    print("="*70)

if __name__ == "__main__":
    main()
