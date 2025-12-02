"""
run_test.py - Quick test to verify setup
"""


def main():
    from emotion_recognition import EmotionRecognizer

    print("=" * 60)
    print("  RAVDESS + TESS Setup Test")
    print("=" * 60)

    # Initialize with 3 emotions (fastest to test)
    print("\n1. Initializing EmotionRecognizer...")
    rec = EmotionRecognizer(
        emotions=['happy', 'sad', 'neutral'],
        use_ravdess=True,
        use_tess=True,
        use_emodb=False,
        use_hindi=False,
        use_custom=False,
        balance=True,
        verbose=1
    )

    # Train
    print("\n2. Training model...")
    rec.train()

    # Results
    print("\n3. Results:")
    print(f"   Training Accuracy: {rec.train_score():.2%}")
    print(f"   Testing Accuracy:  {rec.test_score():.2%}")

    # Confusion Matrix
    print("\n4. Confusion Matrix:")
    print(rec.confusion_matrix())

    # Sample Distribution
    print("\n5. Sample Distribution:")
    print(rec.get_samples_by_class())

    # Save model
    print("\n6. Saving model...")
    rec.save_model('models/ravdess_tess_model.pkl')

    print("\n" + "=" * 60)
    print("  Setup Complete! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
