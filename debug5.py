print("Testing emotion_recognition.py...")

try:
    from emotion_recognition import EmotionRecognizer
    print("✓ EmotionRecognizer imported successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
