"""
generate_4class_csv.py - Generate CSV files for 4-emotion classification
Emotions: sad, neutral, happy, angry
"""

import os
import random

# Set random seed for reproducibility
random.seed(42)

def parse_ravdess(data_path, emotions):
    """Parse RAVDESS dataset for 4 emotions."""
    data = []

    # RAVDESS emotion mapping: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    ravdess_map = {
        '01': 'neutral',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry'
    }

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                parts = file.replace('.wav', '').split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in ravdess_map:
                        emotion = ravdess_map[emotion_code]
                        if emotion in emotions:
                            file_path = os.path.join(root, file)
                            data.append((file_path, emotion))

    return data

def parse_tess(data_path, emotions):
    """Parse TESS dataset for 4 emotions."""
    data = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                parts = file.replace('.wav', '').split('_')
                if len(parts) >= 3:
                    emotion_raw = parts[-1].lower()
                    # TESS uses 'ps' for surprised
                    if emotion_raw == 'ps':
                        emotion = 'surprised'
                    else:
                        emotion = emotion_raw

                    if emotion in emotions:
                        file_path = os.path.join(root, file)
                        data.append((file_path, emotion))

    return data

def parse_hindi(data_path, emotions):
    """Parse Hindi dataset for 4 emotions."""
    data = []

    hindi_emotion_map = {
        'anger': 'angry',
        'angry': 'angry',
        'happy': 'happy',
        'happiness': 'happy',
        'sad': 'sad',
        'sadness': 'sad',
        'neutral': 'neutral',
        'normal': 'neutral',
    }

    # Check emotion folders
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            folder_emotion = hindi_emotion_map.get(item.lower())
            if folder_emotion and folder_emotion in emotions:
                for file in os.listdir(item_path):
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        file_path = os.path.join(item_path, file)
                        data.append((file_path, folder_emotion))

    # Parse filenames
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                name = file.rsplit('.', 1)[0].lower()
                for key, emotion in hindi_emotion_map.items():
                    if key in name and emotion in emotions:
                        file_path = os.path.join(root, file)
                        if (file_path, emotion) not in data:
                            data.append((file_path, emotion))
                        break

    return data

def write_csv(data, train_file, test_file, train_size=0.8):
    """Write train/test CSV files."""
    # Shuffle data
    random.shuffle(data)

    # Split by emotion to maintain balance
    emotion_data = {}
    for path, emotion in data:
        if emotion not in emotion_data:
            emotion_data[emotion] = []
        emotion_data[emotion].append((path, emotion))

    train_data = []
    test_data = []

    for emotion, samples in emotion_data.items():
        split_idx = int(len(samples) * train_size)
        train_data.extend(samples[:split_idx])
        test_data.extend(samples[split_idx:])

    # Shuffle again
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Write train CSV
    with open(train_file, 'w') as f:
        f.write('path,emotion\n')
        for path, emotion in train_data:
            f.write(f'{path},{emotion}\n')

    # Write test CSV
    with open(test_file, 'w') as f:
        f.write('path,emotion\n')
        for path, emotion in test_data:
            f.write(f'{path},{emotion}\n')

    return len(train_data), len(test_data)

def main():
    emotions = ['sad', 'neutral', 'happy', 'angry']

    print("="*70)
    print("GENERATING 4-EMOTION CSV FILES")
    print("="*70)
    print(f"\nEmotions: {', '.join(emotions)}")
    print(f"Train/Test split: 80/20")

    # RAVDESS
    print("\n[1/3] Processing RAVDESS dataset...")
    ravdess_data = parse_ravdess('data/ravdess', emotions)
    if ravdess_data:
        train_count, test_count = write_csv(
            ravdess_data,
            'data/csv/train_ravdess_4class.csv',
            'data/csv/test_ravdess_4class.csv'
        )
        print(f"  ✓ Found {len(ravdess_data)} files")
        print(f"  ✓ Train: {train_count}, Test: {test_count}")

        # Count by emotion
        emotion_counts = {}
        for _, emotion in ravdess_data:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        for emotion, count in sorted(emotion_counts.items()):
            print(f"    - {emotion}: {count}")
    else:
        print("  ⚠ No files found")

    # TESS
    print("\n[2/3] Processing TESS dataset...")
    tess_data = parse_tess('data/tess', emotions)
    if tess_data:
        train_count, test_count = write_csv(
            tess_data,
            'data/csv/train_tess_4class.csv',
            'data/csv/test_tess_4class.csv'
        )
        print(f"  ✓ Found {len(tess_data)} files")
        print(f"  ✓ Train: {train_count}, Test: {test_count}")

        # Count by emotion
        emotion_counts = {}
        for _, emotion in tess_data:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        for emotion, count in sorted(emotion_counts.items()):
            print(f"    - {emotion}: {count}")
    else:
        print("  ⚠ No files found")

    # Hindi
    print("\n[3/3] Processing Hindi dataset...")
    if os.path.exists('data/hindi'):
        hindi_data = parse_hindi('data/hindi', emotions)
        if hindi_data:
            train_count, test_count = write_csv(
                hindi_data,
                'data/csv/train_hindi_4class.csv',
                'data/csv/test_hindi_4class.csv'
            )
            print(f"  ✓ Found {len(hindi_data)} files")
            print(f"  ✓ Train: {train_count}, Test: {test_count}")

            # Count by emotion
            emotion_counts = {}
            for _, emotion in hindi_data:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            for emotion, count in sorted(emotion_counts.items()):
                print(f"    - {emotion}: {count}")
        else:
            print("  ⚠ No files found")
    else:
        print("  ⚠ Hindi dataset directory not found")

    print("\n" + "="*70)
    print("✓ CSV GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/csv/train_ravdess_4class.csv")
    print("  - data/csv/test_ravdess_4class.csv")
    print("  - data/csv/train_tess_4class.csv")
    print("  - data/csv/test_tess_4class.csv")
    print("  - data/csv/train_hindi_4class.csv")
    print("  - data/csv/test_hindi_4class.csv")
    print("\nTo train with 4 emotions:")
    print("  python train_transformer.py --emotions sad neutral happy angry --train_csv data/csv/train_ravdess_4class.csv data/csv/train_tess_4class.csv --test_csv data/csv/test_ravdess_4class.csv data/csv/test_tess_4class.csv")
    print("="*70)

if __name__ == '__main__':
    main()
