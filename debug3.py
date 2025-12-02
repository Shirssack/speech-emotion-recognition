import os

print("Checking files...\n")

files = [
    'utils.py',
    'create_csv.py',
    'data_extractor.py',
    'parameters.py',
    'feature_manager.py',
    'grid_search.py',
    'emotion_recognition.py',
    'deep_emotion_recognition.py',
    'test.py'
]

for f in files:
    if os.path.exists(f):
        print(f"✓ {f}")
    else:
        print(f"✗ {f} - MISSING!")

print("\n--- Data folders ---")

data_folders = [
    'data',
    'data/ravdess',
    'data/tess'
]

for folder in data_folders:
    if os.path.exists(folder):
        # Count files
        count = 0
        for root, dirs, files in os.walk(folder):
            count += len([f for f in files if f.endswith('.wav')])
        print(f"✓ {folder}/ ({count} .wav files)")
    else:
        print(f"✗ {folder}/ - MISSING!")
