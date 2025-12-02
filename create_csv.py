"""
create_csv.py - Dataset CSV Generator
Author: Shirssack

This module parses different emotion speech datasets and creates
standardized CSV files mapping audio paths to emotion labels.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import RAVDESS_EMOTIONS, EMODB_EMOTIONS


def write_ravdess_csv(data_path, emotions=['sad', 'neutral', 'happy'], 
                      train_name='train_ravdess.csv', test_name='test_ravdess.csv',
                      train_size=0.8, verbose=1):
    """Parse RAVDESS dataset and create train/test CSVs."""
    
    data = []
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                parts = file.replace('.wav', '').split('-')
                
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    
                    if emotion_code in RAVDESS_EMOTIONS:
                        emotion = RAVDESS_EMOTIONS[emotion_code]
                        
                        if emotion in emotions:
                            file_path = os.path.join(root, file)
                            data.append((file_path, emotion))
    
    if verbose:
        print(f"[RAVDESS] Found {len(data)} audio files")
        for emotion in emotions:
            count = sum(1 for _, e in data if e == emotion)
            print(f"  - {emotion}: {count} files")
    
    if len(data) > 0:
        paths, labels = zip(*data)
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, train_size=train_size, random_state=42, stratify=labels
        )
        
        train_df = pd.DataFrame({'path': train_paths, 'emotion': train_labels})
        test_df = pd.DataFrame({'path': test_paths, 'emotion': test_labels})
        
        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)
        
        if verbose:
            print(f"[RAVDESS] Saved {len(train_df)} training, {len(test_df)} testing")
    
    return len(data)


def write_tess_csv(data_path, emotions=['sad', 'neutral', 'happy'],
                   train_name='train_tess.csv', test_name='test_tess.csv',
                   train_size=0.8, verbose=1):
    """Parse TESS dataset."""
    
    data = []
    
    tess_emotion_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'ps': 'surprised',
        'sad': 'sad'
    }
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                parts = file.replace('.wav', '').split('_')
                
                if len(parts) >= 3:
                    emotion_raw = parts[-1].lower()
                    emotion = tess_emotion_map.get(emotion_raw, emotion_raw)
                    
                    if emotion in emotions:
                        file_path = os.path.join(root, file)
                        data.append((file_path, emotion))
    
    if verbose:
        print(f"[TESS] Found {len(data)} audio files")
    
    if len(data) > 0:
        paths, labels = zip(*data)
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, train_size=train_size, random_state=42, stratify=labels
        )
        
        train_df = pd.DataFrame({'path': train_paths, 'emotion': train_labels})
        test_df = pd.DataFrame({'path': test_paths, 'emotion': test_labels})
        
        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)
    
    return len(data)


def write_emodb_csv(data_path, emotions=['sad', 'neutral', 'happy'],
                    train_name='train_emodb.csv', test_name='test_emodb.csv',
                    train_size=0.8, verbose=1):
    """Parse EMO-DB dataset."""
    
    data = []
    
    emodb_to_standard = {
        'W': 'angry',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happy',
        'T': 'sad',
        'N': 'neutral'
    }
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                if len(file) > 5:
                    emotion_code = file[5].upper()
                    emotion = emodb_to_standard.get(emotion_code)
                    
                    if emotion and emotion in emotions:
                        file_path = os.path.join(root, file)
                        data.append((file_path, emotion))
    
    if verbose:
        print(f"[EMO-DB] Found {len(data)} audio files")
    
    if len(data) > 0:
        paths, labels = zip(*data)
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, train_size=train_size, random_state=42, stratify=labels
        )
        
        train_df = pd.DataFrame({'path': train_paths, 'emotion': train_labels})
        test_df = pd.DataFrame({'path': test_paths, 'emotion': test_labels})
        
        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)
    
    return len(data)


def write_custom_csv(data_path, emotions=['sad', 'neutral', 'happy'],
                     train_name='train_custom.csv', test_name='test_custom.csv',
                     train_size=0.8, verbose=1):
    """Parse custom dataset (filename_emotion.wav format)."""
    
    data = []
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                name_without_ext = file.replace('.wav', '')
                parts = name_without_ext.split('_')
                
                if len(parts) >= 2:
                    emotion = parts[-1].lower()
                    
                    if emotion in emotions:
                        file_path = os.path.join(root, file)
                        data.append((file_path, emotion))
    
    if verbose:
        print(f"[Custom] Found {len(data)} audio files")
    
    if len(data) > 0:
        paths, labels = zip(*data)
        
        if len(set(labels)) > 1:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                paths, labels, train_size=train_size, random_state=42, stratify=labels
            )
        else:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                paths, labels, train_size=train_size, random_state=42
            )
        
        train_df = pd.DataFrame({'path': train_paths, 'emotion': train_labels})
        test_df = pd.DataFrame({'path': test_paths, 'emotion': test_labels})
        
        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)
    
    return len(data)


def write_hindi_csv(data_path, emotions=['angry', 'happy', 'neutral', 'sad'],
                    train_name='train_hindi.csv', test_name='test_hindi.csv',
                    train_size=0.8, verbose=1):
    """Parse Hindi emotion speech datasets."""
    
    data = []
    
    hindi_emotion_map = {
        'angry': 'angry', 'anger': 'angry',
        'happy': 'happy', 'happiness': 'happy', 'joy': 'happy',
        'sad': 'sad', 'sadness': 'sad', 'sorrow': 'sad',
        'neutral': 'neutral', 'normal': 'neutral',
        'fear': 'fear', 'fearful': 'fear',
        'disgust': 'disgust',
        'surprise': 'surprised', 'surprised': 'surprised',
        'sarcastic': 'sarcastic',
        'gussa': 'angry',
        'khushi': 'happy',
        'dukh': 'sad',
        'dar': 'fear',
        'samanya': 'neutral',
    }
    
    # Method 1: Check emotion folders
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        
        if os.path.isdir(item_path):
            folder_emotion = hindi_emotion_map.get(item.lower())
            
            if folder_emotion and folder_emotion in emotions:
                for file in os.listdir(item_path):
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        file_path = os.path.join(item_path, file)
                        data.append((file_path, folder_emotion))
    
    # Method 2: Parse filenames
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                name = file.rsplit('.', 1)[0].lower()
                
                found_emotion = None
                
                if '_' in name:
                    last_part = name.split('_')[-1]
                    found_emotion = hindi_emotion_map.get(last_part)
                
                if not found_emotion:
                    for key, emotion in hindi_emotion_map.items():
                        if key in name:
                            found_emotion = emotion
                            break
                
                if found_emotion and found_emotion in emotions:
                    file_path = os.path.join(root, file)
                    if (file_path, found_emotion) not in data:
                        data.append((file_path, found_emotion))
    
    if verbose:
        print(f"[Hindi] Found {len(data)} audio files")
        for emotion in emotions:
            count = sum(1 for _, e in data if e == emotion)
            print(f"  - {emotion}: {count} files")
    
    if len(data) > 0:
        paths, labels = zip(*data)
        
        if len(set(labels)) > 1 and min([labels.count(e) for e in set(labels)]) >= 2:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                paths, labels, train_size=train_size, random_state=42, stratify=labels
            )
        else:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                paths, labels, train_size=train_size, random_state=42
            )
        
        train_df = pd.DataFrame({'path': train_paths, 'emotion': train_labels})
        test_df = pd.DataFrame({'path': test_paths, 'emotion': test_labels})
        
        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)
    
    return len(data)


if __name__ == "__main__":
    print("CSV generator module loaded successfully.")
