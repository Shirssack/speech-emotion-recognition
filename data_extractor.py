"""
data_extractor.py - Feature Extraction Pipeline
Author: Shirssack
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from collections import Counter

from utils import extract_feature, get_feature_vector_length


class AudioExtractor:
    """Main class for extracting and managing audio features."""
    
    def __init__(self, audio_config=None, classification=True, emotions=['sad', 'neutral', 'happy'],
                 balance=True, features_folder='features', base_paths=None, verbose=1):
        
        self.audio_config = audio_config or {
            'mfcc': True,
            'chroma': True,
            'mel': True,
            'contrast': False,
            'tonnetz': False
        }
        
        self.classification = classification
        self.emotions = emotions
        self.balance = balance
        self.features_folder = features_folder
        self.base_paths = [os.path.abspath(p) for p in (base_paths or [os.getcwd()])]
        self.verbose = verbose
        
        self.feature_length = get_feature_vector_length(**self.audio_config)
        os.makedirs(features_folder, exist_ok=True)
        
        self.train_features = None
        self.train_labels = None
        self.train_paths = None
        self.test_features = None
        self.test_labels = None
        self.test_paths = None
        
        self.emotion_to_int = {e: i for i, e in enumerate(emotions)}
        self.int_to_emotion = {i: e for e, i in self.emotion_to_int.items()}
    
    def extract_from_csv(self, train_csv, test_csv):
        """Extract features from audio files listed in CSV files."""
        
        if isinstance(train_csv, str):
            train_csv = [train_csv]
        if isinstance(test_csv, str):
            test_csv = [test_csv]
        
        cache_name = self._get_cache_name()
        train_cache = os.path.join(self.features_folder, f"train_{cache_name}.npz")
        test_cache = os.path.join(self.features_folder, f"test_{cache_name}.npz")
        
        if os.path.exists(train_cache) and os.path.exists(test_cache):
            if self.verbose:
                print(f"Loading cached features from {self.features_folder}/")
            self._load_cached_features(train_cache, test_cache)
        else:
            self._extract_features(train_csv, test_csv)
            
            if self.balance:
                self._balance_data()
            
            self._save_cached_features(train_cache, test_cache)
        
        return (self.train_features, self.train_labels, 
                self.test_features, self.test_labels)
    
    def _get_cache_name(self):
        feature_str = '-'.join([k for k, v in self.audio_config.items() if v])
        emotion_str = ''.join([e[0].upper() for e in sorted(self.emotions)])
        return f"{feature_str}_{emotion_str}"
    
    def _extract_features(self, train_csvs, test_csvs):
        train_data = []
        for csv_path in train_csvs:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                train_data.append(df)
        
        if train_data:
            train_df = pd.concat(train_data, ignore_index=True)
        else:
            raise FileNotFoundError(f"No training CSV files found: {train_csvs}")
        
        test_data = []
        for csv_path in test_csvs:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                test_data.append(df)
        
        if test_data:
            test_df = pd.concat(test_data, ignore_index=True)
        else:
            raise FileNotFoundError(f"No testing CSV files found: {test_csvs}")
        
        train_df = train_df[train_df['emotion'].isin(self.emotions)]
        test_df = test_df[test_df['emotion'].isin(self.emotions)]
        
        if self.verbose:
            print(f"\nExtracting features from {len(train_df)} training files...")
        
        self.train_features, self.train_labels, self.train_paths = self._extract_from_df(
            train_df, "Training"
        )
        
        if self.verbose:
            print(f"\nExtracting features from {len(test_df)} testing files...")
        
        self.test_features, self.test_labels, self.test_paths = self._extract_from_df(
            test_df, "Testing"
        )
    
    def _extract_from_df(self, df, partition_name):
        features = []
        labels = []
        paths = []
        errors = 0
        
        iterator = tqdm(df.iterrows(), total=len(df), desc=partition_name) if self.verbose else df.iterrows()
        
        for _, row in iterator:
            audio_path = row['path']
            emotion = row['emotion']

            resolved_path = self._resolve_audio_path(audio_path)

            try:
                feature_vector = extract_feature(resolved_path, **self.audio_config)

                if feature_vector is not None and len(feature_vector) == self.feature_length:
                    features.append(feature_vector)
                    labels.append(emotion)
                    paths.append(resolved_path)
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
        
        if self.verbose and errors > 0:
            print(f"  {errors} files failed to process")

        return np.array(features), np.array(labels), np.array(paths)

    def _resolve_audio_path(self, audio_path):
        expanded = os.path.expanduser(os.path.expandvars(str(audio_path)))

        if os.path.isabs(expanded) and os.path.exists(expanded):
            return expanded

        for base in self.base_paths:
            candidate = os.path.abspath(os.path.join(base, expanded))
            if os.path.exists(candidate):
                return candidate

        # fall back to the first base even if the file is missing so the caller sees
        # a consistent absolute path in any downstream errors
        if self.base_paths:
            return os.path.abspath(os.path.join(self.base_paths[0], expanded))

        return os.path.abspath(expanded)
    
    def _balance_data(self):
        if self.verbose:
            print("\nBalancing dataset...")
        
        self.train_features, self.train_labels, self.train_paths = self._undersample(
            self.train_features, self.train_labels, self.train_paths
        )
        
        self.test_features, self.test_labels, self.test_paths = self._undersample(
            self.test_features, self.test_labels, self.test_paths
        )
    
    def _undersample(self, features, labels, paths):
        counter = Counter(labels)
        min_count = min(counter.values())
        
        new_features = []
        new_labels = []
        new_paths = []
        
        for emotion in self.emotions:
            indices = np.where(labels == emotion)[0]
            np.random.seed(42)
            selected = np.random.choice(indices, size=min_count, replace=False)
            
            new_features.extend(features[selected])
            new_labels.extend(labels[selected])
            new_paths.extend(paths[selected])
        
        new_features, new_labels, new_paths = shuffle(
            new_features, new_labels, new_paths, random_state=42
        )
        
        return np.array(new_features), np.array(new_labels), np.array(new_paths)
    
    def _save_cached_features(self, train_path, test_path):
        np.savez(train_path,
                 features=self.train_features,
                 labels=self.train_labels,
                 paths=self.train_paths)
        
        np.savez(test_path,
                 features=self.test_features,
                 labels=self.test_labels,
                 paths=self.test_paths)
        
        if self.verbose:
            print(f"\nCached features saved to {self.features_folder}/")
    
    def _load_cached_features(self, train_path, test_path):
        train_data = np.load(train_path, allow_pickle=True)
        self.train_features = train_data['features']
        self.train_labels = train_data['labels']
        self.train_paths = train_data['paths']
        
        test_data = np.load(test_path, allow_pickle=True)
        self.test_features = test_data['features']
        self.test_labels = test_data['labels']
        self.test_paths = test_data['paths']
        
        if self.verbose:
            print(f"Loaded {len(self.train_features)} training, {len(self.test_features)} testing samples")


def load_data(train_csv, test_csv, audio_config=None, classification=True,
              emotions=['sad', 'neutral', 'happy'], balance=True, verbose=1):
    """Convenience function to load data."""
    
    extractor = AudioExtractor(
        audio_config=audio_config,
        classification=classification,
        emotions=emotions,
        balance=balance,
        verbose=verbose
    )
    
    X_train, y_train, X_test, y_test = extractor.extract_from_csv(train_csv, test_csv)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Data extractor module loaded successfully.")
    print(f"Default feature vector length: {get_feature_vector_length()}")
