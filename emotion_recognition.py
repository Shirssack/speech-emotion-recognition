"""
emotion_recognition.py - Main Speech Emotion Recognition Class
Author: Shirssack
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

from utils import extract_feature, get_feature_vector_length
from data_extractor import AudioExtractor
from create_csv import (write_ravdess_csv, write_tess_csv, write_emodb_csv, 
                        write_custom_csv, write_hindi_csv)


class EmotionRecognizer:
    """Main class for Speech Emotion Recognition."""
    
    def __init__(self, model=None, emotions=['sad', 'neutral', 'happy'],
                 features=['mfcc', 'chroma', 'mel'], balance=True,
                 data_path='data', use_ravdess=True, use_tess=True,
                 use_emodb=False, use_custom=False, use_hindi=False,
                 verbose=1):
        
        self.emotions = emotions
        self.features = features
        self.balance = balance

        # Resolve data_path robustly so users can point to datasets regardless of
        # their working directory.
        expanded_path = os.path.expanduser(os.path.expandvars(data_path))
        base_path = os.path.dirname(os.path.abspath(__file__))

        if os.path.isabs(expanded_path):
            resolved_path = expanded_path
        else:
            project_default = os.path.join(base_path, expanded_path)
            cwd_candidate = os.path.abspath(expanded_path)

            if os.path.exists(project_default):
                resolved_path = project_default
            elif os.path.exists(cwd_candidate):
                resolved_path = cwd_candidate
            else:
                # Fall back to the project-relative location even if it does not
                # exist yet so downstream messaging reflects that path.
                resolved_path = project_default

        self.data_path = resolved_path
        self.verbose = verbose
        
        self.use_ravdess = use_ravdess
        self.use_tess = use_tess
        self.use_emodb = use_emodb
        self.use_custom = use_custom
        self.use_hindi = use_hindi
        
        self.audio_config = {
            'mfcc': 'mfcc' in features,
            'chroma': 'chroma' in features,
            'mel': 'mel' in features,
            'contrast': 'contrast' in features,
            'tonnetz': 'tonnetz' in features
        }

        self._base_paths = []
        for candidate in (os.getcwd(), base_path, self.data_path):
            abs_candidate = os.path.abspath(candidate)
            if abs_candidate not in self._base_paths:
                self._base_paths.append(abs_candidate)
        
        if model is None:
            self.model = MLPClassifier(
                hidden_layer_sizes=(300,),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=500,
                random_state=42
            )
        else:
            self.model = model
        
        self.scaler = StandardScaler()
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.emotion_to_int = {e: i for i, e in enumerate(emotions)}
        self.int_to_emotion = {i: e for e, i in self.emotion_to_int.items()}
        
        self._is_trained = False
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare CSV files and extract features."""
        
        train_csvs = []
        test_csvs = []
        
        csv_folder = os.path.join(self.data_path, 'csv')
        os.makedirs(csv_folder, exist_ok=True)
        
        def add_dataset(dataset_path, train_name, test_name, writer_fn):
            train_csv = os.path.join(csv_folder, train_name)
            test_csv = os.path.join(csv_folder, test_name)

            if os.path.exists(train_csv) and os.path.exists(test_csv):
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
                return True

            if os.path.exists(dataset_path):
                if not os.path.exists(train_csv):
                    writer_fn(dataset_path, self.emotions,
                              train_csv, test_csv, verbose=self.verbose)

                if os.path.exists(train_csv):
                    train_csvs.append(train_csv)
                if os.path.exists(test_csv):
                    test_csvs.append(test_csv)
                return True

            return False

        if self.use_ravdess:
            ravdess_path = os.path.join(self.data_path, 'ravdess')
            add_dataset(ravdess_path, 'train_ravdess.csv', 'test_ravdess.csv', write_ravdess_csv)
        
        if self.use_tess:
            tess_path = os.path.join(self.data_path, 'tess')
            add_dataset(tess_path, 'train_tess.csv', 'test_tess.csv', write_tess_csv)
        
        if self.use_emodb:
            emodb_path = os.path.join(self.data_path, 'emodb')
            add_dataset(emodb_path, 'train_emodb.csv', 'test_emodb.csv', write_emodb_csv)
        
        if self.use_custom:
            custom_path = os.path.join(self.data_path, 'custom')
            add_dataset(custom_path, 'train_custom.csv', 'test_custom.csv', write_custom_csv)
        
        if self.use_hindi:
            hindi_path = os.path.join(self.data_path, 'hindi')
            add_dataset(hindi_path, 'train_hindi.csv', 'test_hindi.csv', write_hindi_csv)
        
        train_csvs = [c for c in train_csvs if os.path.exists(c)]
        test_csvs = [c for c in test_csvs if os.path.exists(c)]

        enabled_sources = {
            'ravdess': self.use_ravdess,
            'tess': self.use_tess,
            'emodb': self.use_emodb,
            'custom': self.use_custom,
            'hindi': self.use_hindi,
        }

        if not any(enabled_sources.values()):
            message = (
                "No datasets are enabled.\n"
                "Turn on at least one source (e.g., use_ravdess=True or use_custom=True).\n"
                f"Current data_path: {os.path.abspath(self.data_path)}\n"
                "If your data is elsewhere, pass data_path='/absolute/path/to/data' when creating the recognizer."
            )

            if self.verbose:
                print(message)

            raise ValueError(message)

        if not train_csvs:
            resolved_data_path = os.path.abspath(self.data_path)
            expected_sources = []

            if self.use_ravdess:
                expected_sources.append(os.path.join(resolved_data_path, 'ravdess'))
            if self.use_tess:
                expected_sources.append(os.path.join(resolved_data_path, 'tess'))
            if self.use_emodb:
                expected_sources.append(os.path.join(resolved_data_path, 'emodb'))
            if self.use_custom:
                expected_sources.append(os.path.join(resolved_data_path, 'custom'))
            if self.use_hindi:
                expected_sources.append(os.path.join(resolved_data_path, 'hindi'))

            expected_str = '\n  - '.join(expected_sources)
            csv_hint = os.path.join(resolved_data_path, 'csv')
            message = (
                "No dataset found for the enabled sources.\n"
                f"Checked data_path: {resolved_data_path}\n"
                f"Expected folders or CSVs in: {csv_hint}\n"
                f"Expected folders:\n  - {expected_str}\n\n"
                "Fixes:\n"
                "  1) Download the datasets into the folders above (see README > Datasets).\n"
                "  2) Place train/test CSVs in the csv/ folder if you already generated them.\n"
                "  3) Pass data_path='/absolute/path/to/data' when creating the recognizer.\n"
                "  4) Disable sources you do not have (e.g., use_ravdess=False, use_tess=False)."
            )

            if self.verbose:
                print(message)

            raise FileNotFoundError(message)
        
        extractor = AudioExtractor(
            audio_config=self.audio_config,
            emotions=self.emotions,
            balance=self.balance,
            base_paths=self._base_paths,
            verbose=self.verbose
        )
        
        self.X_train, self.y_train, self.X_test, self.y_test = extractor.extract_from_csv(
            train_csvs, test_csvs
        )
        
        if self.X_train is not None and len(self.X_train) > 0:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            if self.verbose:
                print(f"\nData prepared: {len(self.X_train)} train, {len(self.X_test)} test")
    
    def train(self):
        """Train the model."""
        
        if self.X_train is None or len(self.X_train) == 0:
            raise ValueError("No training data available.")
        
        if self.verbose:
            print(f"\nTraining {self.model.__class__.__name__}...")
        
        self.model.fit(self.X_train, self.y_train)
        self._is_trained = True
        
        if self.verbose:
            print(f"Train accuracy: {self.train_score():.2%}")
            print(f"Test accuracy: {self.test_score():.2%}")
        
        return self
    
    def train_score(self):
        return accuracy_score(self.y_train, self.model.predict(self.X_train))
    
    def test_score(self):
        return accuracy_score(self.y_test, self.model.predict(self.X_test))
    
    def predict(self, audio_path):
        """Predict emotion for a single audio file."""
        
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = extract_feature(audio_path, **self.audio_config)
        features_scaled = self.scaler.transform([features])
        
        return self.model.predict(features_scaled)[0]
    
    def predict_proba(self, audio_path):
        """Predict emotion probabilities."""
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
        features = extract_feature(audio_path, **self.audio_config)
        features_scaled = self.scaler.transform([features])
        
        probas = self.model.predict_proba(features_scaled)[0]
        
        return {self.model.classes_[i]: float(p) for i, p in enumerate(probas)}
    
    def confusion_matrix(self, percentage=True, as_dataframe=True):
        """Get confusion matrix."""
        
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred, labels=self.emotions)
        
        if percentage:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        if as_dataframe:
            cm = pd.DataFrame(
                cm,
                index=[f"true_{e}" for e in self.emotions],
                columns=[f"pred_{e}" for e in self.emotions]
            )
        
        return cm
    
    def classification_report(self):
        y_pred = self.model.predict(self.X_test)
        return classification_report(self.y_test, y_pred, target_names=self.emotions)
    
    def get_samples_by_class(self):
        train_counts = Counter(self.y_train) if self.y_train is not None else {}
        test_counts = Counter(self.y_test) if self.y_test is not None else {}
        
        data = []
        for emotion in self.emotions:
            data.append({
                'emotion': emotion,
                'train': train_counts.get(emotion, 0),
                'test': test_counts.get(emotion, 0)
            })
        
        return pd.DataFrame(data)
    
    def save_model(self, path='models/emotion_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'emotions': self.emotions,
                'audio_config': self.audio_config
            }, f)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load_model(self, path='models/emotion_model.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.emotions = data['emotions']
        self.audio_config = data['audio_config']
        self._is_trained = True
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def determine_best_model(self, models=None):
        """Try multiple models and select the best."""
        
        if models is None:
            models = [
                SVC(kernel='rbf', C=1.0, random_state=42),
                MLPClassifier(hidden_layer_sizes=(300,), max_iter=500, random_state=42),
                RandomForestClassifier(n_estimators=100, random_state=42),
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                KNeighborsClassifier(n_neighbors=5),
                BaggingClassifier(n_estimators=50, random_state=42)
            ]
        
        best_model = None
        best_score = 0
        results = []
        
        for model in models:
            model_name = model.__class__.__name__
            
            if self.verbose:
                print(f"Testing {model_name}...", end=" ")
            
            model.fit(self.X_train, self.y_train)
            score = accuracy_score(self.y_test, model.predict(self.X_test))
            results.append({'model': model_name, 'accuracy': score})
            
            if self.verbose:
                print(f"{score:.2%}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        self._is_trained = True
        
        if self.verbose:
            print(f"\nBest: {best_model.__class__.__name__} ({best_score:.2%})")
        
        return pd.DataFrame(results).sort_values('accuracy', ascending=False)


def load_best_model(emotions=['sad', 'neutral', 'happy'], grid_folder='grid'):
    """Load the best pre-trained model from grid search."""
    
    emotion_str = '_'.join([e[0].upper() for e in sorted(emotions)])
    estimators_path = os.path.join(grid_folder, f'best_estimators_{emotion_str}.pkl')
    
    if not os.path.exists(estimators_path):
        raise FileNotFoundError(f"No estimators for {emotions}. Run grid_search.py first.")
    
    with open(estimators_path, 'rb') as f:
        data = pickle.load(f)
    
    rec = EmotionRecognizer(emotions=emotions, verbose=0)
    
    estimators = data['estimators']
    if 'MLPClassifier' in estimators:
        rec.model = estimators['MLPClassifier']
    elif 'RandomForestClassifier' in estimators:
        rec.model = estimators['RandomForestClassifier']
    else:
        rec.model = list(estimators.values())[0]
    
    rec.scaler = data['scaler']
    rec.emotions = data['emotions']
    rec.audio_config = data['audio_config']
    rec._is_trained = True
    
    return rec, estimators


if __name__ == "__main__":
    print("EmotionRecognizer module loaded.")
    print("\nUsage:")
    print("  rec = EmotionRecognizer(emotions=['happy', 'sad', 'neutral'])")
    print("  rec.train()")
    print("  print(rec.predict('audio.wav'))")
