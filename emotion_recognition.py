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
        self.data_path = data_path
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
        
        if self.use_ravdess:
            ravdess_path = os.path.join(self.data_path, 'ravdess')
            if os.path.exists(ravdess_path):
                train_csv = os.path.join(csv_folder, 'train_ravdess.csv')
                test_csv = os.path.join(csv_folder, 'test_ravdess.csv')
                
                if not os.path.exists(train_csv):
                    write_ravdess_csv(ravdess_path, self.emotions, 
                                     train_csv, test_csv, verbose=self.verbose)
                
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
        
        if self.use_tess:
            tess_path = os.path.join(self.data_path, 'tess')
            if os.path.exists(tess_path):
                train_csv = os.path.join(csv_folder, 'train_tess.csv')
                test_csv = os.path.join(csv_folder, 'test_tess.csv')
                
                if not os.path.exists(train_csv):
                    write_tess_csv(tess_path, self.emotions,
                                  train_csv, test_csv, verbose=self.verbose)
                
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
        
        if self.use_emodb:
            emodb_path = os.path.join(self.data_path, 'emodb')
            if os.path.exists(emodb_path):
                train_csv = os.path.join(csv_folder, 'train_emodb.csv')
                test_csv = os.path.join(csv_folder, 'test_emodb.csv')
                
                if not os.path.exists(train_csv):
                    write_emodb_csv(emodb_path, self.emotions,
                                   train_csv, test_csv, verbose=self.verbose)
                
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
        
        if self.use_custom:
            custom_path = os.path.join(self.data_path, 'custom')
            if os.path.exists(custom_path):
                train_csv = os.path.join(csv_folder, 'train_custom.csv')
                test_csv = os.path.join(csv_folder, 'test_custom.csv')
                
                if not os.path.exists(train_csv):
                    write_custom_csv(custom_path, self.emotions,
                                    train_csv, test_csv, verbose=self.verbose)
                
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
        
        if self.use_hindi:
            hindi_path = os.path.join(self.data_path, 'hindi')
            if os.path.exists(hindi_path):
                train_csv = os.path.join(csv_folder, 'train_hindi.csv')
                test_csv = os.path.join(csv_folder, 'test_hindi.csv')
                
                if not os.path.exists(train_csv):
                    write_hindi_csv(hindi_path, self.emotions,
                                   train_csv, test_csv, verbose=self.verbose)
                
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
        
        train_csvs = [c for c in train_csvs if os.path.exists(c)]
        test_csvs = [c for c in test_csvs if os.path.exists(c)]
        
        if not train_csvs:
            if self.verbose:
                print("No dataset found. Please add data to the 'data' folder.")
            return
        
        extractor = AudioExtractor(
            audio_config=self.audio_config,
            emotions=self.emotions,
            balance=self.balance,
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
