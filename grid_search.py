"""
grid_search.py - Hyperparameter Optimization via Grid Search
Author: Shirssack
"""

import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    BaggingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from data_extractor import AudioExtractor
from parameters import CLASSIFIER_PARAMS, CLASSIFIER_PARAMS_FAST, get_param_grid


MODELS = {
    'SVC': SVC,
    'MLPClassifier': MLPClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'BaggingClassifier': BaggingClassifier,
    'AdaBoostClassifier': AdaBoostClassifier
}


class GridSearchOptimizer:
    """Performs grid search optimization for SER models."""
    
    def __init__(self, emotions=['sad', 'neutral', 'happy'], 
                 data_path='data', features=['mfcc', 'chroma', 'mel'],
                 balance=True, fast=False, cv=5, n_jobs=-1, verbose=1):
        
        self.emotions = emotions
        self.data_path = data_path
        self.features = features
        self.balance = balance
        self.fast = fast
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.audio_config = {
            'mfcc': 'mfcc' in features,
            'chroma': 'chroma' in features,
            'mel': 'mel' in features,
            'contrast': 'contrast' in features,
            'tonnetz': 'tonnetz' in features
        }
        
        self.results = {}
        self.best_estimators = {}
        
        os.makedirs('grid', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        self.scaler = StandardScaler()
        self._load_data()
    
    def _load_data(self):
        if self.verbose:
            print("\nLoading dataset...")
        
        csv_folder = os.path.join(self.data_path, 'csv')
        train_csvs = []
        test_csvs = []
        
        for dataset in ['ravdess', 'tess', 'emodb', 'hindi', 'custom']:
            train_csv = os.path.join(csv_folder, f'train_{dataset}.csv')
            test_csv = os.path.join(csv_folder, f'test_{dataset}.csv')
            
            if os.path.exists(train_csv) and os.path.exists(test_csv):
                train_csvs.append(train_csv)
                test_csvs.append(test_csv)
        
        if not train_csvs:
            raise FileNotFoundError("No CSV files found. Run EmotionRecognizer first.")
        
        extractor = AudioExtractor(
            audio_config=self.audio_config,
            emotions=self.emotions,
            balance=self.balance,
            verbose=self.verbose
        )
        
        self.X_train, self.y_train, self.X_test, self.y_test = extractor.extract_from_csv(
            train_csvs, test_csvs
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        if self.verbose:
            print(f"Loaded: {len(self.X_train)} train, {len(self.X_test)} test")
    
    def grid_search_model(self, model_name):
        """Run grid search for a specific model."""
        
        param_grid = get_param_grid(model_name, fast=self.fast)
        
        if not param_grid:
            return None
        
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        if self.verbose:
            print(f"\n  {model_name}: {total_combinations} combinations")
        
        model = MODELS[model_name]()
        
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        elapsed_time = time.time() - start_time
        test_score = grid_search.best_estimator_.score(self.X_test_scaled, self.y_test)
        
        result = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_score': test_score,
            'best_estimator': grid_search.best_estimator_,
            'elapsed_time': elapsed_time
        }
        
        if self.verbose:
            print(f"    CV: {grid_search.best_score_:.4f}, Test: {test_score:.4f}, Time: {elapsed_time:.1f}s")
        
        return result
    
    def run_grid_search(self, models=None):
        """Run grid search for all models."""
        
        if models is None:
            models = list(MODELS.keys())
        
        if self.verbose:
            print(f"\nStarting Grid Search ({'Fast' if self.fast else 'Full'} mode)")
            print(f"Models: {', '.join(models)}")
        
        for model_name in models:
            try:
                result = self.grid_search_model(model_name)
                
                if result:
                    self.results[model_name] = result
                    self.best_estimators[model_name] = result['best_estimator']
                    
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
        
        self._save_results()
        return self.results
    
    def _save_results(self):
        emotion_str = '_'.join([e[0].upper() for e in sorted(self.emotions)])
        
        # Save best estimators
        estimators_path = os.path.join('grid', f'best_estimators_{emotion_str}.pkl')
        with open(estimators_path, 'wb') as f:
            pickle.dump({
                'estimators': self.best_estimators,
                'scaler': self.scaler,
                'emotions': self.emotions,
                'audio_config': self.audio_config
            }, f)
        
        # Save summary CSV
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'model': model_name,
                'cv_score': result['best_cv_score'],
                'test_score': result['test_score'],
                'time': result['elapsed_time']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('test_score', ascending=False)
        summary_df.to_csv(os.path.join('grid', f'summary_{emotion_str}.csv'), index=False)
        
        if self.verbose:
            print(f"\nResults saved to grid/")
            print(summary_df[['model', 'cv_score', 'test_score']].to_string(index=False))
    
    def get_best_model(self):
        """Get the best model."""
        best = max(self.results.items(), key=lambda x: x[1]['test_score'])
        return best[0], best[1]['best_estimator']


def load_best_estimators(emotions=['sad', 'neutral', 'happy']):
    """Load pre-computed best estimators."""
    
    emotion_str = '_'.join([e[0].upper() for e in sorted(emotions)])
    estimators_path = os.path.join('grid', f'best_estimators_{emotion_str}.pkl')
    
    if not os.path.exists(estimators_path):
        raise FileNotFoundError(f"No estimators for {emotions}. Run grid_search.py first.")
    
    with open(estimators_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Grid search for SER models")
    parser.add_argument('--emotions', '-e', type=str, default='sad,neutral,happy,angry',
                        help='Comma-separated list of emotions (default: sad,neutral,happy,angry)')
    parser.add_argument('--models', '-m', type=str, default=None)
    parser.add_argument('--fast', '-f', action='store_true')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--data-path', '-d', type=str, default='data')
    
    args = parser.parse_args()
    
    emotions = [e.strip() for e in args.emotions.split(',')]
    models = [m.strip() for m in args.models.split(',')] if args.models else None
    
    optimizer = GridSearchOptimizer(
        emotions=emotions,
        data_path=args.data_path,
        fast=args.fast,
        cv=args.cv
    )
    
    optimizer.run_grid_search(models=models)
    
    best_name, _ = optimizer.get_best_model()
    print(f"\n[BEST] {best_name} ({optimizer.results[best_name]['test_score']:.2%})")


if __name__ == "__main__":
    main()
