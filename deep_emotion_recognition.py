"""
deep_emotion_recognition.py - Deep Learning Based Emotion Recognition
Author: Shirssack
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

from emotion_recognition import EmotionRecognizer
from utils import extract_feature


class DeepEmotionRecognizer(EmotionRecognizer):
    """Deep Learning based Speech Emotion Recognizer using LSTM/GRU."""
    
    def __init__(self, emotions=['sad', 'neutral', 'happy'],
                 features=['mfcc', 'chroma', 'mel'],
                 n_rnn_layers=2, n_dense_layers=2,
                 rnn_units=128, dense_units=128,
                 cell_type='LSTM', dropout=0.3,
                 balance=True, data_path='data', verbose=1,
                 **kwargs):
        
        self.n_rnn_layers = n_rnn_layers
        self.n_dense_layers = n_dense_layers
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.cell_type = cell_type.upper()
        
        if self.cell_type == 'LSTM':
            self.cell = LSTM
        elif self.cell_type == 'GRU':
            self.cell = GRU
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        
        super().__init__(
            model=None,
            emotions=emotions,
            features=features,
            balance=balance,
            data_path=data_path,
            verbose=verbose,
            **kwargs
        )
        
        self._build_model()
    
    def _build_model(self):
        """Build the neural network."""
        
        if self.X_train is None:
            input_length = 180
        else:
            input_length = self.X_train.shape[1]
        
        n_classes = len(self.emotions)
        
        self.model = Sequential()
        
        # RNN Layers
        for i in range(self.n_rnn_layers):
            if i == 0:
                self.model.add(self.cell(
                    self.rnn_units,
                    return_sequences=(i < self.n_rnn_layers - 1),
                    input_shape=(1, input_length)
                ))
            else:
                self.model.add(self.cell(
                    self.rnn_units,
                    return_sequences=(i < self.n_rnn_layers - 1)
                ))
            
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))
        
        # Dense Layers
        for i in range(self.n_dense_layers):
            self.model.add(Dense(self.dense_units, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))
        
        # Output
        self.model.add(Dense(n_classes, activation='softmax'))
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if self.verbose:
            print(f"\nModel: {self.cell_type} with {self.n_rnn_layers} RNN layers")
            self.model.summary()
    
    def _prepare_data_for_rnn(self, X):
        return X.reshape((X.shape[0], 1, X.shape[1]))
    
    def _encode_labels(self, y):
        y_int = np.array([self.emotion_to_int[label] for label in y])
        return to_categorical(y_int, num_classes=len(self.emotions))
    
    def _decode_labels(self, y_onehot):
        y_int = np.argmax(y_onehot, axis=1)
        return np.array([self.int_to_emotion[i] for i in y_int])
    
    def train(self, epochs=50, batch_size=32, validation_split=0.1):
        """Train the model."""
        
        if self.X_train is None:
            raise ValueError("No training data available")
        
        X_train_rnn = self._prepare_data_for_rnn(self.X_train)
        y_train_encoded = self._encode_labels(self.y_train)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/best_deep_model.keras', monitor='val_accuracy', save_best_only=True)
        ]
        
        os.makedirs('models', exist_ok=True)
        
        if self.verbose:
            print(f"\nTraining {self.cell_type} for {epochs} epochs...")
        
        history = self.model.fit(
            X_train_rnn,
            y_train_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        self._is_trained = True
        
        if self.verbose:
            print(f"\nTest accuracy: {self.test_score():.2%}")
        
        return history
    
    def test_score(self):
        X_test_rnn = self._prepare_data_for_rnn(self.X_test)
        y_test_encoded = self._encode_labels(self.y_test)
        _, accuracy = self.model.evaluate(X_test_rnn, y_test_encoded, verbose=0)
        return accuracy
    
    def train_score(self):
        X_train_rnn = self._prepare_data_for_rnn(self.X_train)
        y_train_encoded = self._encode_labels(self.y_train)
        _, accuracy = self.model.evaluate(X_train_rnn, y_train_encoded, verbose=0)
        return accuracy
    
    def predict(self, audio_path):
        features = extract_feature(audio_path, **self.audio_config)
        features_scaled = self.scaler.transform([features])
        features_rnn = self._prepare_data_for_rnn(features_scaled)
        
        prediction_proba = self.model.predict(features_rnn, verbose=0)
        prediction_idx = np.argmax(prediction_proba[0])
        
        return self.int_to_emotion[prediction_idx]
    
    def predict_proba(self, audio_path):
        features = extract_feature(audio_path, **self.audio_config)
        features_scaled = self.scaler.transform([features])
        features_rnn = self._prepare_data_for_rnn(features_scaled)
        
        prediction_proba = self.model.predict(features_rnn, verbose=0)[0]
        
        return {self.int_to_emotion[i]: float(p) for i, p in enumerate(prediction_proba)}
    
    def confusion_matrix(self, percentage=True, as_dataframe=True):
        X_test_rnn = self._prepare_data_for_rnn(self.X_test)
        y_pred_proba = self.model.predict(X_test_rnn, verbose=0)
        y_pred = self._decode_labels(y_pred_proba)
        
        cm = confusion_matrix(self.y_test, y_pred, labels=self.emotions)
        
        if percentage:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        if as_dataframe:
            cm = pd.DataFrame(cm,
                index=[f"true_{e}" for e in self.emotions],
                columns=[f"pred_{e}" for e in self.emotions])
        
        return cm
    
    def save_model(self, path='models/deep_emotion_model'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.model.save(f"{path}.keras")
        
        import pickle
        with open(f"{path}_config.pkl", 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'emotions': self.emotions,
                'audio_config': self.audio_config,
                'emotion_to_int': self.emotion_to_int,
                'int_to_emotion': self.int_to_emotion
            }, f)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load_model(self, path='models/deep_emotion_model'):
        import pickle
        
        self.model = load_model(f"{path}.keras")
        
        with open(f"{path}_config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        self.scaler = config['scaler']
        self.emotions = config['emotions']
        self.audio_config = config['audio_config']
        self.emotion_to_int = config['emotion_to_int']
        self.int_to_emotion = config['int_to_emotion']
        self._is_trained = True


if __name__ == "__main__":
    print("DeepEmotionRecognizer module loaded.")
    print("\nUsage:")
    print("  rec = DeepEmotionRecognizer(emotions=['happy', 'sad'], cell_type='LSTM')")
    print("  rec.train(epochs=50)")
    print("  print(rec.predict('audio.wav'))")
