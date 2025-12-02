"""
utils.py - Core Utility Functions for Speech Emotion Recognition
Author: Shirssack

This module contains the fundamental feature extraction functions
that form the backbone of the entire SER system.
"""

import numpy as np
import librosa
import soundfile as sf
import os


def extract_feature(file_name, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False):
    """
    Extract acoustic features from an audio file.
    
    Parameters:
    -----------
    file_name : str
        Path to the audio file
    mfcc : bool
        Extract Mel-Frequency Cepstral Coefficients (40 values)
    chroma : bool
        Extract Chromagram features (12 values)
    mel : bool
        Extract Mel Spectrogram (128 values)
    contrast : bool
        Extract Spectral Contrast (7 values)
    tonnetz : bool
        Extract Tonal Centroid Features (6 values)
    
    Returns:
    --------
    numpy.ndarray
        Concatenated feature vector
        Default size: 40 (MFCC) + 12 (Chroma) + 128 (Mel) = 180
    """
    
    # Read audio file
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        # Handle stereo by converting to mono
        if len(X.shape) > 1:
            X = np.mean(X, axis=1)
    
    # Initialize empty result array
    result = np.array([])
    
    # Compute STFT once if needed for chroma or contrast
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    
    # MFCC
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, 
            sr=sample_rate, 
            n_mfcc=40
        ).T, axis=0)
        result = np.hstack((result, mfccs))
    
    # Chroma
    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(
            S=stft, 
            sr=sample_rate
        ).T, axis=0)
        result = np.hstack((result, chroma_feat))
    
    # Mel Spectrogram
    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(
            y=X, 
            sr=sample_rate
        ).T, axis=0)
        result = np.hstack((result, mel_feat))
    
    # Spectral Contrast
    if contrast:
        contrast_feat = np.mean(librosa.feature.spectral_contrast(
            S=stft, 
            sr=sample_rate
        ).T, axis=0)
        result = np.hstack((result, contrast_feat))
    
    # Tonnetz
    if tonnetz:
        harmonic = librosa.effects.harmonic(X)
        tonnetz_feat = np.mean(librosa.feature.tonnetz(
            y=harmonic, 
            sr=sample_rate
        ).T, axis=0)
        result = np.hstack((result, tonnetz_feat))
    
    return result


def get_feature_vector_length(mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False):
    """Calculate the total length of the feature vector."""
    length = 0
    if mfcc:
        length += 40
    if chroma:
        length += 12
    if mel:
        length += 128
    if contrast:
        length += 7
    if tonnetz:
        length += 6
    return length


# Emotion label mappings
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprised'
}

EMODB_EMOTIONS = {
    'W': 'angry',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happy',
    'T': 'sad',
    'N': 'neutral'
}

EMOTION_TO_INT = {
    'angry': 1,
    'sad': 2,
    'neutral': 3,
    'ps': 4,
    'happy': 5
}


if __name__ == "__main__":
    print("Feature vector length (default):", get_feature_vector_length())
    print("Feature vector length (all):", get_feature_vector_length(
        mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True
    ))
