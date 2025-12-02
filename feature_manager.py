"""
parameters.py - Hyperparameter Search Space Definitions
Author: Shirssack
"""

# SVM parameters
SVC_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'degree': [2, 3, 4],
    'probability': [True]
}

SVC_PARAMS_FAST = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
    'probability': [True]
}

# MLP parameters
MLP_PARAMS = {
    'hidden_layer_sizes': [
        (100,), (200,), (300,),
        (100, 100), (200, 100), (300, 200),
        (200, 100, 50),
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000],
    'early_stopping': [True],
    'random_state': [42]
}

MLP_PARAMS_FAST = {
    'hidden_layer_sizes': [(100,), (200,), (300,), (200, 100)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'max_iter': [500],
    'early_stopping': [True],
    'random_state': [42]
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [42]
}

RF_PARAMS_FAST = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'random_state': [42]
}

# Gradient Boosting parameters
GB_PARAMS = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0],
    'random_state': [42]
}

GB_PARAMS_FAST = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'random_state': [42]
}

# KNN parameters
KNN_PARAMS = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

KNN_PARAMS_FAST = {
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Decision Tree parameters
DT_PARAMS = {
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'random_state': [42]
}

# Bagging parameters
BAG_PARAMS = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
    'random_state': [42]
}

# AdaBoost parameters
ADA_PARAMS = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [42]
}

# Parameter collections
CLASSIFIER_PARAMS = {
    'SVC': SVC_PARAMS,
    'MLPClassifier': MLP_PARAMS,
    'RandomForestClassifier': RF_PARAMS,
    'GradientBoostingClassifier': GB_PARAMS,
    'KNeighborsClassifier': KNN_PARAMS,
    'DecisionTreeClassifier': DT_PARAMS,
    'BaggingClassifier': BAG_PARAMS,
    'AdaBoostClassifier': ADA_PARAMS
}

CLASSIFIER_PARAMS_FAST = {
    'SVC': SVC_PARAMS_FAST,
    'MLPClassifier': MLP_PARAMS_FAST,
    'RandomForestClassifier': RF_PARAMS_FAST,
    'GradientBoostingClassifier': GB_PARAMS_FAST,
    'KNeighborsClassifier': KNN_PARAMS_FAST,
    'DecisionTreeClassifier': DT_PARAMS,
    'BaggingClassifier': BAG_PARAMS,
    'AdaBoostClassifier': ADA_PARAMS
}

# Feature configurations
FEATURE_CONFIGS = {
    'mfcc_only': {'mfcc': True, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False},
    'chroma_only': {'mfcc': False, 'chroma': True, 'mel': False, 'contrast': False, 'tonnetz': False},
    'mel_only': {'mfcc': False, 'chroma': False, 'mel': True, 'contrast': False, 'tonnetz': False},
    'default': {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False},
    'all_features': {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': True, 'tonnetz': True}
}

# Emotion configurations
EMOTION_CONFIGS = {
    '3_class': ['sad', 'neutral', 'happy'],
    '4_class': ['angry', 'sad', 'neutral', 'happy'],
    '5_class': ['angry', 'sad', 'neutral', 'happy', 'fear'],
    '6_class': ['angry', 'sad', 'neutral', 'happy', 'fear', 'disgust'],
    '8_class': ['angry', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'surprised', 'calm'],
    'hindi_8_class': ['angry', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'surprised', 'sarcastic']
}


def get_param_grid(model_name, fast=False):
    """Get parameter grid for a specific model."""
    params = CLASSIFIER_PARAMS_FAST if fast else CLASSIFIER_PARAMS
    return params.get(model_name, {})


def get_feature_config(config_name='default'):
    """Get feature configuration by name."""
    return FEATURE_CONFIGS.get(config_name, FEATURE_CONFIGS['default'])


def get_emotions(config_name='3_class'):
    """Get emotion list by configuration name."""
    return EMOTION_CONFIGS.get(config_name, EMOTION_CONFIGS['3_class'])


if __name__ == "__main__":
    print("Available classifiers:", list(CLASSIFIER_PARAMS.keys()))
    print("Available emotion configs:", list(EMOTION_CONFIGS.keys()))
