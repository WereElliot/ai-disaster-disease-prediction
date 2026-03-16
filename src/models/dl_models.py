from sklearn.neural_network import MLPRegressor
import numpy as np
from typing import Dict, Any

def build_lstm_model(input_shape, units=64, dropout=0.2):
    """
    NOTE: Using MLPRegressor as a fallback for LSTM due to environment limitations 
    (No TensorFlow/Keras support for Python 3.14+ yet).
    This simulates the feature extraction behavior of the LSTM.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(units, units // 2),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    return model

def create_lstm_dataset(data, lookback=12):
    """Create windows for training. Flattening for MLP fallback."""
    X, y = [], []
    for i in range(len(data) - lookback):
        # Flatten the lookback window into a single feature vector for MLP
        window = data[i:(i + lookback), :-1].flatten()
        X.append(window)
        y.append(data[i + lookback, -1])
    return np.array(X), np.array(y)
