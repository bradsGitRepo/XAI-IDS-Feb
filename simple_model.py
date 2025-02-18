import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

"""
simple_model.py
---------------

This script defines 'SimpleLSTMAutoencoder', a simplified LSTM-based autoencoder with:
    - A symmetrical architecture: 2-layer encoder, 2-layer decoder
    - Anomaly detection method (detect_anomaly) based on reconstruction MSE
    - Integrated gradients method (compute_integrated_gradients) for interpretability

Key class:
    SimpleLSTMAutoencoder(seq_length, num_features):
      1) _build_model: constructs the LSTM AE
      2) load_weights: loads trained weights from .h5 or .keras
      3) detect_anomaly: returns (is_anomaly, reconstruction_error)
      4) compute_integrated_gradients: returns attributions for each feature
      5) create_sequences: helper to create 3D sequences from 2D data

How it Connects:
    - Usually trained via 'train_lstm_autoencoder.py' or reloaded in 'simple_pipeline.py' for real-time detection
    - The 'detect_anomaly' method is used in the pipeline worker to decide if a sample is anomalous
    - 'compute_integrated_gradients' used in the IG worker to produce feature-level attributions

How to Use:
    - The pipeline or your training script imports SimpleLSTMAutoencoder
    - You specify sequence length, number of features, build or load weights
    - Then call detect_anomaly(...) on 3D input to get anomaly results
    - If anomaly, compute IG to see top features contributing

Dependencies:
    - TensorFlow / Keras
    - Typically scikit-learn or numpy for supporting tasks
"""

class SimpleLSTMAutoencoder:
    def __init__(self, seq_length=20, num_features=42):
        self.seq_length = seq_length
        self.num_features = num_features
        self.model = self._build_model(seq_length, num_features)

    def _build_model(self, seq_length, num_features):
        inputs = tf.keras.Input(shape=(seq_length, num_features))

        # Encoder
        x = layers.LSTM(32, return_sequences=True, activation='tanh')(inputs)
        x = layers.LSTM(16, return_sequences=False, activation='tanh')(x)

        # Repeat latent
        x = layers.RepeatVector(seq_length)(x)

        # Decoder
        x = layers.LSTM(16, return_sequences=True, activation='tanh')(x)
        x = layers.LSTM(32, return_sequences=True, activation='tanh')(x)
        x = layers.TimeDistributed(layers.Dense(num_features))(x)

        model = Model(inputs, x)
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_weights(self, path: str):
        self.model.load_weights(path)

    def detect_anomaly(self, sequence_3d: np.ndarray, threshold: float = 0.5):
        recon = self.model.predict(sequence_3d, verbose=0)  # shape=(1, seq_len, num_feat)
        mse = float(np.mean((recon - sequence_3d)**2))

        # For demonstration, normalizing MSE to [0,1] with an exponential approach
        normalized_score = 1 - np.exp(-mse / 1000.0)
        is_anomaly = normalized_score > threshold
        return (is_anomaly, normalized_score)

    def compute_integrated_gradients(self, sequence_3d: np.ndarray, baseline_3d: np.ndarray = None, steps: int = 50) -> np.ndarray:
        if baseline_3d is None:
            baseline_3d = np.zeros_like(sequence_3d)

        import tensorflow as tf
        sequence_tensor = tf.cast(sequence_3d, tf.float32)
        baseline_tensor = tf.cast(baseline_3d, tf.float32)

        def reconstruction_error(x):
            recon_x = self.model(x)
            return tf.reduce_mean(tf.square(recon_x - x), axis=[1, 2])

        total_grad = tf.zeros_like(sequence_tensor)
        for alpha in tf.linspace(0.0, 1.0, steps):
            inter_x = baseline_tensor + alpha * (sequence_tensor - baseline_tensor)
            with tf.GradientTape() as tape:
                tape.watch(inter_x)
                out = reconstruction_error(inter_x)
            grads = tape.gradient(out, inter_x)
            if grads is not None:
                total_grad += grads

        avg_grad = total_grad / float(steps)
        ig = (sequence_tensor - baseline_tensor) * avg_grad
        ig_raw = ig.numpy()[0]

        ig_abs = np.abs(ig_raw)
        ig_normalized = ig_raw * (10.0 / (np.max(ig_abs) + 1e-10))

        return ig_normalized

    @staticmethod
    def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
        sequences = []
        for i in range(len(data) - seq_length + 1):
            seq = data[i : i + seq_length]
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)