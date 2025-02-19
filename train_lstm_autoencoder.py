#!/usr/bin/env python3

"""
train_lstm_autoencoder.py
-------------------------

This script trains a LSTM autoencoder on a preprocessed CSV dataset, then computes:
  - Reconstruction error
  - Anomaly threshold from a percentile of validation errors
  - Integrated gradients for anomalies

Steps:
    1) Load a preprocessed CSV (so no further engineering is done here).
    2) Create sliding windows (sequence_length=20 by default).
    3) Train LSTM AE, pick the best model from validation via callbacks.
    4) Evaluate the entire dataset, compute reconstruction error + IG for anomalies.
    5) Output:
       - Best model saved as best_model.h5 / best_model.keras
       - A CSV 'lstm_anomaly_explanations.csv' with columns:
         * Original numeric columns
         * recon_error
         * is_anomaly
         * pos_sum / neg_sum (IG sums)
         * ig_explanation (feature->IG)
         * most_important_features (top 5 by absolute IG)

Usage:
    - Typically run AFTER you have a training dataset (e.g. from 'initial_data_capture.py'
      or other means) that is preprocessed (using 'pre_processing.py').
    - Example:
      python3 train_lstm_autoencoder.py --input-file processed_data/preprocessed_data.csv
          --seq-length 20
          --epochs 10
          --batch-size 32
          --save-dir models/lstm_autoencoder
          --threshold-percentile 95.0

Dependencies:
    - 'simple_model.py' for the SimpleLSTMAutoencoder definition
    - Keras/TensorFlow for training
    - Numpy/Pandas for data handling

In the Overall IDS:
    - You do data capture -> feature engineering -> pre-processing -> training -> produce
      a saved LSTM model + threshold.
    - The 'simple_pipeline.py' then loads this model to do real-time anomaly detection.

"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json

from simple_model import SimpleLSTMAutoencoder

def create_sequences(data: np.ndarray, seq_length: int):
    """
    Split 2D data (num_samples, num_features) into 3D sequences (num_sequences, seq_length, num_features).
    We also return an index array for mapping each 3D slice back to its "last row" index in the original data.
    """
    sequences = []
    indexes = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i : i + seq_length]
        sequences.append(seq)
        # We'll consider i+seq_length-1 as the "final row index" for this window
        indexes.append(i + seq_length - 1)
    return np.array(sequences, dtype=np.float32), np.array(indexes, dtype=np.int32)

def print_error_summary(errors: np.ndarray, label="Set"):
    """
    Print a summary of reconstruction errors: min, max, mean, percentiles.
    """
    min_err = np.min(errors)
    max_err = np.max(errors)
    mean_err = np.mean(errors)
    p25 = np.percentile(errors, 25)
    p50 = np.percentile(errors, 50)
    p75 = np.percentile(errors, 75)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)
    
    print(f"\nReconstruction Error Summary ({label}):")
    print(f"  Min: {min_err:.6f}")
    print(f"  25th pct: {p25:.6f}")
    print(f"  Median: {p50:.6f}")
    print(f"  75th pct: {p75:.6f}")
    print(f"  90th pct: {p90:.6f}")
    print(f"  95th pct: {p95:.6f}")
    print(f"  Max: {max_err:.6f}")
    print(f"  Mean: {mean_err:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="Train an LSTM Autoencoder with IG, store anomalies + explanations in CSV.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to preprocessed CSV file.")
    parser.add_argument("--seq_length", type=int, default=20, help="Sliding window size (sequence length).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--save_dir", type=str, default="models/lstm_autoencoder", help="Where to save the model.")
    parser.add_argument("--threshold_percentile", type=float, default=95.0,
                        help="Percentile for choosing anomaly threshold from validation errors.")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print(f"Loading CSV from {args.input_file} ...")
    df = pd.read_csv(args.input_file)
    print(f"Data shape: {df.shape}")
    feature_names = df.columns.tolist()  # keep track of feature names

    data_array = df.values.astype("float32")

    # 2) Create sequences
    print("Creating sliding window sequences ...")
    sequences, indexes = create_sequences(data_array, seq_length=args.seq_length)
    print(f"Sequences shape: {sequences.shape}")

    # 3) Simple train/validation split
    train_size = int(0.8 * len(sequences))
    X_train = sequences[:train_size]
    X_val = sequences[train_size:] if train_size < len(sequences) else sequences[:1]

    # 4) Initialize LSTM autoencoder
    num_features = data_array.shape[1]
    autoencoder = SimpleLSTMAutoencoder(seq_length=args.seq_length, num_features=num_features)
    model = autoencoder.model

    model.summary()
    model.compile(optimizer="adam", loss="mse")

    # 5) Train with callbacks
    best_model_h5_path = os.path.join(args.save_dir, "best_model.h5")
    best_model_keras_path = os.path.join(args.save_dir, "best_model.keras")
    
    class DualFormatModelCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, h5_path, keras_path):
            super().__init__()
            self.h5_path = h5_path
            self.keras_path = keras_path
            self.best_val_loss = float('inf')
            
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save(self.h5_path, save_format='h5')
                self.model.save(self.keras_path, save_format='keras')
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        DualFormatModelCheckpoint(best_model_h5_path, best_model_keras_path)
    ]

    print("Starting training ...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True
    )

    # Reload best model (load from keras format as it's used in simple_pipeline.py)
    print(f"Loading best model from {best_model_keras_path} ...")
    model = tf.keras.models.load_model(best_model_keras_path)

    # Compute reconstruction errors on training set
    recon_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.mean(np.square(recon_train - X_train), axis=2), axis=1)
    print_error_summary(train_errors, label="Training set")

    # 6) Compute threshold from validation set
    if len(X_val) < 1:
        # Fallback if we have no val set
        print("Warning: No validation set. Using a default threshold=0.5")
        threshold = 0.5
        val_errors = []
    else:
        # Predict reconstruction for val
        recon_val = model.predict(X_val, verbose=0)
        val_errors = np.mean(np.mean(np.square(recon_val - X_val), axis=2), axis=1)
        print_error_summary(val_errors, label="Validation set")
        threshold = np.percentile(val_errors, args.threshold_percentile)
    print(f"Chosen anomaly threshold={threshold:.4f} (percentile={args.threshold_percentile})")

    # Save threshold to a text file
    threshold_file_path = os.path.join(args.save_dir, "threshold.txt")
    with open(threshold_file_path, "w") as f:
        f.write(str(threshold))
    print(f"Threshold saved to: {threshold_file_path}")

    # 7) Evaluate entire dataset => detect anomalies => compute IG => store CSV
    print("Evaluating entire dataset for anomalies + integrated gradients...")
    recon_all = model.predict(sequences, verbose=0)
    errors_all = np.mean(np.mean(np.square(recon_all - sequences), axis=2), axis=1)
    is_anomaly = errors_all > threshold

    # Prepare final DataFrame
    results_df = df.copy()
    results_df["recon_error"] = 0.0
    results_df["is_anomaly"] = 0
    results_df["pos_sum"] = 0.0
    results_df["neg_sum"] = 0.0
    results_df["ig_explanation"] = "N/A"
    results_df["most_important_features"] = "N/A"

    total_sequences = len(indexes)

    for seq_idx, end_idx in enumerate(indexes):
        # Print percentage progress
        pct = 100.0 * (seq_idx + 1) / total_sequences
        print(f"Computing explanations: {seq_idx+1}/{total_sequences} ({pct:.2f}%)", end="\r", flush=True)

        err = errors_all[seq_idx]
        results_df.at[end_idx, "recon_error"] = err
        if is_anomaly[seq_idx]:
            results_df.at[end_idx, "is_anomaly"] = 1
            # compute integrated gradients
            seq_3d = sequences[seq_idx:seq_idx+1]
            ig_attribs = autoencoder.compute_integrated_gradients(seq_3d)

            # sum across time dimension => shape(num_features,)
            ig_summed = ig_attribs.sum(axis=0)

            pos_sum = float(ig_summed[ig_summed > 0].sum())
            neg_sum = float(ig_summed[ig_summed < 0].sum())
            results_df.at[end_idx, "pos_sum"] = pos_sum
            results_df.at[end_idx, "neg_sum"] = neg_sum

            # build dict of feature->IG
            import json
            ig_dict = {}
            for i, feat_name in enumerate(feature_names):
                ig_dict[feat_name] = float(ig_summed[i])

            # convert to JSON
            ig_json = json.dumps(ig_dict)
            results_df.at[end_idx, "ig_explanation"] = ig_json

            # Also pick top 5 by absolute IG
            sorted_ig = sorted(ig_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            top_5 = sorted_ig[:5]
            top_5_str_parts = [f"{k}={v:.3f}" for (k, v) in top_5]
            top_5_str = "; ".join(top_5_str_parts)
            results_df.at[end_idx, "most_important_features"] = top_5_str

    # 8) Save final model & CSV
    final_model_h5_path = os.path.join(args.save_dir, "model.h5")
    final_model_keras_path = os.path.join(args.save_dir, "model.keras")
    
    # Save in both formats
    model.save(final_model_h5_path, save_format='h5')
    model.save(final_model_keras_path, save_format='keras')
    print(f"\nFinal model saved to: {final_model_h5_path} and {final_model_keras_path}")

    output_csv_path = os.path.join(args.save_dir, "lstm_anomaly_explanations.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Anomaly+IG results written to: {output_csv_path}")
    print("Training & explanation process complete.")

if __name__ == "__main__":
    main()