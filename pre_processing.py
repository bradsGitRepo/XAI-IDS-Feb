#!/usr/bin/env python3

"""
pre_processing.py
-----------------

This script merges the functionality of:
    - Training a preprocessor (scalers, label encoders) on a CSV dataset
    - Transforming new CSV data using a pre-trained preprocessor

It provides:
    1) train_preprocessor() to fit a DataPreprocessor on a CSV (with optional minmax or standard scaling).
    2) transform_data() to load an existing preprocessor state and apply it to new data.

Classes inside:
    - DataCleaner, FeatureNormalizer, DataPreprocessor:
      * Each handles different steps of data cleaning, label-encoding, and normalization.

How to Run:
    - For Training:
      python3 pre_processing.py --mode train
          --training_file captured_data/training/ics_capture.csv
          --save_dir models/preprocessing
          --normalization_method minmax
    - For Transforming a csv which has been captured without preprocessing being applied:
      python3 pre_processing.py --mode transform
          --input_file captured_data/training/ics_capture.csv
          --output_dir processed_data
          --preprocessor_state models/preprocessing
          --normalization minmax

Overall Use in the IDS:
    1) You can use the "train" mode to fit the scaling and encoding on the training CSV.
    2) Then use the "transform" mode (with the saved state) to apply the same transformations
       to new data or different CSVs to ensure consistent feature representation for the LSTM model.

Dependencies:
    - Uses scikit-learn for scaling.
    - Called typically before 'train_lstm_autoencoder.py' so you have preprocessed data for training the model.
    - Additional references are made to 'simple_preprocessor.py' if you're using real-time sliding window logic.

This script is typically run on CSV files that were generated from "initial_data_capture.py".
"""

import argparse
import os
import glob
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

##############################
# DataCleaner, FeatureNormalizer, DataPreprocessor
##############################

class DataCleaner:
    ...

class FeatureNormalizer:
    ...

class DataPreprocessor:
    ...

##############################
# Train Preprocessor Function
##############################

def train_preprocessor(training_file=None, save_dir='models/preprocessing', normalization_method='minmax'):
    ...

##############################
# Transform Function
##############################

def transform_data(input_file, output_dir, preprocessor_state, normalization_method='minmax'):
    ...

##############################
# Main CLI
##############################

def main():
    parser = argparse.ArgumentParser(description='Merged Preprocessing Script (Train or Transform)')
    parser.add_argument('--mode', required=True, choices=['train','transform'],
                        help='Choose whether to train or transform.')

    # Training Args
    parser.add_argument('--training_file', type=str, help='Path to specific training file to use')
    parser.add_argument('--save_dir', type=str, default='models/preprocessing',
                        help='Directory to save preprocessor state')
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'standard'],
                        help='Method for normalization')

    # Transform Args
    parser.add_argument('--input_file', type=str, help='Path to CSV file for transformation')
    parser.add_argument('--output_dir', type=str, help='Directory to save transformed CSV')
    parser.add_argument('--preprocessor_state', type=str,
                        help='Directory containing trained preprocessor state for transformation')
    parser.add_argument('--normalization', type=str, default='minmax',
                        choices=['minmax', 'standard'],
                        help='Normalization method (for transform)')

    args = parser.parse_args()

    if args.mode == 'train':
        train_preprocessor(
            training_file=args.training_file,
            save_dir=args.save_dir,
            normalization_method=args.normalization_method
        )
    elif args.mode == 'transform':
        if not args.input_file or not args.output_dir or not args.preprocessor_state:
            parser.error("--input_file, --output_dir, and --preprocessor_state are required in transform mode.")

        transform_data(
            input_file=args.input_file,
            output_dir=args.output_dir,
            preprocessor_state=args.preprocessor_state,
            normalization_method=args.normalization
        )

if __name__ == "__main__":
    main()