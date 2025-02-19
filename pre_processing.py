#!/usr/bin/env python3

"""
Merged Preprocessing Script

This single script merges the functionality of:
- train_preprocessor.py (training the data preprocessor)
- data_preprocessing.py (transforming new data)

Usage Examples:

1) Training the preprocessor:
   python3 pre_processing.py --mode train \
       --training_file captured_data/training/ics_capture_20250213_135819.csv \
       --save_dir models/preprocessing \
       --normalization_method minmax

2) Transforming data using a trained preprocessor:
   python3 pre_processing.py --mode transform \
       --input_file captured_data/training/ics_capture_20250213_135819.csv \
       --output_dir processed_data \
       --preprocessor_state models/preprocessing \
       --normalization minmax

The DataCleaner, FeatureNormalizer, and DataPreprocessor classes
are taken from the previous data_preprocessing.py module, and
train_preprocessor logic is merged here.
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
    """
    Handles data cleaning operations including null value handling and data type conversion.
    """

    def __init__(self):
        from config import get_whitelist
        self.whitelist = get_whitelist()

        self.timestamp_features = [
            'timestamp'
        ]

        self.numeric_features = [
            'packet_size', 'modbus_length',
            'mean_interarrival_time', 'std_interarrival_time',
            'max_interarrival_time', 'mean_packet_size',
            'std_packet_size', 'packet_size_entropy',
            'arp_anomaly_score', 'tcp_syn_rate',
            'error_rate', 'modbus_rate',
            'function_code_entropy',
            'src_ip', 'dst_ip',
            'src_mac', 'dst_mac'
        ]

        self.boolean_features = [
            'is_modbus',
            'error_code'
        ]

        self.categorical_features = [
            'protocol',
            'tcp_flags',
            'modbus_function_code',
            'packet_type'
        ]

        self.label_encoders = {}
        self.next_label_index_dict = {}
        self.feature_medians = {}

    def _convert_ip_to_int(self, ip_str):
        import ipaddress
        try:
            ip_obj = ipaddress.ip_address(str(ip_str))
            if ip_obj.version == 6:
                return 0
            return int(ip_obj)
        except:
            return 0

    def _convert_mac_to_int(self, mac_str):
        try:
            cleaned = ''.join(c for c in mac_str.lower() if c in '0123456789abcdef')
            if len(cleaned) != 12:
                return 0
            return int(cleaned, 16)
        except:
            return 0

    def fit(self, df):
        for feature in self.numeric_features:
            if feature in df.columns:
                if feature not in ['src_ip', 'dst_ip', 'src_mac', 'dst_mac']:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    self.feature_medians[feature] = df[feature].median()

        for feature in self.categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna('UNKNOWN').astype(str)
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = {}
                    self.next_label_index_dict[feature] = 0
                for val in df[feature].unique():
                    if val not in self.label_encoders[feature]:
                        self.label_encoders[feature][val] = self.next_label_index_dict[feature]
                        self.next_label_index_dict[feature] += 1

    def transform(self, df):
        df = df.copy(deep=True)

        from datetime import datetime
        for feature in self.timestamp_features:
            if feature in df.columns:
                df[feature] = pd.to_datetime(df[feature], errors='coerce')
                current_time = datetime.now().timestamp()
                df[feature] = df[feature].apply(lambda x: x.timestamp() if pd.notnull(x) else current_time)

        if 'src_ip' in df.columns:
            df['src_ip'] = df['src_ip'].astype(str).apply(self._convert_ip_to_int)
        if 'dst_ip' in df.columns:
            df['dst_ip'] = df['dst_ip'].astype(str).apply(self._convert_ip_to_int)

        if 'src_mac' in df.columns:
            df['src_mac'] = df['src_mac'].astype(str).apply(self._convert_mac_to_int)
        if 'dst_mac' in df.columns:
            df['dst_mac'] = df['dst_mac'].astype(str).apply(self._convert_mac_to_int)

        for feature in self.numeric_features:
            if feature in df.columns:
                if feature in self.feature_medians:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    median_val = float(self.feature_medians[feature])
                    df.loc[:, feature] = df[feature].fillna(median_val)
                else:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

        for feature in self.boolean_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0).astype(int)

        for feature in self.categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna('UNKNOWN').astype(str)
                if feature in self.label_encoders:
                    df[feature] = df[feature].map(
                        lambda x: self.label_encoders[feature].get(x, self.next_label_index_dict[feature])
                    ).astype('int32')
                    self.next_label_index_dict[feature] = max(
                        max(self.label_encoders[feature].values(), default=0) + 1,
                        self.next_label_index_dict[feature]
                    )
                else:
                    df[feature] = 0

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save_state(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(f"{directory}/feature_medians.pkl", 'wb') as f:
            pickle.dump(self.feature_medians, f)
        encoders_to_save = {
            'label_encoders': self.label_encoders,
            'next_label_index_dict': self.next_label_index_dict
        }
        with open(f"{directory}/label_encoders.pkl", 'wb') as f:
            pickle.dump(encoders_to_save, f)

    def load_state(self, directory):
        with open(f"{directory}/feature_medians.pkl", 'rb') as f:
            self.feature_medians = pickle.load(f)
        with open(f"{directory}/label_encoders.pkl", 'rb') as f:
            encoders_data = pickle.load(f)
            self.label_encoders = encoders_data['label_encoders']
            self.next_label_index_dict = encoders_data['next_label_index_dict']


class FeatureNormalizer:
    """
    Handles feature normalization using various techniques.
    """
    def __init__(self, method='minmax'):
        self.method = method
        self.scalers = {}
        self.time_features = [
            'mean_interarrival_time', 'std_interarrival_time',
            'min_interarrival_time', 'max_interarrival_time'
        ]
        self.count_features = [
            'packet_size', 'modbus_length', 'unique_src_ips',
            'unique_src_macs', 'src_port', 'dst_port'
        ]
        self.skip_features = [
            'arp_anomaly_score', 'tcp_syn_rate', 'tcp_rst_rate',
            'tcp_fin_rate', 'modbus_error_rate'
        ]
        self.fitted = False
        self.standard_scaler = None
        self.minmax_scaler = None
        self.label_encoders = {}
        self.categorical_cols = []
        self.numeric_cols = []
        self.fitted_columns = []

    def fit(self, df):
        for feature in self.time_features:
            if feature in df.columns:
                data = df[[feature]].values
                if not np.isnan(data).all():
                    from sklearn.preprocessing import StandardScaler
                    self.scalers[feature] = StandardScaler(with_mean=True, with_std=True)
                    self.scalers[feature].fit(data)
        
        for feature in self.count_features:
            if feature in df.columns:
                data = df[[feature]].values
                if not np.isnan(data).all():
                    from sklearn.preprocessing import MinMaxScaler
                    self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
                    self.scalers[feature].fit(data)
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = X.copy()
        X_scaled = self.standard_scaler.transform(X_scaled)
        X_scaled = self.minmax_scaler.transform(X_scaled)
        return X_scaled
    
    def fit_transform(self, df):
        if isinstance(df, pd.DataFrame):
            X = df.values
        else:
            X = df.copy()
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self.standard_scaler = StandardScaler()
        X_scaled = self.standard_scaler.fit_transform(X)
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.minmax_scaler.fit_transform(X_scaled)
        self.fitted = True
        return X_scaled
    
    def save_state(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, 'standard_scaler.pkl'), 'wb') as f:
            pickle.dump(self.standard_scaler, f)
        with open(os.path.join(directory, 'minmax_scaler.pkl'), 'wb') as f:
            pickle.dump(self.minmax_scaler, f)
        with open(os.path.join(directory, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(os.path.join(directory, 'columns.pkl'), 'wb') as f:
            pickle.dump({
                'categorical_cols': self.categorical_cols,
                'numeric_cols': self.numeric_cols,
                'fitted_columns': self.fitted_columns
            }, f)

        
    def load_state(self, directory):
        with open(os.path.join(directory, 'standard_scaler.pkl'), 'rb') as f:
            self.standard_scaler = pickle.load(f)
        with open(os.path.join(directory, 'minmax_scaler.pkl'), 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        with open(os.path.join(directory, 'label_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(os.path.join(directory, 'columns.pkl'), 'rb') as f:
            cols = pickle.load(f)
            self.categorical_cols = cols['categorical_cols']
            self.numeric_cols = cols['numeric_cols']
            self.fitted_columns = cols.get('fitted_columns', [])
        self.fitted = True


class DataPreprocessor:
    """
    High-level interface combining cleaning and normalization.
    """

    def __init__(self, normalization_method='minmax'):
        self.normalization_method = normalization_method
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.categorical_cols = []
        self.numeric_cols = []
        self.fitted = False
        self.feature_order = []
        
        self.excluded_features = [
            'timestamp'
        ]

    def _encode_categorical(self, df):
        encoded = df.copy()
        self.categorical_cols = [
            col for col in df.select_dtypes(include=['object']).columns
            if col not in self.excluded_features
        ]
        self.numeric_cols = [
            col for col in df.select_dtypes(include=['int64','float64']).columns
            if col not in self.excluded_features
        ]

        for col in self.categorical_cols:
            if col not in self.label_encoders:
                unique_values = df[col].unique()
                self.label_encoders[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }
            encoded[col] = df[col].map(
                lambda x: self.label_encoders[col].get(x, -1)
            )
        return encoded

    def _handle_missing_values(self, df):
        numeric_cols = [
            c for c in df.select_dtypes(include=['int64','float64']).columns
            if c not in self.excluded_features
        ]
        df[numeric_cols] = df[numeric_cols].fillna(0)

        categorical_cols = [
            c for c in df.select_dtypes(include=['object']).columns
            if c not in self.excluded_features
        ]
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        return df

    def fit(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        self.feature_order = [col for col in data.columns if col not in self.excluded_features]
        self.logger.info(f"Tracking {len(self.feature_order)} features in order")

        data = data.drop(columns=self.excluded_features, errors='ignore').copy()
        data = self._handle_missing_values(data)
        encoded_data = self._encode_categorical(data)

        self.fitted_columns = encoded_data.columns.tolist()

        X = encoded_data.values.astype(float)
        self.standard_scaler.fit(X)
        X_std = self.standard_scaler.transform(X)
        self.minmax_scaler.fit(X_std)

        self.fitted = True
        return self

    def transform(self, data):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data = data.drop(columns=self.excluded_features, errors='ignore').copy()
        data = self._handle_missing_values(data)
        encoded_data = self._encode_categorical(data)

        encoded_data = encoded_data.reindex(columns=self.fitted_columns)
        X = encoded_data.values.astype(float)
        X_std = self.standard_scaler.transform(X)
        X_scaled = self.minmax_scaler.transform(X_std)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)

        return pd.DataFrame(X_scaled, columns=encoded_data.columns)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save_state(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, 'standard_scaler.pkl'), 'wb') as f:
            pickle.dump(self.standard_scaler, f)
        with open(os.path.join(directory, 'minmax_scaler.pkl'), 'wb') as f:
            pickle.dump(self.minmax_scaler, f)
        with open(os.path.join(directory, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(os.path.join(directory, 'columns.pkl'), 'wb') as f:
            pickle.dump({
                'categorical_cols': self.categorical_cols,
                'numeric_cols': self.numeric_cols,
                'fitted_columns': getattr(self, 'fitted_columns', [])
            }, f)

    def load_state(self, directory):
        with open(os.path.join(directory, 'standard_scaler.pkl'), 'rb') as f:
            self.standard_scaler = pickle.load(f)
        with open(os.path.join(directory, 'minmax_scaler.pkl'), 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        with open(os.path.join(directory, 'label_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(os.path.join(directory, 'columns.pkl'), 'rb') as f:
            cols = pickle.load(f)
            self.categorical_cols = cols['categorical_cols']
            self.numeric_cols = cols['numeric_cols']
            self.fitted_columns = cols.get('fitted_columns', [])
        self.fitted = True

##############################
# Train Preprocessor Function
##############################

def train_preprocessor(training_file=None, save_dir='models/preprocessing', normalization_method='minmax'):
    """
    Train a DataPreprocessor on a CSV file. If training_file is not provided,
    it attempts to find the latest CSV in captured_data/training.
    """
    if training_file:
        if not os.path.exists(training_file):
            raise FileNotFoundError(f"Training file not found: {training_file}")
        print(f"Loading training data from: {training_file}")
        df = pd.read_csv(training_file)
    else:
        training_dir = 'captured_data/training'
        capture_files = glob.glob(os.path.join(training_dir, '*.csv'))
        if not capture_files:
            raise FileNotFoundError(f"No capture files found in {training_dir}")
        latest_capture = max(capture_files, key=os.path.getctime)
        print(f"Loading training data from: {latest_capture}")
        df = pd.read_csv(latest_capture)
    
    print(f"Loaded {len(df)} packets for training.")
    
    print(f"Fitting DataPreprocessor with {normalization_method} normalization...")
    preprocessor = DataPreprocessor(normalization_method=normalization_method)
    preprocessor.fit(df)
    
    print(f"Saving preprocessor state to: {save_dir}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    preprocessor.save_state(save_dir)
    print("Preprocessor training complete!\n")


##############################
# Transform Function
##############################

def transform_data(input_file, output_dir, preprocessor_state, normalization_method='minmax'):
    """
    Transform a CSV file using a previously trained DataPreprocessor.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples")

    print("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor(normalization_method=normalization_method)
    
    print(f"Loading preprocessor state from {preprocessor_state}")
    preprocessor.load_state(preprocessor_state)

    print("Preprocessing data...")
    preprocessed_df = preprocessor.transform(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'preprocessed_data.csv'
    print(f"Saving preprocessed data to {output_file}")
    preprocessed_df.to_csv(output_file, index=False)
    print("Data transformation complete!\n")


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