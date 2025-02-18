import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""
simple_preprocessor.py
----------------------

This script defines the 'SimpleFeatureEngineeringPreprocessor', which:
    - Maintains a sliding window of the last N packets
    - Computes aggregated/engineered features (e.g. inter-arrival times, rates, stats)
    - Encodes IP/MAC/protocol using label encoders
    - Optionally applies scaling (MinMax or Standard) to numeric features
    - Provides 'add_packet' for incremental usage, and 'transform_packet_list' for batch usage

Key usage:
    1) The detection pipeline (simple_pipeline.py) uses add_packet or transform_packet_list
       to produce a numeric row for LSTM.
    2) This class can be saved/loaded so label encoders and scalers remain consistent.

Important Functions:
    - add_packet(packet_dict):
      * Adds the new packet to the sliding window
      * If the window is not full, engineered features (like interarrival times) are zero
      * Once the window is full, aggregates the entire window to compute stats
      * Encodes, scales, returns a single row of data for that packet

    - transform_packet_list(list_of_dicts):
      * For offline usage (like building a big dataset from a pcap)
      * Processes each packet in sequence, calling add_packet internally

    - save_state() / load_state():
      * Saves/loads label encoders, minmax_scaler, standard_scaler, etc.

How it Connects:
    - 'initial_data_capture.py' uses it to generate CSV with engineered features offline
    - 'simple_pipeline.py' uses it in real-time detection
    - 'pre_processing.py' is a separate script for train vs. transform mode, but this approach
      can handle the same tasks.

To Run:
    - Typically not run directly, but imported into other scripts.
      You can do: from simple_preprocessor import SimpleFeatureEngineeringPreprocessor
      Then instantiate and call add_packet() or transform_packet_list() as needed.
"""

# Optional environment-based logic if needed
try:
    from config import get_whitelist
except ImportError:
    # Provide a default fallback
    def get_whitelist(env=None):
        return set()

class SimpleFeatureEngineeringPreprocessor:
    """
    Merged class that:
      1) Maintains a sliding window of packets for feature engineering
      2) Computes raw + engineered features
      3) Label-encodes IP/MAC/protocol
      4) Scales numeric features via either MinMax or Standard Scaler
      5) Saves/loads state from multiple .pkl files (columns, label_encoders, minmax_scaler, standard_scaler)
      6) Exposes methods for both "online" (add_packet) and "offline" (transform_packet_list)
    """

    def __init__(self, load_state=None, window_size=20, environment=None):
        """
        :param load_state: If provided, path to a directory with columns.pkl, label_encoders.pkl,
                           minmax_scaler.pkl, standard_scaler.pkl
        :param window_size: Number of packets in the sliding window for feature engineering
        :param environment: Optional environment string for whitelisting logic
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # -- Sliding Window / Feature Eng. Config --
        self.window_size = window_size
        self.environment = environment
        self.packet_window = []
        # If environment-based whitelisting is needed
        self.whitelist_ip_mac_pairs = get_whitelist(environment) if callable(get_whitelist) else set()

        # -- Label Encoding Storage --
        # We store incremental counters so new IPs or MACs get new IDs
        self.label_encoders = {
            'src_ip': {},
            'dst_ip': {},
            'src_mac': {},
            'dst_mac': {},
            'protocol': {},
            'packet_type': {}
        }
        self.ip_counter = 1
        self.mac_counter = 1
        self.protocol_counter = 1

        # -- Numeric Feature Scaling --
        self.minmax_scaler = None
        self.standard_scaler = None

        # All numeric engineered features
        self.engineered_numeric_features = [
            "mean_interarrival_time",
            "std_interarrival_time",
            "min_interarrival_time",
            "max_interarrival_time",
            "median_interarrival_time",
            "interarrival_time_entropy",
            "interarrival_time_skew",
            "interarrival_time_kurtosis",

            "mean_packet_size",
            "std_packet_size",
            "min_packet_size",
            "max_packet_size",
            "median_packet_size",
            "packet_size_entropy",
            "packet_size_skew",
            "packet_size_kurtosis",

            "tcp_syn_rate",
            "tcp_fin_rate",
            "tcp_rst_rate",
            "tcp_psh_rate",
            "tcp_ack_rate",
            "tcp_urg_rate",

            "unique_src_ips",
            "unique_src_macs",
            "arp_anomaly_score",
            "error_rate",
            "modbus_rate",
            "function_code_entropy"
        ]

        # The "raw numeric" columns
        self.raw_numeric_features = [
            "packet_size",
            "modbus_length",
            "src_port",
            "dst_port",
            "tcp_flags",
            "is_modbus",
            "modbus_function_code",
            "error_code"
        ]

        # Summarize numeric columns
        self.numeric_features = self.raw_numeric_features + self.engineered_numeric_features

        # We consider these raw fields as well
        self.raw_categorical_fields = [
            "src_ip", "dst_ip", "src_mac", "dst_mac", "protocol", "packet_type"
        ]

        # The core set of features = raw + engineered
        self.all_features = [
            "src_ip",
            "dst_ip",
            "src_mac",
            "dst_mac",
            "protocol",
            "src_port",
            "dst_port",
            "packet_size",
            "tcp_flags",
            "is_modbus",
            "modbus_function_code",
            "modbus_length",
            "error_code",
            "packet_type"
        ] + self.engineered_numeric_features

        # If we have a saved state directory, load it
        if load_state and os.path.isdir(load_state):
            self.load_state(load_state)

    # --------------------------------------------------------------------------
    #  1) Packet-level addition => sliding window, feature engineering, encoding
    # --------------------------------------------------------------------------
    def add_packet(self, packet_dict):
        """
        For live usage: add an individual packet dict,
        engineer features using the sliding window approach,
        encode & scale, and return the final processed row as a dict.
        NOTE: If the window isn't yet 'full', certain engineered features remain 0.
        """
        # Add to window
        self.packet_window.append(packet_dict)
        if len(self.packet_window) > self.window_size:
            self.packet_window.pop(0)

        # Compute engineered features for the entire window
        feat_dict = self._extract_window_features()

        # Merge raw + engineered
        combined = {**packet_dict, **feat_dict}

        # Convert to DataFrame
        df = pd.DataFrame([combined])

        # Debug before ensuring features
        self.logger.info(f"DataFrame columns before ensuring features: {list(df.columns)}")
        self.logger.info(f"Number of features before ensuring: {len(df.columns)}")

        # Ensure all features exist with default values
        for col in self.all_features:
            if col not in df.columns:
                if col in self.numeric_features:
                    df[col] = 0.0
                else:
                    df[col] = ''

        # Debug after ensuring features
        self.logger.info(f"DataFrame columns after ensuring features: {list(df.columns)}")
        self.logger.info(f"Number of features after ensuring: {len(df.columns)}")

        # Ensure columns are in the correct order
        df = df[self.all_features]

        # Debug after ordering
        self.logger.info(f"DataFrame columns after ordering: {list(df.columns)}")
        self.logger.info(f"Number of features after ordering: {len(df.columns)}")

        # Process and return
        processed_df = self._encode_and_scale(df)
        row_dict = processed_df.to_dict(orient='records')[0]
        return row_dict

    def _extract_window_features(self):
        """
        Produce aggregated features for the entire window.
        If the window is not full, fill 0.
        """
        if len(self.packet_window) < self.window_size:
            return self._zero_engineered_features()
        return self._compute_engineered_features()

    def _zero_engineered_features(self):
        out = {}
        for f in self.engineered_numeric_features:
            out[f] = 0.0
        return out

    def _compute_engineered_features(self):
        """
        Compute all engineered features for the current window.
        """
        self.logger.info("Computing engineered features...")
        
        interarrival_stats = self._calculate_interarrival_stats(self.packet_window)
        self.logger.info(f"Interarrival features: {list(interarrival_stats.keys())}")
        
        packet_size_stats = self._calculate_packet_size_stats(self.packet_window)
        self.logger.info(f"Packet size features: {list(packet_size_stats.keys())}")
        
        tcp_stats = self._calculate_tcp_flags_stats(self.packet_window)
        self.logger.info(f"TCP features: {list(tcp_stats.keys())}")
        
        unique_addr_stats = self._calculate_unique_addr_stats(self.packet_window)
        self.logger.info(f"Address features: {list(unique_addr_stats.keys())}")
        
        arp_stats = self._calculate_arp_anomaly(self.packet_window)
        self.logger.info(f"ARP features: {list(arp_stats.keys())}")
        
        error_stats = self._calculate_error_stats(self.packet_window)
        self.logger.info(f"Error features: {list(error_stats.keys())}")
        
        modbus_stats = self._calculate_modbus_stats(self.packet_window)
        self.logger.info(f"Modbus features: {list(modbus_stats.keys())}")

        final_stats = {}
        final_stats.update(interarrival_stats)
        final_stats.update(packet_size_stats)
        final_stats.update(tcp_stats)
        final_stats.update(unique_addr_stats)
        final_stats.update(arp_stats)
        final_stats.update(error_stats)
        final_stats.update(modbus_stats)

        self.logger.info(f"Total engineered features: {len(final_stats)}")
        self.logger.info(f"All engineered feature names: {list(final_stats.keys())}")

        # Verify all engineered features are present
        missing_features = [f for f in self.engineered_numeric_features if f not in final_stats]
        if missing_features:
            self.logger.warning(f"Missing engineered features: {missing_features}")

        return final_stats

    # --------------------------------------------------------------------------
    #  2) Offline usage => transform entire list of packet dicts
    # --------------------------------------------------------------------------
    def transform_packet_list(self, packets):
        """
        For offline usage: pass a list of packet dictionaries to produce
        a final 2D numeric array of shape (N, num_features).
        We do the sliding window approach, so the first window_size - 1 entries won't have full features.
        """
        self.packet_window = []
        processed_rows = []
        for pkt in packets:
            row_dict = self.add_packet(pkt)
            processed_rows.append(row_dict)

        df = pd.DataFrame(processed_rows)

        # Ensure all features exist with default values
        for col in self.all_features:
            if col not in df.columns:
                if col in self.numeric_features:
                    df[col] = 0.0
                else:
                    df[col] = ''

        # Ensure columns are in the correct order
        df = df[self.all_features]
        return df.values.astype(np.float32)

    # --------------------------------------------------------------------------
    #  3) Label Encoding + Numeric Scaling
    # --------------------------------------------------------------------------
    def _encode_and_scale(self, df):
        # First encode categorical columns
        df_encoded = self._encode_categorical_columns(df.copy())

        # Ensure all numeric features exist and are float64
        for col in self.numeric_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').astype('float64').fillna(0.0)

        # Debug logging
        self.logger.info(f"Numeric features expected: {len(self.numeric_features)}")
        self.logger.info(f"Numeric features actual: {len([col for col in self.numeric_features if col in df_encoded.columns])}")
        self.logger.info(f"Missing numeric features: {[col for col in self.numeric_features if col not in df_encoded.columns]}")
        self.logger.info(f"All features expected: {len(self.all_features)}")
        self.logger.info(f"All features actual: {len(df_encoded.columns)}")
        self.logger.info(f"Missing features: {[col for col in self.all_features if col not in df_encoded.columns]}")

        # If we have a minmax_scaler, apply it to all numeric features including encoded categorical ones
        if self.minmax_scaler is not None:
            all_features = self.numeric_features + self.raw_categorical_fields
            
            # Convert all columns that will be scaled to float64
            for col in all_features:
                df_encoded[col] = df_encoded[col].astype('float64')
            
            numeric_df = df_encoded[all_features].fillna(0.0)
            self.logger.info(f"Shape of numeric_part before scaling: {numeric_df.values.shape}")
            scaled_vals = self.minmax_scaler.transform(numeric_df.values)
            
            # Create a new DataFrame with scaled values and float64 dtype
            scaled_df = pd.DataFrame(scaled_vals, columns=all_features, dtype='float64', index=df_encoded.index)
            
            # Update the original DataFrame with scaled values
            df_encoded[all_features] = scaled_df

        return df_encoded

    def _encode_categorical_columns(self, df):
        # src_ip
        df['src_ip'] = df['src_ip'].astype(str).apply(self._encode_ip, args=('src_ip',))
        df['dst_ip'] = df['dst_ip'].astype(str).apply(self._encode_ip, args=('dst_ip',))
        df['src_mac'] = df['src_mac'].astype(str).apply(self._encode_mac, args=('src_mac',))
        df['dst_mac'] = df['dst_mac'].astype(str).apply(self._encode_mac, args=('dst_mac',))
        df['protocol'] = df['protocol'].astype(str).apply(self._encode_proto)
        
        # Add packet_type encoding
        if 'packet_type' not in self.label_encoders:
            self.label_encoders['packet_type'] = {}
        df['packet_type'] = df['packet_type'].astype(str).apply(lambda x: self._encode_packet_type(x))

        # Convert integer columns
        for col in ["src_port","dst_port","tcp_flags","is_modbus","error_code","modbus_function_code"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df

    def _encode_packet_type(self, pkt_type):
        col = 'packet_type'
        if pkt_type in self.label_encoders[col]:
            return self.label_encoders[col][pkt_type]
        else:
            self.label_encoders[col][pkt_type] = len(self.label_encoders[col]) + 1
            return self.label_encoders[col][pkt_type]

    def _encode_ip(self, ip_str, col):
        if ip_str in self.label_encoders[col]:
            return self.label_encoders[col][ip_str]
        else:
            self.label_encoders[col][ip_str] = self.ip_counter
            self.ip_counter += 1
            return self.label_encoders[col][ip_str]

    def _encode_mac(self, mac_str, col):
        if mac_str in self.label_encoders[col]:
            return self.label_encoders[col][mac_str]
        else:
            self.label_encoders[col][mac_str] = self.mac_counter
            self.mac_counter += 1
            return self.label_encoders[col][mac_str]

    def _encode_proto(self, proto_str):
        col = 'protocol'
        if proto_str in self.label_encoders[col]:
            return self.label_encoders[col][proto_str]
        else:
            self.label_encoders[col][proto_str] = self.protocol_counter
            self.protocol_counter += 1
            return self.label_encoders[col][proto_str]

    # --------------------------------------------------------------------------
    #  4) Saving / Loading State from multiple pkl files
    # --------------------------------------------------------------------------
    def fit_minmax_scaler(self, df):
        # First encode the categorical columns
        df_encoded = self._encode_categorical_columns(df.copy())
        # Now all columns are numeric and should be scaled
        all_features = self.numeric_features + self.raw_categorical_fields
        numeric_df = df_encoded[all_features].fillna(0.0)
        self.minmax_scaler = MinMaxScaler(feature_range=(0,1))
        self.minmax_scaler.fit(numeric_df.values)

    def save_state(self, directory):
        """
        Save the entire encoding + scaling state to separate pickle files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # label_encoders
        with open(os.path.join(directory, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'ip_counter': self.ip_counter,
                'mac_counter': self.mac_counter,
                'protocol_counter': self.protocol_counter
            }, f)

        # minmax_scaler
        if self.minmax_scaler is not None:
            with open(os.path.join(directory, 'minmax_scaler.pkl'), 'wb') as f:
                pickle.dump(self.minmax_scaler, f)

        # standard_scaler (if used)
        if self.standard_scaler is not None:
            with open(os.path.join(directory, 'standard_scaler.pkl'), 'wb') as f:
                pickle.dump(self.standard_scaler, f)

        # columns
        with open(os.path.join(directory, 'columns.pkl'), 'wb') as f:
            pickle.dump({
                'numeric_features': self.numeric_features,
                'engineered_numeric_features': self.engineered_numeric_features,
                'raw_numeric_features': self.raw_numeric_features,
                'raw_categorical_fields': self.raw_categorical_fields,
                'all_features': self.all_features
            }, f)

    def load_state(self, directory):
        """
        Load columns, label_encoders, minmax_scaler, standard_scaler.
        """
        # columns
        columns_path = os.path.join(directory, 'columns.pkl')
        if os.path.isfile(columns_path):
            with open(columns_path, 'rb') as f:
                data = pickle.load(f)
                self.numeric_features = data.get('numeric_features', self.numeric_features)
                self.engineered_numeric_features = data.get('engineered_numeric_features', self.engineered_numeric_features)
                self.raw_numeric_features = data.get('raw_numeric_features', self.raw_numeric_features)
                self.raw_categorical_fields = data.get('raw_categorical_fields', self.raw_categorical_fields)
                self.all_features = data.get('all_features', self.all_features)

        # label_encoders
        encoders_path = os.path.join(directory, 'label_encoders.pkl')
        if os.path.isfile(encoders_path):
            with open(encoders_path, 'rb') as f:
                data = pickle.load(f)
                self.label_encoders = data.get('label_encoders', self.label_encoders)
                self.ip_counter = data.get('ip_counter', 1)
                self.mac_counter = data.get('mac_counter', 1)
                self.protocol_counter = data.get('protocol_counter', 1)

        # minmax_scaler
        minmax_path = os.path.join(directory, 'minmax_scaler.pkl')
        if os.path.isfile(minmax_path):
            with open(minmax_path, 'rb') as f:
                self.minmax_scaler = pickle.load(f)

        # standard_scaler
        std_path = os.path.join(directory, 'standard_scaler.pkl')
        if os.path.isfile(std_path):
            with open(std_path, 'rb') as f:
                self.standard_scaler = pickle.load(f)

    # --------------------------------------------------------------------------
    #  Interarrival, Packet size, etc. helper methods
    # --------------------------------------------------------------------------
    def _calculate_interarrival_stats(self, window):
        if len(window) < 2:
            return { f:0.0 for f in [
                "mean_interarrival_time","std_interarrival_time",
                "min_interarrival_time","max_interarrival_time",
                "median_interarrival_time","interarrival_time_entropy",
                "interarrival_time_skew","interarrival_time_kurtosis"
            ]}
        timestamps = []
        for p in window:
            ts_str = p.get('timestamp','')
            if not ts_str:
                continue
            try:
                dt = datetime.fromisoformat(ts_str)
                timestamps.append(dt.timestamp())
            except:
                try:
                    timestamps.append(float(ts_str))
                except:
                    pass
        if len(timestamps) < 2:
            return { f:0.0 for f in [
                "mean_interarrival_time","std_interarrival_time",
                "min_interarrival_time","max_interarrival_time",
                "median_interarrival_time","interarrival_time_entropy",
                "interarrival_time_skew","interarrival_time_kurtosis"
            ]}
        arr_times = np.diff(timestamps)
        arr_times = arr_times[arr_times >= 0]
        if len(arr_times) < 1:
            return { f:0.0 for f in [
                "mean_interarrival_time","std_interarrival_time",
                "min_interarrival_time","max_interarrival_time",
                "median_interarrival_time","interarrival_time_entropy",
                "interarrival_time_skew","interarrival_time_kurtosis"
            ]}
        mean_ia = float(np.mean(arr_times))
        std_ia = float(np.std(arr_times))
        min_ia = float(np.min(arr_times))
        max_ia = float(np.max(arr_times))
        med_ia = float(np.median(arr_times))
        hist_vals, _ = np.histogram(arr_times, bins=10)
        ent_ia = float(entropy(hist_vals)) if hist_vals.sum() > 0 else 0.0
        if std_ia < 1e-9:
            skew_ia = 0.0
            kurt_ia = 0.0
        else:
            try:
                skew_ia = float(skew(arr_times))
                kurt_ia = float(kurtosis(arr_times))
                if not np.isfinite(skew_ia): skew_ia = 0.0
                if not np.isfinite(kurt_ia): kurt_ia = 0.0
            except:
                skew_ia = 0.0
                kurt_ia = 0.0
        return {
            "mean_interarrival_time": mean_ia,
            "std_interarrival_time": std_ia,
            "min_interarrival_time": min_ia,
            "max_interarrival_time": max_ia,
            "median_interarrival_time": med_ia,
            "interarrival_time_entropy": ent_ia,
            "interarrival_time_skew": skew_ia,
            "interarrival_time_kurtosis": kurt_ia
        }

    def _calculate_packet_size_stats(self, window):
        sizes = [p.get('packet_size',0) for p in window]
        if not sizes:
            return { f:0.0 for f in [
                "mean_packet_size","std_packet_size","min_packet_size","max_packet_size",
                "median_packet_size","packet_size_entropy","packet_size_skew","packet_size_kurtosis"
            ]}
        arr = np.array(sizes, dtype=float)
        mean_sz = float(np.mean(arr))
        std_sz = float(np.std(arr))
        min_sz = float(np.min(arr))
        max_sz = float(np.max(arr))
        med_sz = float(np.median(arr))
        hist_vals, _ = np.histogram(arr, bins=10)
        ent_sz = float(entropy(hist_vals)) if hist_vals.sum() > 0 else 0.0
        if std_sz < 1e-9:
            skew_sz = 0.0
            kurt_sz = 0.0
        else:
            try:
                skew_sz = float(skew(arr))
                kurt_sz = float(kurtosis(arr))
                if not np.isfinite(skew_sz): skew_sz = 0.0
                if not np.isfinite(kurt_sz): kurt_sz = 0.0
            except:
                skew_sz = 0.0
                kurt_sz = 0.0
        return {
            "mean_packet_size": mean_sz,
            "std_packet_size": std_sz,
            "min_packet_size": min_sz,
            "max_packet_size": max_sz,
            "median_packet_size": med_sz,
            "packet_size_entropy": ent_sz,
            "packet_size_skew": skew_sz,
            "packet_size_kurtosis": kurt_sz
        }

    def _calculate_tcp_flags_stats(self, window):
        n = len(window)
        if n == 0:
            return {f:0.0 for f in [
                "tcp_syn_rate","tcp_fin_rate","tcp_rst_rate","tcp_psh_rate","tcp_ack_rate","tcp_urg_rate"
            ]}
        from collections import defaultdict
        tcp_flags_counter = defaultdict(int)
        for p in window:
            flags = p.get('tcp_flags',0)
            if flags > 0:
                bits = format(flags, '08b')
                # last bit => FIN, second last => SYN, etc.
                tcp_flags_counter['F'] += int(bits[-1])
                tcp_flags_counter['S'] += int(bits[-2])
                tcp_flags_counter['R'] += int(bits[-3])
                tcp_flags_counter['P'] += int(bits[-4])
                tcp_flags_counter['A'] += int(bits[-5])
                tcp_flags_counter['U'] += int(bits[-6])
        return {
            "tcp_syn_rate": tcp_flags_counter['S']/n,
            "tcp_fin_rate": tcp_flags_counter['F']/n,
            "tcp_rst_rate": tcp_flags_counter['R']/n,
            "tcp_psh_rate": tcp_flags_counter['P']/n,
            "tcp_ack_rate": tcp_flags_counter['A']/n,
            "tcp_urg_rate": tcp_flags_counter['U']/n
        }

    def _calculate_unique_addr_stats(self, window):
        unique_ips = set()
        unique_macs = set()
        for p in window:
            sip = p.get('src_ip','')
            smac = p.get('src_mac','')
            if sip:
                unique_ips.add(sip)
            if smac:
                unique_macs.add(smac)
        return {
            "unique_src_ips": len(unique_ips),
            "unique_src_macs": len(unique_macs)
        }

    def _calculate_arp_anomaly(self, window):
        current_pairs = set()
        for p in window:
            sip = p.get('src_ip','')
            smac= p.get('src_mac','')
            if sip and smac:
                current_pairs.add((sip, smac))
        if not current_pairs:
            return {"arp_anomaly_score": 0.0}
        suspicious = [pair for pair in current_pairs if pair not in self.whitelist_ip_mac_pairs]
        anomaly_score = len(suspicious)/len(current_pairs)
        return {"arp_anomaly_score": anomaly_score}

    def _calculate_error_stats(self, window):
        if not window:
            return {"error_rate": 0.0}
        err_count = sum(p.get('error_code',0) for p in window)
        return {"error_rate": err_count/len(window) if len(window) else 0.0}

    def _calculate_modbus_stats(self, window):
        if not window:
            return {"modbus_rate":0.0,"function_code_entropy":0.0}
        modbus_pkts = [p for p in window if p.get('is_modbus',0)==1]
        if not modbus_pkts:
            return {"modbus_rate":0.0,"function_code_entropy":0.0}
        modbus_rate = len(modbus_pkts)/len(window)
        fcodes = [p.get('modbus_function_code',0) for p in modbus_pkts]
        from collections import Counter
        code_counts = Counter(fcodes)
        total = sum(code_counts.values())
        if total<1:
            f_entropy = 0.0
        else:
            from scipy.stats import entropy
            probs = [c/total for c in code_counts.values()]
            f_entropy = float(entropy(probs))
        return {
            "modbus_rate": modbus_rate,
            "function_code_entropy": f_entropy
        }

    def _extract_packet_info(self, packet):
        """Extract features from a single packet."""
        # Basic packet info
        pkt_dict = {
            'src_ip': packet.get('IP', {}).get('src', ''),
            'dst_ip': packet.get('IP', {}).get('dst', ''),
            'src_mac': packet.get('Ethernet', {}).get('src', ''),
            'dst_mac': packet.get('Ethernet', {}).get('dst', ''),
            'protocol': packet.get('highest_layer', ''),
            'src_port': packet.get('TCP', {}).get('sport', packet.get('UDP', {}).get('sport', 0)),
            'dst_port': packet.get('TCP', {}).get('dport', packet.get('UDP', {}).get('dport', 0)),
            'packet_size': len(packet),
            'tcp_flags': packet.get('TCP', {}).get('flags', 0),
            'is_modbus': 1 if packet.get('TCP', {}).get('dport', 0) == 502 else 0,
            'modbus_function_code': 0,  # Default
            'modbus_length': 0,         # Default
            'error_code': 0,            # Default
            'packet_type': packet.get('highest_layer', '')  # Default to highest layer
        }

        # Debug logging
        self.logger.info(f"Extracted packet features: {list(pkt_dict.keys())}")
        self.logger.info(f"Raw packet layers: {packet.layers()}")

        # Modbus-specific features
        if pkt_dict['is_modbus'] and 'Raw' in packet:
            try:
                raw_data = bytes(packet['Raw'])
                if len(raw_data) >= 8:  # Minimum Modbus/TCP length
                    pkt_dict['modbus_function_code'] = raw_data[7]
                    pkt_dict['modbus_length'] = int.from_bytes(raw_data[4:6], byteorder='big')
                    if raw_data[7] >= 0x80:  # Error response
                        pkt_dict['error_code'] = raw_data[8] if len(raw_data) > 8 else 0
            except Exception as e:
                self.logger.warning(f"Error parsing Modbus data: {e}")

        return pkt_dict