#!/usr/bin/env python3

"""
initial_data_capture.py
-----------------------

This script provides a capturing mechanism for ICS traffic on a specified interface.
It uses Scapy to sniff packets, applies a sliding window approach for feature engineering,
and writes the engineered features to a CSV file. It prepares a csv to the point of training the LSTM model.

Key Functions and Workflow:
    1. offline_capture(interface, output_csv, window_size, load_state):
       - Uses Scapy to sniff packets from a given interface.
       - For each packet:
         * Extract minimal raw fields (timestamp, IP, ports, etc.).
         * Add to a sliding window in the SimpleFeatureEngineeringPreprocessor (imported).
         * When the window is full, it computes engineered features, label-encodes, scales, etc.
         * Writes the resulting row to the CSV.

    2. _extract_packet_info(packet):
       - Extracts raw details (IP, MAC, etc.) from the scapy packet to feed into the preprocessor.

How to Run:
    - python3 initial_data_capture.py --interface <your_interface> --output-csv <output_path.csv> --window-size 20 --load-state <path_to_preprocessor_state>
    Example:
      python3 initial_data_capture.py --interface eth0 --output-csv captured_data.csv --window-size 20

Dependencies:
    - scapy for packet sniffing
    - SimpleFeatureEngineeringPreprocessor from 'simple_preprocessor.py' to handle sliding window features.
    - Runs offline capture (no detection). For detection, see 'simple_pipeline.py'.

Typical Usage in the Overall IDS:
    - If you want to generate a CSV dataset before training or testing a model, you can use this script to do a capture.
    - The CSV output can be further used by the pre-processing or training scripts later on.

See Also:
    - 'simple_preprocessor.py' for the actual feature engineering logic.
    - 'pre_processing.py' if you want a separate approach to handle training the scaler/encoders.
    - 'train_lstm_autoencoder.py' to train your LSTM model after generating data.
"""

import os
import sys
import csv
import logging
import signal
from datetime import datetime

import scapy.all as scapy
from simple_preprocessor import SimpleFeatureEngineeringPreprocessor

def offline_capture(interface="eth0", output_csv=None, window_size=20, load_state=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("initial_data_capture")

    if not output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"training_capture_{timestamp}.csv"
    logger.info(f"Writing captured data to: {output_csv}")

    preprocessor = SimpleFeatureEngineeringPreprocessor(load_state=load_state, window_size=window_size)

    csv_file = open(output_csv, 'w', newline='')
    writer = None
    fieldnames = None

    def packet_callback(packet):
        try:
            pkt_info = _extract_packet_info(packet)
            processed_dict = preprocessor.add_packet(pkt_info)

            # Skip writing if window is not yet full
            if len(preprocessor.packet_window) < window_size:
                return

            nonlocal writer, fieldnames
            if writer is None:
                fieldnames = preprocessor.all_features
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()

            writer.writerow(processed_dict)
            csv_file.flush()

        except Exception as ex:
            logger.error(f"Error processing packet: {ex}", exc_info=True)

    def _stop_handler(sig, frame):
        logger.info("Capture stopped by user.")
        csv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop_handler)
    logger.info(f"Starting offline capture on interface={interface} with window_size={window_size} ...")
    scapy.sniff(iface=interface, store=False, prn=packet_callback)

def _extract_packet_info(packet):
    info = {
        "timestamp": datetime.now().isoformat(),
        "src_ip": "",
        "dst_ip": "",
        "src_mac": "",
        "dst_mac": "",
        "protocol": "0",
        "src_port": 0,
        "dst_port": 0,
        "packet_size": len(packet),
        "tcp_flags": 0,
        "is_modbus": 0,
        "modbus_function_code": 0,
        "modbus_length": 0,
        "error_code": 0,
        "packet_type": "Unknown"
    }

    if scapy.Ether in packet:
        info["src_mac"] = packet[scapy.Ether].src
        info["dst_mac"] = packet[scapy.Ether].dst
    if scapy.IP in packet:
        info["packet_type"] = "IPv4"
        info["src_ip"] = packet[scapy.IP].src
        info["dst_ip"] = packet[scapy.IP].dst
        info["protocol"] = str(packet[scapy.IP].proto)
        if scapy.TCP in packet:
            info["src_port"] = int(packet[scapy.TCP].sport)
            info["dst_port"] = int(packet[scapy.TCP].dport)
            info["tcp_flags"] = packet[scapy.TCP].flags.value
        elif scapy.UDP in packet:
            info["src_port"] = int(packet[scapy.UDP].sport)
            info["dst_port"] = int(packet[scapy.UDP].dport)
    elif scapy.IPv6 in packet:
        info["packet_type"] = "IPv6"
        info["src_ip"] = packet[scapy.IPv6].src
        info["dst_ip"] = packet[scapy.IPv6].dst
        info["protocol"] = str(packet[scapy.IPv6].nh)
        if scapy.TCP in packet:
            info["src_port"] = int(packet[scapy.TCP].sport)
            info["dst_port"] = int(packet[scapy.TCP].dport)
            info["tcp_flags"] = packet[scapy.TCP].flags.value
        elif scapy.UDP in packet:
            info["src_port"] = int(packet[scapy.UDP].sport)
            info["dst_port"] = int(packet[scapy.UDP].dport)
    elif scapy.ARP in packet:
        info["packet_type"] = "ARP"
        info["protocol"] = "ARP"
        info["src_ip"] = packet[scapy.ARP].psrc
        info["dst_ip"] = packet[scapy.ARP].pdst

    if (info["src_port"] == 502 or info["dst_port"] == 502):
        info["is_modbus"] = 1

    return info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline ICS capture: Feature engineering only, no detection.")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface to sniff on.")
    parser.add_argument("--output-csv", type=str, help="Path to output CSV with processed features.")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size for feature engineering.")
    parser.add_argument("--load-state", type=str, default=None, help="Path to pickled state for scaler & label encoders.")
    args = parser.parse_args()

    offline_capture(
        interface=args.interface,
        output_csv=args.output_csv,
        window_size=args.window_size,
        load_state=args.load_state
    )