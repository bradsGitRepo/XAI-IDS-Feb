import os
import csv
import sys
import time
import signal
import logging
import threading
from queue import Queue
from datetime import datetime

import scapy.all as scapy
import pandas as pd

from simple_preprocessor import SimpleFeatureEngineeringPreprocessor
from simple_model import SimpleLSTMAutoencoder

"""
simple_pipeline.py
------------------

This script implements the *live* detection pipeline for ICS intrusion detection:
    1) capture_packets: uses Scapy's AsyncSniffer to continuously read packets from a network interface,
       attaching a timestamp to each packet, placing it on a queue.

    2) detection_worker:
       - Pulls packets from packet_queue into a ring buffer of WINDOW_SIZE.
       - Once full, calls transform_packet_list to produce the final row with full engineered features.
       - Uses SimpleLSTMAutoencoder's detect_anomaly to see if it's malicious.
       - Writes all data to 'live_capture.csv'.
       - If anomaly => sends the window + final row to an IG worker via ig_queue.

    3) ig_worker:
       - Receives anomalies from the detection worker.
       - Calls compute_integrated_gradients to figure out feature-level attributions.
       - Writes the expanded row + IG results to 'anomalies_ig.csv'.

Usage Example:
    python3 simple_pipeline.py

Steps:
    1) main() spawns:
       - capture_packets thread
       - detection_worker thread
       - ig_worker thread
    2) Stop by pressing CTRL+C. The threads shut down gracefully.

Dependencies:
    - scapy for packet sniffing
    - 'simple_preprocessor.py' for real-time feature engineering
    - 'simple_model.py' for LSTM autoencoder detection & integrated gradients
    - The user must have a trained model and preprocessor state to be loaded from (PREPROCESSOR_DIR, MODEL_WEIGHTS).

Connections:
    - Typically, you'd have a model trained by 'train_lstm_autoencoder.py'
      and a preprocessor state from 'pre_processing.py' or 'simple_preprocessor.py'.
    - This pipeline uses those states to do real-time detection.

How to Run:
    - Make sure you have a valid interface (e.g. en0 on macOS, eth0 on Linux).
    - Ensure "models/test_model/best_model.keras" or other path is correct.
    - Then run: python3 simple_pipeline.py
    - Watch "live_capture.csv" for all traffic, and "anomalies_ig.csv" for anomalies w/ IG.
"""

INTERFACE = "en0"
WINDOW_SIZE = 20
NUM_FEATURES = 42
THRESHOLD = 0.5
PREPROCESSOR_DIR = "models/preprocessing"
MODEL_WEIGHTS = "models/lstm_autoencoder/model.keras"
OUTPUT_DIR = "live_detection"

packet_queue = Queue(maxsize=0)
ig_queue = Queue(maxsize=0)

stop_event = threading.Event()

def capture_packets(interface=INTERFACE):
    logging.info("Starting async packet capture on interface=%s...", interface)
    def enqueue_pkt(pkt):
        pkt.time_captured = datetime.now().isoformat()
        packet_queue.put(pkt, block=False)
    sniffer = scapy.AsyncSniffer(iface=interface, store=False, prn=enqueue_pkt)
    sniffer.start()
    while not stop_event.is_set():
        time.sleep(1)
    sniffer.stop()
    logging.info("Packet capture thread stopping...")

def detection_worker():
    logging.info("Initializing detection worker: loading preprocessor + model.")
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    preprocessor = SimpleFeatureEngineeringPreprocessor(load_state=PREPROCESSOR_DIR, window_size=WINDOW_SIZE)
    model_wrapper = SimpleLSTMAutoencoder(seq_length=WINDOW_SIZE, num_features=NUM_FEATURES)
    model_wrapper.load_weights(MODEL_WEIGHTS)

    logging.info(f"Detection worker loaded model weights from {MODEL_WEIGHTS}, threshold={THRESHOLD}")

    fieldnames = preprocessor.all_features + ["timestamp", "is_anomaly", "anomaly_score"]
    live_csv = open(os.path.join(OUTPUT_DIR, "live_capture.csv"), 'w', newline='')
    writer_all = csv.DictWriter(live_csv, fieldnames=fieldnames)
    writer_all.writeheader()

    ring_buffer = []

    while not stop_event.is_set():
        try:
            pkt = packet_queue.get(timeout=1)
        except:
            continue

        pkt_info = _extract_packet_info(pkt)
        pkt_info["timestamp"] = getattr(pkt, "time_captured", datetime.now().isoformat())
        ring_buffer.append(pkt_info)
        if len(ring_buffer) > WINDOW_SIZE:
            ring_buffer.pop(0)

        if len(ring_buffer) < WINDOW_SIZE:
            single_df = preprocessor.transform_packet_list([pkt_info])
            row_dict = dict(single_df_to_dict(single_df, preprocessor.all_features)[0])
            row_dict["timestamp"] = pkt_info["timestamp"]
            row_dict["is_anomaly"] = 0
            row_dict["anomaly_score"] = 0.0
            writer_all.writerow(row_dict)
            live_csv.flush()
            continue

        array_2d = preprocessor.transform_packet_list(ring_buffer)
        final_row_array = array_2d[-1]
        final_row_dict = array_to_dict(final_row_array, preprocessor.all_features)
        final_row_dict["timestamp"] = pkt_info["timestamp"]

        seq_3d = array_2d.reshape(1, WINDOW_SIZE, NUM_FEATURES)
        is_anomaly, score = model_wrapper.detect_anomaly(seq_3d, threshold=THRESHOLD)
        final_row_dict["is_anomaly"] = int(is_anomaly)
        final_row_dict["anomaly_score"] = float(score)

        writer_all.writerow(final_row_dict)
        live_csv.flush()

        if is_anomaly:
            ig_data = {
                "raw_full_dict": final_row_dict,
                "array_2d": array_2d.copy(),
                "score": score
            }
            try:
                ig_queue.put(ig_data, block=True)
            except Exception as e:
                logging.error(f"Error queueing anomaly for IG processing: {e}")

    live_csv.close()
    logging.info("Detection worker thread stopping...")

def ig_worker():
    logging.info("IG worker started. Will compute integrated gradients for anomalies.")
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    preprocessor = SimpleFeatureEngineeringPreprocessor(load_state=PREPROCESSOR_DIR, window_size=WINDOW_SIZE)
    model_wrapper = SimpleLSTMAutoencoder(seq_length=WINDOW_SIZE, num_features=NUM_FEATURES)
    model_wrapper.load_weights(MODEL_WEIGHTS)

    fieldnames = (
        preprocessor.all_features
        + ["timestamp", "is_anomaly", "anomaly_score"]
        + ["ig_positive", "ig_negative", "top_5_features", "ig_explanation"]
    )
    ig_csv = open(os.path.join(OUTPUT_DIR, "anomalies_ig.csv"), 'w', newline='')
    writer_ig = csv.DictWriter(ig_csv, fieldnames=fieldnames)
    writer_ig.writeheader()

    while not stop_event.is_set():
        try:
            data = ig_queue.get(timeout=1)
        except:
            continue

        final_dict = dict(data["raw_full_dict"])
        array_2d = data["array_2d"]
        score = data["score"]

        seq_3d = array_2d.reshape(1, WINDOW_SIZE, NUM_FEATURES)
        ig_attribs = model_wrapper.compute_integrated_gradients(seq_3d)

        pos_sum = float(ig_attribs[ig_attribs > 0].sum())
        neg_sum = float(ig_attribs[ig_attribs < 0].sum())
        ig_summed = ig_attribs.sum(axis=0)
        ig_dict = {}
        for i, feat_name in enumerate(preprocessor.all_features):
            ig_dict[feat_name] = float(ig_summed[i])
        sorted_ig = sorted(ig_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_5 = sorted_ig[:5]
        top_5_str_parts = [f"{k}={ig_dict[k]:.3f}" for (k, _) in top_5]
        top_5_str = "; ".join(top_5_str_parts)

        final_dict["ig_positive"] = pos_sum
        final_dict["ig_negative"] = neg_sum
        final_dict["top_5_features"] = top_5_str
        import json
        final_dict["ig_explanation"] = json.dumps(ig_dict)

        writer_ig.writerow(final_dict)
        ig_csv.flush()

    logging.info("Processing remaining anomalies in the queue...")
    while not ig_queue.empty():
        try:
            data = ig_queue.get_nowait()
            final_dict = dict(data["raw_full_dict"])
            array_2d = data["array_2d"]
            score = data["score"]

            seq_3d = array_2d.reshape(1, WINDOW_SIZE, NUM_FEATURES)
            ig_attribs = model_wrapper.compute_integrated_gradients(seq_3d)
            pos_sum = float(ig_attribs[ig_attribs > 0].sum())
            neg_sum = float(ig_attribs[ig_attribs < 0].sum())

            ig_summed = ig_attribs.sum(axis=0)
            ig_dict = {}
            for i, feat_name in enumerate(preprocessor.all_features):
                ig_dict[feat_name] = float(ig_summed[i])
            sorted_ig = sorted(ig_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            top_5 = sorted_ig[:5]
            top_5_str_parts = [f"{k}={ig_dict[k]:.3f}" for (k, _) in top_5]
            top_5_str = "; ".join(top_5_str_parts)

            final_dict["ig_positive"] = pos_sum
            final_dict["ig_negative"] = neg_sum
            final_dict["top_5_features"] = top_5_str
            final_dict["ig_explanation"] = json.dumps(ig_dict)

            writer_ig.writerow(final_dict)
            ig_csv.flush()
        except Exception as e:
            logging.error(f"Error processing anomaly in queue: {e}")
            continue

    ig_csv.close()
    logging.info("IG worker thread stopping...")

def _extract_packet_info(packet):
    info = {
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

def single_df_to_dict(df, columns):
    import pandas as pd
    import numpy as np
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=columns)
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns].to_dict(orient="records")

def array_to_dict(arr_1d, columns):
    out = {}
    for i, col in enumerate(columns):
        out[col] = float(arr_1d[i])
    return out

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("simple_pipeline")

    def _stop_handler(sig, frame):
        logger.info("Received stop signal, shutting down threads...")
        stop_event.set()

    signal.signal(signal.SIGINT, _stop_handler)

    capture_thread = threading.Thread(target=capture_packets, args=(INTERFACE,), daemon=True)
    detect_thread = threading.Thread(target=detection_worker, daemon=True)
    grad_thread = threading.Thread(target=ig_worker, daemon=True)

    capture_thread.start()
    detect_thread.start()
    grad_thread.start()

    while not stop_event.is_set():
        time.sleep(1)

    logger.info("Main thread finishing, waiting for workers to stop...")
    capture_thread.join(timeout=2)
    detect_thread.join(timeout=2)
    grad_thread.join(timeout=2)
    logger.info("All done.")

if __name__ == "__main__":
    main()