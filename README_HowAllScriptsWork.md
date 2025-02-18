# Overview: How All Scripts in `simple_code` Work Together

This document explains the overall Intrusion Detection System (IDS) pipeline, focusing on the "simple_code" directory:

1. **initial_data_capture.py**
	- **Purpose**: Perform an **offline** capture of ICS traffic from a given network interface, engineering features with a sliding window.
	- **Core**:
		- Uses `scapy.sniff` or a callback to read packets.
		- Calls `SimpleFeatureEngineeringPreprocessor.add_packet(...)` for each captured packet.
		- Writes resulting features to a CSV.
	- **When to use**: If you want to generate a dataset for training or analysis in offline scenarios.

2. **pre_processing.py**
	- **Purpose**: Provide a single entry point for training or applying a data preprocessor on CSV data.
	- **Core**:
		- `--mode train`: Fit the scaler/label encoders on a training CSV, saving state to disk.
		- `--mode transform`: Load that saved state to transform new CSV data consistently.
	- **When to use**: If you have raw CSV data and want to produce final numeric data for training or any other step.

3. **simple_preprocessor.py**
	- **Purpose**: Offers a more direct, real-time *sliding window* approach to feature engineering.
	- **Core**:
		- Maintains a buffer of last N packets.
		- Computes aggregated features such as interarrival times, unique IP counts, etc.
		- Provides label encoding & optional scaling.
	- **When to use**: Real-time scenarios (like in `simple_pipeline.py`) or if you prefer an online approach for offline data capture.

4. **simple_model.py**
	- **Purpose**: Contains `SimpleLSTMAutoencoder` for anomaly detection and integrated gradients (IG).
	- **Core**:
		- A symmetrical LSTM AE (2-layer encoder, 2-layer decoder).
		- `detect_anomaly(...)` returns a boolean + anomaly score.
		- `compute_integrated_gradients(...)` yields per-feature attributions.
	- **When to use**:
		- Called by a pipeline script for real-time detection.
		- Or by `train_lstm_autoencoder.py` during training.

5. **train_lstm_autoencoder.py**
	- **Purpose**: Train the LSTM autoencoder on a *preprocessed* CSV dataset and compute anomalies/IG on that dataset.
	- **Core**:
		- Creates sliding windows from 2D data.
		- Trains and picks best model from validation (via Keras callbacks).
		- Generates a final CSV with anomaly labels and integrated gradients.
	- **When to use**:
		- After you have data (e.g. from `initial_data_capture.py` or from `pre_processing.py`).
		- Produces a `model.h5` (and `.keras`) plus an anomaly threshold text file.

6. **simple_pipeline.py**
	- **Purpose**: **Live** real-time detection pipeline that:
		- Captures packets asynchronously.
		- Feeds them into a detection thread using `SimpleFeatureEngineeringPreprocessor`.
		- Uses `SimpleLSTMAutoencoder.detect_anomaly` to label each packet’s anomaly score.
		- Outputs all data to `live_capture.csv`.
		- If anomaly, pushes sample to an IG worker, which logs results to `anomalies_ig.csv`.
	- **When to use**: The final real-time IDS pipeline, once you have a trained model & preprocessor.

## Summary of Flow:
1. **Offline Data Generation (Optional)**:
	- `initial_data_capture.py`: produce an offline CSV with features for training data or testing.
2. **Preprocessing**:
	- If not using the “simple_preprocessor.py” approach, you can do `pre_processing.py --mode train` to get scaling/encoder states,
		then `pre_processing.py --mode transform` to create final numeric CSV from your raw data.
3. **Model Training**:
	- `train_lstm_autoencoder.py --input-file path_to_preprocessed.csv`
		=> trains the LSTM autoencoder, picks best model, determines threshold, writes `model.h5`, etc.
4. **Live Detection**:
	- `simple_pipeline.py` uses:
		- The same preprocessor state (`SimpleFeatureEngineeringPreprocessor.load_state(...)`)
		- The trained LSTM model weights from `model.keras` or `model.h5`
		=> Real-time detection on a network interface, with anomalies logged to `anomalies_ig.csv`.

## Typical Commands
- **initial_data_capture**:
	```bash
	python3 initial_data_capture.py --interface eth0 --output-csv training_capture.csv --window-size 20

	•	pre_processing:
	•	Train:

python3 pre_processing.py --mode train --training_file training_capture.csv \
	--save_dir models/preprocessing --normalization_method minmax


	•	Transform:

python3 pre_processing.py --mode transform --input_file new_data.csv \
	--output_dir processed_data --preprocessor_state models/preprocessing --normalization minmax


	•	train_lstm_autoencoder:

python3 train_lstm_autoencoder.py --input-file processed_data/preprocessed_data.csv \
	--seq-length 20 --epochs 10 --batch-size 32 \
	--save-dir models/lstm_autoencoder --threshold-percentile 95.0


	•	simple_pipeline (live detection):

python3 simple_pipeline.py



With these steps, you have a complete pipeline from capturing ICS data, to training a model, to running real-time detection with integrated gradients.