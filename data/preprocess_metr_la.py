#!/usr/bin/env python3
"""
METR-LA Dataset Preprocessing Script

This script processes the METR-LA traffic dataset into a format suitable for 
DCRNN training with distributed computing.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Try to import h5py, fall back to creating synthetic data
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("h5py not available, will use synthetic data generation")


def load_metr_la_data(data_path):
    """Load METR-LA data from h5 file"""
    if not HAS_H5PY:
        return None
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    with h5py.File(data_path, 'r') as f:
        data = f['df']['block0_values'][:]
    return data


def load_adjacency_matrix(distances_path, sensor_ids_path, sigma=0.1, epsilon=0.5):
    """
    Load and construct adjacency matrix from sensor distances
    
    Args:
        distances_path: Path to distances CSV
        sensor_ids_path: Path to sensor IDs file
        sigma: Gaussian kernel parameter
        epsilon: Threshold for adjacency
    
    Returns:
        Adjacency matrix (num_sensors x num_sensors)
    """
    if not os.path.exists(distances_path) or not os.path.exists(sensor_ids_path):
        return None
    
    # Load sensor IDs
    with open(sensor_ids_path, 'r') as f:
        sensor_ids = [int(line.strip()) for line in f if line.strip()]
    
    num_sensors = len(sensor_ids)
    sensor_id_to_idx = {sid: idx for idx, sid in enumerate(sensor_ids)}
    
    # Load distances
    distances_df = pd.read_csv(distances_path)
    
    # Build adjacency matrix
    adj = np.zeros((num_sensors, num_sensors))
    
    for _, row in distances_df.iterrows():
        from_id = int(row['from'])
        to_id = int(row['to'])
        distance = float(row['cost'])
        
        if from_id in sensor_id_to_idx and to_id in sensor_id_to_idx:
            i = sensor_id_to_idx[from_id]
            j = sensor_id_to_idx[to_id]
            # Gaussian kernel
            weight = np.exp(-distance ** 2 / (2 * sigma ** 2))
            if weight >= epsilon:
                adj[i, j] = weight
                adj[j, i] = weight
    
    return adj


def normalize_adjacency(adj):
    """Row-normalize adjacency matrix"""
    rowsum = adj.sum(axis=1) + 1e-8
    return adj / rowsum[:, np.newaxis]


def create_sequences(data, seq_len=12, pred_len=1):
    """
    Create input-output sequences for training
    
    Args:
        data: Raw time series data (time_steps x num_sensors)
        seq_len: Input sequence length
        pred_len: Prediction horizon
    
    Returns:
        X: Input sequences (num_samples x seq_len x num_sensors x 1)
        Y: Target sequences (num_samples x pred_len x num_sensors x 1)
    """
    num_samples = data.shape[0] - seq_len - pred_len + 1
    num_sensors = data.shape[1]
    
    X = np.zeros((num_samples, seq_len, num_sensors, 1))
    Y = np.zeros((num_samples, pred_len, num_sensors, 1))
    
    for i in range(num_samples):
        X[i] = data[i:i+seq_len, :, np.newaxis]
        Y[i] = data[i+seq_len:i+seq_len+pred_len, :, np.newaxis]
    
    return X, Y


def normalize_data(data, mean=None, std=None):
    """Z-score normalization"""
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / (std + 1e-8), mean, std


def split_data(X, Y, train_ratio=0.7, val_ratio=0.1):
    """Split data into train/val/test sets"""
    num_samples = X.shape[0]
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def main():
    script_dir = Path(__file__).parent
    raw_dir = script_dir / 'raw'
    processed_dir = script_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    print("METR-LA Data Preprocessing")
    print("=" * 50)
    
    # Load raw data
    data_path = raw_dir / 'metr-la.h5'
    distances_path = raw_dir / 'distances_la_2012.csv'
    sensor_ids_path = raw_dir / 'graph_sensor_ids.txt'
    
    data = load_metr_la_data(data_path)
    
    if data is None:
        print("Could not load METR-LA data. Run generate_sample_data.py instead.")
        return False
    
    print(f"Loaded data shape: {data.shape}")
    
    # Parameters
    seq_len = 12
    pred_len = 1
    
    # Normalize data
    data_norm, mean, std = normalize_data(data)
    print(f"Data normalized: mean={mean:.4f}, std={std:.4f}")
    
    # Create sequences
    X, Y = create_sequences(data_norm, seq_len=seq_len, pred_len=pred_len)
    print(f"Created sequences: X={X.shape}, Y={Y.shape}")
    
    # Split data
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Load adjacency matrix
    adj = load_adjacency_matrix(distances_path, sensor_ids_path)
    if adj is None:
        # Create random adjacency if distances not available
        num_sensors = data.shape[1]
        adj = np.random.rand(num_sensors, num_sensors)
        adj = (adj + adj.T) / 2
        adj[adj < 0.3] = 0
        np.fill_diagonal(adj, 0)
        print("Created synthetic adjacency matrix")
    
    adj_norm = normalize_adjacency(adj)
    
    # Save processed data
    np.savez(
        processed_dir / 'train.npz',
        X=X_train.astype(np.float32),
        Y=Y_train.astype(np.float32)
    )
    np.savez(
        processed_dir / 'val.npz',
        X=X_val.astype(np.float32),
        Y=Y_val.astype(np.float32)
    )
    np.savez(
        processed_dir / 'test.npz',
        X=X_test.astype(np.float32),
        Y=Y_test.astype(np.float32)
    )
    np.save(processed_dir / 'adj.npy', adj.astype(np.float32))
    np.save(processed_dir / 'adj_norm.npy', adj_norm.astype(np.float32))
    np.save(processed_dir / 'scaler.npy', np.array([mean, std], dtype=np.float32))
    
    print(f"\nData saved to: {processed_dir}")
    print("Preprocessing complete!")
    
    return True


if __name__ == '__main__':
    main()
