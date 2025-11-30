#!/usr/bin/env python3
"""
Generate Synthetic Sample Data for DCRNN Training

This script creates realistic synthetic traffic data when the actual 
METR-LA dataset is not available. The synthetic data mimics the 
characteristics of real traffic patterns.
"""

import os
import numpy as np
from pathlib import Path


def generate_traffic_pattern(num_timesteps, num_sensors, seed=42):
    """
    Generate synthetic traffic data that mimics real traffic patterns.
    
    Features:
    - Daily periodicity (rush hours)
    - Weekly patterns (weekday vs weekend)
    - Spatial correlations between nearby sensors
    - Random noise
    """
    np.random.seed(seed)
    
    # Time parameters (5-minute intervals = 288 per day)
    intervals_per_hour = 12
    intervals_per_day = 24 * intervals_per_hour
    
    # Generate base time series with daily periodicity
    t = np.arange(num_timesteps)
    
    # Daily pattern: morning rush (7-9), evening rush (17-19)
    hour_of_day = (t % intervals_per_day) / intervals_per_hour
    
    # Morning rush peak around 8am
    morning_rush = np.exp(-((hour_of_day - 8) ** 2) / (2 * 1.5 ** 2))
    # Evening rush peak around 18pm
    evening_rush = np.exp(-((hour_of_day - 18) ** 2) / (2 * 1.5 ** 2))
    
    # Base traffic pattern
    base_pattern = 40 + 30 * morning_rush + 35 * evening_rush
    
    # Weekly pattern (reduce traffic on weekends)
    day_of_week = (t // intervals_per_day) % 7
    weekend_factor = np.where((day_of_week >= 5), 0.7, 1.0)
    
    base_pattern = base_pattern * weekend_factor
    
    # Generate data for all sensors with spatial correlations
    data = np.zeros((num_timesteps, num_sensors))
    
    # Create sensor groups (simulating different highway corridors)
    num_groups = 5
    sensors_per_group = num_sensors // num_groups
    
    for g in range(num_groups):
        start_idx = g * sensors_per_group
        end_idx = min((g + 1) * sensors_per_group, num_sensors)
        
        # Group-specific variation
        group_factor = 0.8 + 0.4 * np.random.rand()
        group_noise = np.random.randn(num_timesteps) * 5
        
        for s in range(start_idx, end_idx):
            # Sensor-specific variation
            sensor_factor = 0.9 + 0.2 * np.random.rand()
            sensor_noise = np.random.randn(num_timesteps) * 3
            
            # Slight phase shift for nearby sensors (propagation delay)
            shift = int((s - start_idx) * 2)
            pattern = np.roll(base_pattern, shift)
            
            # Combine all components
            data[:, s] = pattern * group_factor * sensor_factor + group_noise + sensor_noise
    
    # Ensure non-negative values and reasonable range
    data = np.clip(data, 0, 100)
    
    return data


def generate_adjacency_matrix(num_sensors, sparsity=0.3, seed=42):
    """
    Generate a synthetic adjacency matrix representing sensor connectivity.
    
    Creates a graph structure where sensors are connected if they are
    geographically close (simulated by index distance).
    """
    np.random.seed(seed + 1)
    
    adj = np.zeros((num_sensors, num_sensors))
    
    # Create connections based on sensor proximity
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            # Closer sensors are more likely to be connected
            distance = abs(i - j)
            probability = np.exp(-distance / 5)
            
            if np.random.rand() < probability:
                # Weight based on inverse distance
                weight = 1.0 / (1.0 + distance * 0.1)
                adj[i, j] = weight
                adj[j, i] = weight
    
    return adj


def normalize_adjacency(adj):
    """Row-normalize adjacency matrix"""
    rowsum = adj.sum(axis=1) + 1e-8
    return adj / rowsum[:, np.newaxis]


def create_sequences(data, seq_len=12, pred_len=1):
    """Create input-output sequences"""
    num_samples = data.shape[0] - seq_len - pred_len + 1
    num_sensors = data.shape[1]
    
    X = np.zeros((num_samples, seq_len, num_sensors, 1))
    Y = np.zeros((num_samples, pred_len, num_sensors, 1))
    
    for i in range(num_samples):
        X[i] = data[i:i+seq_len, :, np.newaxis]
        Y[i] = data[i+seq_len:i+seq_len+pred_len, :, np.newaxis]
    
    return X, Y


def split_data(X, Y, train_ratio=0.7, val_ratio=0.1):
    """Split data into train/val/test sets"""
    num_samples = X.shape[0]
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    return (
        (X[:train_end], Y[:train_end]),
        (X[train_end:val_end], Y[train_end:val_end]),
        (X[val_end:], Y[val_end:])
    )


def main():
    script_dir = Path(__file__).parent
    processed_dir = script_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    print("Generating Synthetic Traffic Data")
    print("=" * 50)
    
    # Configuration
    num_sensors = 207  # METR-LA has 207 sensors
    num_days = 30  # ~1 month of data
    intervals_per_day = 288  # 5-minute intervals
    num_timesteps = num_days * intervals_per_day
    
    seq_len = 12  # 1 hour of history
    pred_len = 1   # Predict 5 minutes ahead
    seed = 42
    
    print(f"Configuration:")
    print(f"  Sensors: {num_sensors}")
    print(f"  Days: {num_days}")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Prediction length: {pred_len}")
    print()
    
    # Generate traffic data
    print("Generating traffic patterns...")
    data = generate_traffic_pattern(num_timesteps, num_sensors, seed)
    print(f"  Raw data shape: {data.shape}")
    print(f"  Value range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Normalize data
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / (std + 1e-8)
    print(f"  Normalized: mean={mean:.2f}, std={std:.2f}")
    
    # Generate adjacency matrix
    print("\nGenerating adjacency matrix...")
    adj = generate_adjacency_matrix(num_sensors, seed=seed)
    adj_norm = normalize_adjacency(adj)
    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Edge density: {(adj > 0).sum() / (num_sensors ** 2):.3f}")
    
    # Create sequences
    print("\nCreating sequences...")
    X, Y = create_sequences(data_norm, seq_len=seq_len, pred_len=pred_len)
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    
    # Split data
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y)
    print(f"\nData splits:")
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")
    
    # Save processed data
    print(f"\nSaving data to {processed_dir}...")
    
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
    
    # Save metadata
    metadata = {
        'num_sensors': num_sensors,
        'num_timesteps': num_timesteps,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'mean': float(mean),
        'std': float(std),
        'seed': seed,
        'synthetic': True
    }
    
    with open(processed_dir / 'metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")
    
    print("\n" + "=" * 50)
    print("Data generation complete!")
    print(f"Files saved to: {processed_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
