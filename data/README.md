# Data Directory

This directory contains traffic data for DCRNN training.

## Dataset Overview

### METR-LA Dataset

The METR-LA dataset contains traffic speed data from 207 loop detectors on Los Angeles County highways.

| Property | Value |
|----------|-------|
| **Source** | Caltrans Performance Measurement System (PeMS) |
| **Sensors** | 207 loop detectors |
| **Time Range** | March 1, 2012 - June 30, 2012 |
| **Interval** | 5 minutes |
| **Total Timesteps** | 34,272 |
| **Features** | Traffic speed (mph) |

### Data Structure

```
data/
├── README.md           # This file
├── fetch_data.sh       # Download real METR-LA data
├── generate_sample_data.py  # Generate synthetic data
├── preprocess_metr_la.py    # Preprocess real data
├── raw/                # Raw downloaded data
│   ├── metr-la.h5
│   ├── graph_sensor_ids.txt
│   └── distances_la_2012.csv
└── processed/          # Preprocessed data ready for training
    ├── train.npz       # Training sequences
    ├── val.npz         # Validation sequences
    ├── test.npz        # Test sequences
    ├── adj.npy         # Adjacency matrix
    ├── adj_norm.npy    # Normalized adjacency
    ├── scaler.npy      # Normalization parameters [mean, std]
    └── metadata.txt    # Dataset metadata
```

## Usage

### Option 1: Synthetic Data (Quick Start)

Generate synthetic traffic data for testing:

```bash
cd data
python generate_sample_data.py
```

This creates realistic synthetic data mimicking METR-LA patterns:
- Daily rush hour patterns
- Weekly variations (weekday vs. weekend)
- Spatial correlations between sensors

### Option 2: Real METR-LA Data

Download and preprocess the real dataset:

```bash
cd data
./fetch_data.sh
```

This will:
1. Download `metr-la.h5` from the DCRNN repository
2. Download sensor graph information
3. Preprocess into training format

### Manual Download

If automatic download fails, manually download from:
- https://github.com/liyaguang/DCRNN/tree/master/data

Place files in `data/raw/`:
- `metr-la.h5`
- `graph_sensor_ids.txt`
- `distances_la_2012.csv`

Then run:
```bash
python preprocess_metr_la.py
```

## Data Format

### Input Sequences (X)

```
Shape: (num_samples, seq_len, num_nodes, input_dim)
       (N, 12, 207, 1)

- num_samples: Number of training examples
- seq_len: 12 timesteps (1 hour of history)
- num_nodes: 207 sensors
- input_dim: 1 (speed value)
```

### Target Sequences (Y)

```
Shape: (num_samples, pred_len, num_nodes, output_dim)
       (N, 1, 207, 1)

- pred_len: 1 timestep (5 minutes ahead)
```

### Adjacency Matrix

```
Shape: (num_nodes, num_nodes)
       (207, 207)

- Weighted adjacency based on road network distances
- Gaussian kernel: w_ij = exp(-d_ij^2 / 2σ^2)
- Row-normalized for diffusion convolution
```

## Preprocessing Details

### Normalization

Z-score normalization is applied:
```
x_normalized = (x - mean) / std
```

The scaler parameters are saved in `scaler.npy` for denormalization during inference.

### Train/Val/Test Split

| Split | Percentage | Approximate Samples |
|-------|------------|---------------------|
| Train | 70% | 23,974 |
| Val | 10% | 3,425 |
| Test | 20% | 6,850 |

Splits are chronological (not shuffled) to maintain temporal ordering.

### Adjacency Construction

1. Load sensor distances from `distances_la_2012.csv`
2. Apply Gaussian kernel to distances
3. Threshold to create sparse graph
4. Row-normalize for diffusion convolution

## Alternative Datasets

### PEMS-BAY

Similar dataset from Bay Area (San Francisco):
- 325 sensors
- 6 months of data
- Available from the same source

To use PEMS-BAY, modify `fetch_data.sh` to download `pems-bay.h5`.

### Custom Datasets

To use your own traffic data:

1. Prepare data as NumPy arrays:
   - Speed readings: `(timesteps, num_sensors)`
   - Adjacency matrix: `(num_sensors, num_sensors)`

2. Modify `preprocess_metr_la.py` or create a new preprocessing script

3. Run preprocessing to generate `train.npz`, `val.npz`, `test.npz`

## Citation

If using METR-LA data, please cite:

```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: 
         Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## Troubleshooting

### Download fails

1. Check internet connectivity
2. Try alternative download source (Zenodo)
3. Use synthetic data for development

### Preprocessing errors

1. Ensure all raw files are present
2. Check file integrity (not corrupted)
3. Install h5py: `pip install h5py`

### Memory issues

For large datasets, preprocess in chunks:
```python
# Modify preprocess script to process in batches
chunk_size = 10000
for i in range(0, total_samples, chunk_size):
    process_chunk(i, i + chunk_size)
```
