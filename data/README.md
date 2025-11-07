# Dataset Information

## Dataset Description

This directory contains the dataset for DCRNN (Diffusion Convolutional Recurrent Neural Network) training.

## Dataset Structure

```
data/
├── train/        # Training data
├── val/          # Validation data
└── test/         # Test data
```

## Fetching the Dataset

To download the full dataset, run:

```bash
./fetch_data.sh
```

**Note**: The current `fetch_data.sh` script is a placeholder. Update it with:
- Actual dataset URLs
- Paths to shared cluster storage
- Authentication credentials if required
- Data extraction/preprocessing steps

## Dataset Details

- **Source**: [To be filled]
- **Size**: [To be filled]
- **Format**: [To be filled]
- **License**: [To be filled]

## Sample Data

A small sample dataset is included for testing. The full dataset should be downloaded using `fetch_data.sh` before running full training.

## Preprocessing

If preprocessing is required, add scripts to this directory or reference them in `src/data.py`.

