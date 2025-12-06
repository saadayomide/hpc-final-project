# HPC Final Project: AI-Based Traffic Flow Prediction

## Scalable DCRNN for Urban Digital Twins

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org)

This project implements a distributed deep learning framework for traffic flow prediction using Diffusion Convolutional Recurrent Neural Networks (DCRNN). The system scales efficiently across multiple GPU nodes on HPC clusters, enabling real-time traffic prediction for smart city digital twin applications.

## 🎯 Project Goals

1. **Scalable Training**: Achieve >80% parallel efficiency when scaling from 1 to 8 GPU nodes
2. **Accurate Predictions**: Match state-of-the-art MAE (~3.5 mph) on METR-LA dataset
3. **Reproducibility**: Containerized solution with fixed seeds and documented configurations
4. **Performance Analysis**: Comprehensive profiling and bottleneck identification

## 📊 Latest Results

**Validation Run (Dec 5, 2025)**

| Metric | Value |
|--------|-------|
| Final Validation MAE | 0.336 |
| Final Validation RMSE | 0.424 |
| Training Throughput | 74.65 samples/sec |
| Best Validation Loss | 0.180 |

**Training Progress (5 epochs)**:

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE | Throughput |
|-------|------------|----------|---------|----------|------------|
| 1 | 0.2246 | 0.1859 | 0.342 | 0.431 | 74.69 s/s |
| 2 | 0.1775 | 0.1817 | 0.338 | 0.426 | 74.33 s/s |
| 3 | 0.1762 | 0.1812 | 0.338 | 0.425 | 74.66 s/s |
| 4 | 0.1751 | 0.1808 | 0.338 | 0.425 | 74.69 s/s |
| 5 | 0.1746 | 0.1804 | 0.336 | 0.424 | 74.65 s/s |

**Status Summary**:
- ✅ Single-node training: **Working** (74.65 samples/sec)
- ✅ Model convergence: **Validated** (loss decreasing)
- ✅ Data pipeline: **Working** (6039 train, 863 val samples)
- ✅ Checkpointing: **Working** (best + latest saved)

## 🏗️ Repository Structure

```
hpc-final-project/
├── src/                    # Source code
│   ├── train.py           # Main training script with DDP support
│   ├── data.py            # Data loading utilities
│   ├── model/             # DCRNN model implementation
│   │   └── dcrnn.py
│   └── utils/             # Metrics and monitoring
│       ├── metrics.py
│       └── monitoring.py
├── env/                    # Environment configuration
│   └── project.def        # Apptainer container definition
├── slurm/                  # SLURM job scripts
│   ├── baseline_1node.sbatch
│   ├── ddp_multi_node.sbatch
│   ├── full_train_cpu.sbatch  # Production training
│   └── *.sbatch           # Other job scripts
├── scripts/                # Automation scripts
│   ├── analyze_scaling.py
│   ├── analyze_profiling.py
│   └── *.sh               # Experiment runners
├── data/                   # Dataset and scripts
│   ├── processed/         # Preprocessed data
│   ├── generate_sample_data.py
│   └── README.md
├── results/                # Experiment outputs
│   └── quick_test_cpu_4817/ # Latest successful run
├── docs/                   # Documentation
│   ├── paper.md           # Research paper
│   ├── eurohpc_proposal.md
│   └── pitch.md
├── reproduce.md            # Reproduction instructions
├── SYSTEM.md              # System configuration
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- HPC cluster with SLURM scheduler
- Python 3.10+ with PyTorch 2.1+
- (Optional) Apptainer for containerized execution

### 1. Clone and Setup

```bash
git clone <repository-url>
cd hpc-final-project
```

### 2. Prepare Data

```bash
cd data && python generate_sample_data.py && cd ..
```

### 3. Run Training

```bash
# Quick test (5 epochs, ~7 min)
sbatch slurm/quick_test_cpu_fixed.sbatch

# Full training (50 epochs, ~1.2 hours)
sbatch slurm/full_train_cpu.sbatch
```

### 4. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f results/full_train_cpu_*.out
```

## 🔧 Model Architecture

**DCRNN (Diffusion Convolutional Recurrent Neural Network)**

- Input: 12 timesteps (1 hour) of traffic speed data
- Output: 1 timestep (5 minutes) prediction
- Sensors: 207 (METR-LA dataset)
- Hidden dimension: 64
- Layers: 2 DCGRUCells

Key features:
- Diffusion convolution captures spatial dependencies on road network
- GRU captures temporal dynamics
- Supports mixed-precision training (BF16/FP16)

## 📊 Output Files

Training produces:

| File | Description |
|------|-------------|
| `metrics.csv` | Per-epoch training metrics |
| `cpu_monitor.csv` | CPU utilization logs |
| `checkpoint_best.pth` | Best model checkpoint |
| `checkpoint_latest.pth` | Latest model checkpoint |

## 📖 Documentation

- **[reproduce.md](reproduce.md)**: Step-by-step reproduction instructions
- **[SYSTEM.md](SYSTEM.md)**: System configuration and requirements
- **[data/README.md](data/README.md)**: Dataset documentation
- **[docs/paper.md](docs/paper.md)**: Research paper
- **[docs/eurohpc_proposal.md](docs/eurohpc_proposal.md)**: EuroHPC proposal
- **[docs/pitch.md](docs/pitch.md)**: Presentation slides

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- METR-LA dataset from [DCRNN paper](https://github.com/liyaguang/DCRNN)
- Magic Castle cluster for compute resources
- PyTorch team for distributed training support
