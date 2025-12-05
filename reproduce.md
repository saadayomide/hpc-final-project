# Reproduction Instructions

This document provides step-by-step instructions to reproduce all experiments from the HPC Final Project on traffic flow prediction using distributed DCRNN.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running Experiments](#running-experiments)
5. [Analyzing Results](#analyzing-results)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Access to an HPC cluster with SLURM scheduler
- GPU nodes with NVIDIA GPUs (Tesla T4, V100, or A100)
- Apptainer/Singularity container runtime (v1.0+)
- Git

### Recommended
- Python 3.10+ (for analysis scripts)
- pandoc (for PDF generation)
- matplotlib, numpy, pandas (for plotting)

---

## Quick Start

For a quick verification that everything works:

```bash
# 1. Clone repository
git clone <repository-url>
cd hpc-final-project
git checkout v1.0-course-teamname  # Use release tag

# 2. Build container (if not pre-built)
./run.sh build

# 3. Generate sample data
cd data && python generate_sample_data.py && cd ..

# 4. Verify GPU access
./run.sh python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Run a quick training test
sbatch slurm/baseline_1node.sbatch
```

---

## Detailed Setup

### Step 1: Clone and Verify Repository

```bash
# Clone the repository
git clone <repository-url>
cd hpc-final-project

# For reproducibility, use the tagged release
git checkout v1.0-course-teamname

# Record git hash for your records
echo "Git hash: $(git rev-parse HEAD)"
echo "Date: $(date)"
```

### Step 2: Build Container

The Apptainer container ensures consistent environment across runs.

```bash
# Option A: Build from definition file
./run.sh build

# Option B: Manual build with more control
cd env
apptainer build project.sif project.def
cd ..

# Verify container
./run.sh python --version
./run.sh python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

**Expected output:**
```
Python 3.10.x
PyTorch 2.1.0, CUDA True
```

If the container doesn't build, see [Troubleshooting](#container-issues).

### Step 3: Prepare Data

#### Option A: Synthetic Data (Quick Start)
```bash
cd data
python generate_sample_data.py
cd ..

# Verify
ls -la data/processed/
```

#### Option B: Real METR-LA Dataset
```bash
cd data
./fetch_data.sh
cd ..
```

**Expected files in `data/processed/`:**
- `train.npz` - Training data
- `val.npz` - Validation data  
- `test.npz` - Test data
- `adj.npy` - Adjacency matrix
- `adj_norm.npy` - Normalized adjacency
- `scaler.npy` - Data normalization parameters
- `metadata.txt` - Dataset info

### Step 4: Verify GPU Access

```bash
# Check via container
./run.sh nvidia-smi

# Check PyTorch CUDA
./run.sh python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Running Experiments

### Baseline Training (1 Node)

```bash
# Submit single-node training job
sbatch slurm/baseline_1node.sbatch

# Monitor job
squeue -u $USER
tail -f results/baseline_*.out

# After completion, check results
ls results/baseline_*/
cat results/baseline_*/metrics.csv
```

### Multi-Node DDP Training (2+ Nodes)

```bash
# Submit 2-node DDP job
sbatch slurm/ddp_multi_node.sbatch

# Monitor
squeue -u $USER
tail -f results/ddp_*.out

# Verify DDP initialization in output:
# Look for: "RANK=0, WORLD_SIZE=2"
```

### Scaling Experiments

#### Strong Scaling
```bash
# Run strong scaling (1, 2, 4 nodes)
./scripts/strong_scaling.sh

# Monitor all jobs
squeue -u $USER

# After completion, analyze
python scripts/analyze_scaling.py \
    --results results/strong_scaling_* \
    --type strong
```

#### Weak Scaling
```bash
# Run weak scaling
./scripts/weak_scaling.sh

# Analyze
python scripts/analyze_scaling.py \
    --results results/weak_scaling_* \
    --type weak
```

### Profiling Run

```bash
# Submit profiling job (single node with full monitoring)
sbatch slurm/baseline_1node.sbatch  # Uses --monitor-gpu --monitor-cpu

# After completion
python scripts/analyze_profiling.py \
    --results results/baseline_XXXX/
```

---

## Analyzing Results

### Training Metrics

Each training run produces a `metrics.csv` with per-epoch data:

```bash
# View metrics
cat results/baseline_XXXX/metrics.csv

# Columns:
# epoch, train_loss, val_loss, val_mae, val_rmse, 
# epoch_time_sec, throughput_samples_per_sec, 
# data_load_time_sec, compute_time_sec, gpu_util_avg, cpu_util_avg
```

### Scaling Analysis

```bash
# Generate scaling plots
python scripts/analyze_scaling.py \
    --results results/strong_scaling_* \
    --type strong \
    --output results/scaling/

# Output files:
# - strong_scaling_analysis.csv
# - scaling_analysis.png
# - scaling_analysis.svg
```

### Profiling Analysis

```bash
# Generate profiling report
python scripts/analyze_profiling.py \
    --results results/baseline_XXXX/

# Output files:
# - profiling_report.txt
# - profiling_analysis.json
# - profiling_analysis.png
```

---

## Configuration Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Batch size per GPU |
| `--lr` | 0.001 | Learning rate |
| `--precision` | fp32 | Training precision (fp32/fp16/bf16) |
| `--num-workers` | 4 | DataLoader workers per GPU |
| `--hidden-dim` | 64 | Model hidden dimension |
| `--num-layers` | 2 | Number of GRU layers |
| `--seed` | 42 | Random seed (fixed for reproducibility) |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Input dimension | 1 (speed) |
| Sequence length | 12 timesteps (1 hour) |
| Prediction length | 1 timestep (5 minutes) |
| Number of sensors | 207 (METR-LA) |

### DDP Environment Variables

```bash
export MASTER_ADDR=<first_node_hostname>
export MASTER_PORT=29500
export WORLD_SIZE=<num_nodes>
export RANK=<node_rank>
export LOCAL_RANK=0
export NCCL_DEBUG=WARN
```

---

## Expected Results

### Training Metrics

| Metric | Expected Range |
|--------|----------------|
| Final train loss | 0.05 - 0.10 |
| Final val loss | 0.07 - 0.12 |
| Val MAE | 3.0 - 4.5 mph |
| Val RMSE | 4.5 - 6.5 mph |

### Scaling Efficiency

| Nodes | Strong Scaling Efficiency |
|-------|--------------------------|
| 1 | 100% (baseline) |
| 2 | 85-98% |
| 4 | 75-92% |
| 8 | 55-85% |

---

## Troubleshooting

### Container Issues

**Problem:** Container build fails
```bash
# Check Apptainer version
apptainer --version  # Should be 1.0+

# Use scratch for build cache
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Rebuild
./run.sh build --force
```

**Problem:** GPU not detected in container
```bash
# Ensure --nv flag is used
apptainer exec --nv env/project.sif nvidia-smi

# Check NVIDIA driver on host
nvidia-smi
```

### DDP Issues

**Problem:** DDP hangs during initialization
```bash
# Enable verbose debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check network interface
ip addr  # Look for ib0 (InfiniBand) or eth0

# Set correct interface
export NCCL_SOCKET_IFNAME=ib0  # or eth0
```

**Problem:** Gradient synchronization errors
```bash
# Check firewall (should allow port 29500)
# Try different master port
export MASTER_PORT=29501
```

### Performance Issues

**Problem:** Low GPU utilization (<60%)
```bash
# Increase batch size
--batch-size 64

# Increase data workers
--num-workers 8

# Check data loading bottleneck
# If data_load_time_sec is high, data loading is the issue
```

**Problem:** Out of memory
```bash
# Reduce batch size
--batch-size 16

# Use mixed precision
--precision bf16

# Enable gradient checkpointing (requires code modification)
```

---

## Verification Checklist

Before considering reproduction complete:

- [ ] Container builds successfully
- [ ] Single-node training completes without errors
- [ ] GPU utilization is >70% during training
- [ ] Multi-node DDP initializes correctly (check RANK output)
- [ ] Scaling experiments produce valid CSV files
- [ ] Plots are generated in PNG and SVG formats
- [ ] Results are within expected ranges
- [ ] All random seeds produce reproducible results

---

## Contact

For issues with reproduction, please open a GitHub issue with:
1. Error message (full traceback)
2. SLURM job output files
3. System information (`nvidia-smi`, `module list`)
4. Steps to reproduce the issue
