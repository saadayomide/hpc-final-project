# Reproduction Instructions

This document provides exact commands and configurations to reproduce all experiments.

## Prerequisites

- Access to an HPC cluster with GPU nodes (SLURM scheduler)
- Apptainer/Singularity container runtime
- Git

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd hpc-final-project
git checkout v1.0-course-teamname  # Use release tag

# 2. Build container
./run.sh build

# 3. Generate or fetch data
cd data
python generate_sample_data.py  # OR ./fetch_data.sh for real data
cd ..

# 4. Verify setup
./run.sh python -c "import torch; print(torch.cuda.is_available())"

# 5. Run baseline training
sbatch slurm/baseline_1node.sbatch

# 6. Run scaling experiments
./scripts/strong_scaling.sh
./scripts/weak_scaling.sh
```

---

## Step 1: Environment Setup

### 1.1 Clone Repository

```bash
git clone <repository-url>
cd hpc-final-project

# For reproducibility, use the tagged release
git checkout v1.0-course-teamname

# Record git hash
echo "Git hash: $(git rev-parse HEAD)" > EXPERIMENT_LOG.txt
```

### 1.2 Build Container

```bash
# Build Apptainer container
./run.sh build

# Verify container
./run.sh python --version
./run.sh python -c "import torch; print(torch.__version__)"

# Record container checksum
sha256sum env/project.sif >> EXPERIMENT_LOG.txt
```

### 1.3 Verify GPU Access

```bash
./run.sh python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

Expected output:
```
CUDA available: True
GPU count: 4
GPU name: NVIDIA A100-SXM4-40GB
CUDA version: 11.8
```

---

## Step 2: Data Preparation

### Option A: Synthetic Data (for testing)

```bash
cd data
python generate_sample_data.py
cd ..

# Verify
ls -la data/processed/
```

Expected files:
- `train.npz` (X and Y arrays)
- `val.npz`
- `test.npz`
- `adj.npy` (adjacency matrix)
- `adj_norm.npy` (normalized adjacency)
- `scaler.npy` (mean, std)
- `metadata.txt`

### Option B: METR-LA Dataset (production)

```bash
cd data
./fetch_data.sh
cd ..
```

This downloads and preprocesses the METR-LA traffic dataset.

---

## Step 3: Baseline Training (1 Node)

### 3.1 Interactive Test

```bash
# Quick test (1 epoch, small batch)
./run.sh python src/train.py \
    --data ./data \
    --epochs 1 \
    --batch-size 32 \
    --results ./results/test_run \
    --seed 42
```

### 3.2 Submit SLURM Job

Edit `slurm/baseline_1node.sbatch` to set your account and partition:

```bash
#SBATCH --account=your-account
#SBATCH --partition=gpu
```

Submit:
```bash
sbatch slurm/baseline_1node.sbatch
```

Monitor:
```bash
squeue -u $USER
tail -f results/baseline_*.out
```

---

## Step 4: Multi-Node DDP Training

### 4.1 Edit SLURM Script

Edit `slurm/ddp_multi_node.sbatch`:

```bash
#SBATCH --account=your-account
#SBATCH --partition=gpu
#SBATCH --nodes=2
```

### 4.2 Submit Job

```bash
sbatch slurm/ddp_multi_node.sbatch
```

### 4.3 Verify DDP

Check output for DDP initialization:
```bash
grep "NCCL" results/ddp_*.out
grep "World Size" results/ddp_*.out
```

Expected:
```
Using DDP: True
Rank: 0, World Size: 8, Local Rank: 0
```

---

## Step 5: Scaling Experiments

### 5.1 Configure Scripts

Edit `scripts/strong_scaling.sh` and `scripts/weak_scaling.sh`:

```bash
ACCOUNT="your-account"
PARTITION="gpu"
NODES_LIST=(1 2 4)  # Adjust based on quota
```

### 5.2 Run Strong Scaling

```bash
./scripts/strong_scaling.sh
```

This submits jobs for 1, 2, and 4 nodes with fixed problem size.

### 5.3 Run Weak Scaling

```bash
./scripts/weak_scaling.sh
```

This submits jobs with fixed work per GPU.

### 5.4 Run Sensitivity Sweep

```bash
./scripts/sensitivity_sweep.sh
```

Sweeps batch size and num_workers.

### 5.5 Analyze Results

After all jobs complete:

```bash
# Strong scaling analysis
python scripts/analyze_scaling.py \
    --results results/strong_scaling_* \
    --type strong

# Weak scaling analysis
python scripts/analyze_scaling.py \
    --results results/weak_scaling_* \
    --type weak

# Sensitivity analysis
python scripts/analyze_scaling.py \
    --results results/sensitivity_* \
    --type sensitivity
```

Output files:
- `*_analysis.csv` - Numeric results
- `*_plots.png` - Visualization
- `*_plots.svg` - Vector graphics

---

## Step 6: Profiling

### 6.1 Run with Monitoring

```bash
./run.sh python src/train.py \
    --data ./data \
    --epochs 10 \
    --batch-size 64 \
    --results ./results/profiling_run \
    --seed 42 \
    --monitor-gpu \
    --monitor-cpu
```

### 6.2 Analyze Profiling Data

```bash
python scripts/analyze_profiling.py \
    --results ./results/profiling_run
```

Output:
- `profiling_report.txt` - Text summary
- `profiling_analysis.json` - Machine-readable
- `profiling_analysis.png` - Plots

---

## Step 7: Generate Sample Results (for documentation)

If you need sample results for documentation without running experiments:

```bash
python scripts/generate_sample_results.py
```

This creates realistic synthetic results in `results/`.

---

## Reproducibility Seeds

All experiments use fixed random seeds:

| Seed Type | Value |
|-----------|-------|
| Python random | 42 |
| NumPy | 42 |
| PyTorch | 42 |
| CUDA | 42 (all GPUs) |

Set via `--seed 42` flag in training script.

---

## Configuration Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 1 | Number of epochs |
| `--batch-size` | 32 | Batch size per GPU |
| `--lr` | 0.001 | Learning rate |
| `--precision` | fp32 | fp32, fp16, or bf16 |
| `--num-workers` | 4 | Data loader workers |
| `--hidden-dim` | 64 | Model hidden dimension |
| `--num-layers` | 2 | Number of GRU layers |
| `--seed` | 42 | Random seed |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Input dimension | 1 (speed) |
| Hidden dimension | 64 |
| Number of layers | 2 |
| Dropout | 0.1 |
| Sequence length | 12 (1 hour) |
| Prediction length | 1 (5 minutes) |

### DDP Configuration

| Variable | Value |
|----------|-------|
| Backend | NCCL |
| Master port | 29500 |
| OMP threads | 8 |

---

## Expected Results

### Training Metrics

| Metric | Expected Range |
|--------|----------------|
| Final train loss | 0.05 - 0.08 |
| Final val loss | 0.06 - 0.10 |
| Val MAE | 3.2 - 4.0 mph |
| Val RMSE | 4.8 - 6.0 mph |

### Scaling Performance

| Nodes | Efficiency (expected) |
|-------|----------------------|
| 1 | 100% |
| 2 | 90-98% |
| 4 | 85-95% |
| 8 | 75-90% |

---

## Troubleshooting

### Container Build Fails

```bash
# Check Apptainer version
apptainer --version

# Try with more memory
export APPTAINER_TMPDIR=/scratch/$USER/tmp
mkdir -p $APPTAINER_TMPDIR
./run.sh build --force
```

### GPU Not Found

```bash
# Check NVIDIA driver
nvidia-smi

# Check container GPU access
./run.sh nvidia-smi
```

### DDP Initialization Hangs

```bash
# Check network interface
ip addr

# Set correct interface
export NCCL_SOCKET_IFNAME=ib0

# Enable debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Out of Memory

```bash
# Reduce batch size
--batch-size 16

# Enable gradient checkpointing
# (requires code modification)

# Use mixed precision
--precision bf16
```

---

## Verification Checklist

Before submission, verify:

- [ ] Container builds without errors
- [ ] Single-node training completes
- [ ] Multi-node training initializes correctly
- [ ] Scaling experiments produce valid CSV files
- [ ] Plots are generated in PNG/SVG formats
- [ ] All random seeds are fixed
- [ ] SLURM accounting is saved
- [ ] Results match expected ranges

---

## Contact

For issues, open a GitHub issue or contact the team.
