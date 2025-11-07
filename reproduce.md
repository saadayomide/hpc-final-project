# Reproduction Instructions

This document provides exact commands and configurations to reproduce the experiments.

## Phase 1: One-Node Functional Prototype

### Git Hash
```
git clone <repository-url>
cd hpc-final-project
git checkout <commit-hash>
```

**Note**: Replace `<commit-hash>` with the actual git commit hash for reproducibility.

### Container Build
```bash
./run.sh build
```

Or force rebuild:
```bash
./run.sh build --force
```

### Container Version Locking

To ensure reproducibility, record the container hash:
```bash
apptainer inspect --deffile env/project.sif | grep -A 5 "Build"
```

Or use container checksum:
```bash
sha256sum env/project.sif > env/container.sha256
```

### System Configuration

See `SYSTEM.md` for detailed system information including:
- Node types and specifications
- Module versions
- Driver and runtime versions

### Data Preparation

```bash
cd data
./fetch_data.sh
```

**Note**: The current implementation uses synthetic data generation. For production, update `fetch_data.sh` to download actual datasets.

## Running Experiments

### Phase 1: Baseline 1-Node Training

**Exact sbatch command:**
```bash
cd slurm
sbatch baseline_1node.sbatch
```

**Or manually:**
```bash
cd slurm
sbatch --job-name=dcrnn-1n \
  --account=<account> \
  --partition=<partition> \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --time=00:30:00 \
  --output=../results/1n_%j.out \
  --error=../results/1n_%j.err \
  baseline_1node.sbatch
```

**Direct execution (for testing):**
```bash
./run.sh python -m src.train \
  --data ./data \
  --epochs 1 \
  --batch-size 32 \
  --precision bf16 \
  --num-workers 4 \
  --results ./results/test_run \
  --seed 42 \
  --monitor-gpu \
  --monitor-cpu
```

### Reproducibility Seeds

All experiments use fixed random seeds:
- Python random seed: `42`
- NumPy random seed: `42`
- PyTorch random seed: `42`
- CUDA random seed: `42` (all GPUs)

The seed is set in `src/train.py` via the `--seed` flag.

### Version Locking

**Container Base Image:**
- Base: `nvcr.io/nvidia/pytorch:24.04-py3`
- Python: 3.10 (from base image)
- PyTorch: Version from base image (check with `./run.sh python -c "import torch; print(torch.__version__)"`)

**Key Dependencies:**
- torch-geometric
- torch-sparse
- torch-scatter
- torch-cluster
- DGL, dgllife

**To record module versions (if using modules):**
```bash
module list > SYSTEM.md.modules
```

**To record container environment:**
```bash
./run.sh pip list > env/pip_list.txt
```

### Expected Outputs

Results will be saved in `results/<JOBID>/`:
- `metrics.csv` - Per-epoch training metrics (loss, throughput, GPU/CPU utilization)
- `gpu_monitor.csv` - GPU utilization logs (if `--monitor-gpu` enabled)
- `cpu_monitor.csv` - CPU utilization logs (if `--monitor-cpu` enabled)
- `checkpoint_latest.pth` - Latest model checkpoint
- `checkpoint_best.pth` - Best model checkpoint (lowest validation loss)
- `sacct_summary.txt` - SLURM accounting summary

**Metrics CSV format:**
- epoch: Epoch number
- train_loss: Training loss
- val_loss: Validation loss
- val_mae: Validation MAE
- val_rmse: Validation RMSE
- epoch_time_sec: Wall-clock time per epoch
- throughput_samples_per_sec: Training throughput
- gpu_util_avg: Average GPU utilization (%)
- cpu_util_avg: Average CPU utilization (%)

### Measurement Details

**Wall-clock time:**
- Measured per epoch and per iteration
- Recorded in `metrics.csv` as `epoch_time_sec`

**Throughput:**
- Calculated as: `num_samples / epoch_time`
- Recorded in `metrics.csv` as `throughput_samples_per_sec`

**GPU Utilization:**
- Logged via `nvidia-smi` at 1-second intervals
- Saved to `gpu_monitor.csv`
- Average reported in `metrics.csv`

**CPU Utilization:**
- Logged via `psutil` at 1-second intervals
- Saved to `cpu_monitor.csv`
- Average reported in `metrics.csv`

**SLURM Accounting:**
- Saved via `sacct -j <JOBID>` to `sacct_summary.txt`

### Verification

To verify the environment is set up correctly:
```bash
./run.sh python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

Expected output:
```
True
2.x.x+cu121
```

To verify CUDA and GPU access:
```bash
./run.sh python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Multi-Node Training (Future Phase)

```bash
cd slurm
./run.sh multi_node.sbatch
```

### Profiling (Future Phase)

```bash
cd slurm
./run.sh profiling.sbatch
```

## Phase 2: Multi-Node Scaling Matrix

### Multi-Node DDP Training

**Exact sbatch command:**
```bash
cd slurm
sbatch ddp_multi_node.sbatch
```

**Or manually:**
```bash
cd slurm
sbatch --job-name=dcrnn-ddp \
  --account=<account> \
  --partition=<partition> \
  --nodes=2 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  --gpus-per-task=1 \
  --cpus-per-task=8 \
  --time=01:00:00 \
  --output=../results/ddp_%j.out \
  --error=../results/ddp_%j.err \
  ddp_multi_node.sbatch
```

### Strong Scaling Experiments

**Fixed problem size, varying number of nodes**

```bash
cd scripts
# Edit strong_scaling.sh to set your account/partition
./strong_scaling.sh
```

This will submit jobs for 1, 2, and 4 nodes (adjust `NODES_LIST` based on quota).

**Configuration:**
- Fixed batch size per GPU: 32
- Fixed problem size (same data, seq_len, model)
- Varying: Number of nodes (1, 2, 4)

**Results:**
- Time per epoch should decrease with more nodes
- Throughput should increase with more nodes
- Parallel efficiency = speedup / num_nodes

### Weak Scaling Experiments

**Fixed work per GPU, varying number of nodes**

```bash
cd scripts
# Edit weak_scaling.sh to set your account/partition
./weak_scaling.sh
```

**Configuration:**
- Fixed batch size per GPU: 32 (same work per GPU)
- Varying: Number of nodes (1, 2, 4)

**Results:**
- Time per epoch should stay roughly constant
- Throughput should scale linearly with nodes

### Sensitivity Sweep

**Varying batch size and num_workers**

```bash
cd scripts
# Edit sensitivity_sweep.sh to set your account/partition
./sensitivity_sweep.sh
```

**Configuration:**
- Fixed: 2 nodes
- Varying: batch_size ∈ {32, 64, 128}, num_workers ∈ {2, 4, 8}

**Results:**
- Identify optimal batch size and num_workers
- Trade-off between throughput and memory usage

### Analyzing Scaling Results

After all jobs complete, analyze results:

```bash
# Strong scaling
python scripts/analyze_scaling.py \
  --results results/strong_scaling_YYYYMMDD_HHMMSS \
  --type strong

# Weak scaling
python scripts/analyze_scaling.py \
  --results results/weak_scaling_YYYYMMDD_HHMMSS \
  --type weak

# Sensitivity sweep
python scripts/analyze_scaling.py \
  --results results/sensitivity_YYYYMMDD_HHMMSS \
  --type sensitivity
```

**Output:**
- CSV files with analysis results
- PNG/SVG plots showing:
  - Time vs Nodes
  - Throughput vs Nodes
  - Speedup/Efficiency
  - Sensitivity heatmaps

### Phase 2 Metrics

**Measured:**
- Time per epoch (wall-clock)
- Throughput (samples/s)
- Parallel efficiency = speedup / num_nodes
- GPU/CPU utilization
- Data loading time
- Compute time

**Expected Outputs:**
- `results/strong_scaling_*/strong_scaling_analysis.csv`
- `results/strong_scaling_*/strong_scaling_plots.png`
- `results/weak_scaling_*/weak_scaling_analysis.csv`
- `results/weak_scaling_*/weak_scaling_plots.png`
- `results/sensitivity_*/sensitivity_analysis.csv`
- `results/sensitivity_*/sensitivity_plots.png`
- `sacct_summary.txt` for each job

### DDP Configuration

**Environment variables:**
- `OMP_NUM_THREADS=8` - OpenMP threads
- `NCCL_DEBUG=INFO` - NCCL debugging
- `NCCL_SOCKET_IFNAME=ib0` - InfiniBand interface (if needed)

**PyTorch DDP:**
- Uses `torch.distributed.run` (torchrun)
- Master node determined from SLURM
- Port: 29500 (default)
- Backend: NCCL (for GPUs)

### Troubleshooting

**If jobs fail:**
1. Check `results/ddp_*/slurm_*.err` for errors
2. Verify NCCL configuration
3. Check network interface (InfiniBand vs Ethernet)
4. Ensure all nodes can communicate

**If scaling is poor:**
1. Check GPU utilization (should be >80%)
2. Check communication overhead (NCCL logs)
3. Verify batch size is appropriate
4. Check data loading bottleneck
