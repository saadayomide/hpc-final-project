# System Information

This document describes the cluster system configuration where experiments are run.

## Node Types

### GPU Compute Nodes

| Specification | Value |
|---------------|-------|
| **Node Type** | GPU Compute Node |
| **CPU** | AMD EPYC 7742 (64 cores, 128 threads) |
| **CPU Clock** | 2.25 GHz base, 3.4 GHz boost |
| **Memory** | 256 GB DDR4-3200 |
| **GPUs** | 4× NVIDIA A100 40GB |
| **GPU Memory** | 40 GB HBM2e per GPU |
| **Local Storage** | 1.9 TB NVMe SSD |
| **Interconnect** | InfiniBand HDR (200 Gb/s) |
| **Network Topology** | Fat-tree |

### Login Nodes

| Specification | Value |
|---------------|-------|
| **Node Type** | Login/Interactive Node |
| **CPU** | AMD EPYC 7302 (16 cores) |
| **Memory** | 128 GB DDR4 |
| **Purpose** | Job submission, compilation, testing |

## Software Environment

### Container Runtime

| Component | Version |
|-----------|---------|
| **Apptainer** | 1.2.x |
| **Base Image** | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` |
| **Python** | 3.10.x |

### Deep Learning Stack

| Component | Version |
|-----------|---------|
| **PyTorch** | 2.1.0 |
| **CUDA** | 11.8 |
| **cuDNN** | 8.x |
| **NCCL** | 2.18.x |

### Modules (if using Alliance/EESSI)

```bash
# Example module load commands
module load StdEnv/2023
module load apptainer/1.2
module load cuda/11.8
module load openmpi/4.1.5
```

Record your actual modules:
```bash
# Run this and paste output below:
module list 2>&1 | tee modules_used.txt
```

### Python Dependencies

See `env/project.def` for container dependencies. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.0 | Deep learning framework |
| numpy | 1.24+ | Numerical computing |
| pandas | 2.0+ | Data manipulation |
| scipy | 1.11+ | Scientific computing |
| matplotlib | 3.7+ | Visualization |
| psutil | 5.9+ | System monitoring |
| tqdm | 4.65+ | Progress bars |

## Driver Versions

### NVIDIA Driver and CUDA

```bash
# Get versions with:
nvidia-smi
nvcc --version
```

| Component | Version |
|-----------|---------|
| **NVIDIA Driver** | 535.x+ |
| **CUDA Runtime** | 11.8 |
| **CUDA Compiler** | 11.8.x |

## SLURM Configuration

### Partitions

| Partition | Nodes | Max Time | Max GPUs | Notes |
|-----------|-------|----------|----------|-------|
| gpu | 8 | 24:00:00 | 32 | Default GPU partition |
| interactive | 2 | 04:00:00 | 4 | Interactive jobs |

### Resource Limits

| Resource | Limit |
|----------|-------|
| Max nodes per job | 8 |
| Max GPUs per user | 32 |
| Max walltime | 24 hours |
| Default memory/node | 256 GB |

### SLURM Commands Reference

```bash
# Check job status
squeue -u $USER

# Job details
scontrol show job <JOBID>

# Historical accounting
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS

# Node information
sinfo -p gpu

# Interactive session
srun --nodes=1 --gpus=1 --time=01:00:00 --pty bash
```

## Network Configuration

### Interconnect

| Specification | Value |
|---------------|-------|
| **Type** | NVIDIA Mellanox InfiniBand HDR |
| **Bandwidth** | 200 Gb/s per port |
| **Latency** | ~1 μs |
| **Topology** | Fat-tree |

### NCCL Configuration

```bash
# Environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5
```

## Storage

### Filesystem Layout

| Mount | Type | Quota | Purpose |
|-------|------|-------|---------|
| `/home` | NFS | 50 GB | Code, configs |
| `/project` | Lustre | 1 TB | Datasets, results |
| `/scratch` | Lustre | 10 TB | Temporary files |

### Lustre Striping

```bash
# Check file striping
lfs getstripe <file>

# Set striping for large files
lfs setstripe -c 4 <directory>

# Recommended for checkpoints
lfs setstripe -c 8 -S 4M results/checkpoints/
```

### I/O Best Practices

1. **Stage data to /scratch** before jobs
2. **Use Lustre striping** for large files
3. **Avoid small file operations** on Lustre
4. **Aggregate writes** where possible
5. **Clean /scratch** after jobs complete

## Environment Variables

### Required for DDP Training

```bash
# PyTorch Distributed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Performance
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
```

## Verification Commands

### Check GPU Access

```bash
# In container
./run.sh python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### Check NCCL

```bash
./run.sh python -c "
import torch.distributed as dist
print(f'NCCL available: {dist.is_nccl_available()}')
"
```

### Check Container

```bash
apptainer --version
apptainer exec env/project.sif python --version
apptainer exec env/project.sif nvidia-smi
```

## Notes

- Update this file with actual system information from your cluster
- Run verification commands and record outputs
- Document any cluster-specific workarounds
- Keep module versions consistent across team members
