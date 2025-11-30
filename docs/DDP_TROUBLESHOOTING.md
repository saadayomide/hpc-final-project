# DDP Troubleshooting Guide

## Issue: Multi-Node DDP Training Fails

### Problem
Multi-node distributed training fails with CUDA library compatibility errors:
```
ValueError: libcublas.so.*[0-9] not found
ProcessGroupNCCL is only supported with GPUs, no GPUs found!
```

### Root Cause
1. **CUDA Library Mismatch**: Cluster has CUDA 12.4, but pip-installed PyTorch expects different CUDA libraries
2. **No CUDA-enabled PyTorch Module**: The cluster doesn't provide a pre-built PyTorch module with CUDA support
3. **Container Build Issues**: Apptainer container build fails due to disk space constraints

### Solutions

#### Solution 1: CPU-Based Distributed Training (Immediate Workaround)

Use the `gloo` backend for CPU-based distributed training:

```bash
sbatch slurm/ddp_2node_cpu.sbatch
```

This uses the `gloo` backend which works on CPU and doesn't require CUDA libraries.

**Limitations:**
- Slower than GPU training
- No GPU acceleration
- Still demonstrates distributed training concepts

#### Solution 2: Fix CUDA PyTorch Installation

If you need GPU acceleration, try:

1. **Use conda instead of pip:**
```bash
module load python/3.11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. **Build Apptainer container on a system with more disk space:**
   - Build on a local machine or a system with sufficient disk space
   - Transfer the `.sif` file to the cluster
   - Use the container for training

3. **Request cluster admins to install PyTorch module:**
   - Contact cluster administrators
   - Request installation of a CUDA-enabled PyTorch module compatible with cluster CUDA version

#### Solution 3: Single-Node Multi-GPU Training

If multi-node fails, use single-node multi-GPU:

```bash
# Request multiple GPUs on one node
#SBATCH --nodes=1
#SBATCH --gres=gpu:2  # or 4

# Use DataParallel or single-node DDP
```

### Current Status

✅ **Working:**
- Single-node training (CPU or GPU)
- Baseline training completed successfully
- All analysis scripts and documentation

❌ **Not Working:**
- Multi-node GPU DDP (CUDA library issues)
- Apptainer container build (disk space)

✅ **Workaround Available:**
- CPU-based multi-node DDP (`slurm/ddp_2node_cpu.sbatch`)

### Testing CPU DDP

```bash
cd /home/user42/hpc-final-project
sbatch slurm/ddp_2node_cpu.sbatch
squeue -u user42
tail -f results/ddp_cpu_*.out
```

### Expected Results

CPU DDP will be slower but should demonstrate:
- Distributed data loading
- Gradient synchronization across nodes
- Scaling behavior (though slower than GPU)

### Future Improvements

1. Coordinate with cluster admins for CUDA PyTorch module
2. Build container on external system with more disk space
3. Use conda environment with proper CUDA support
4. Consider using Horovod as alternative to PyTorch DDP
