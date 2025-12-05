# Multi-Node DDP Training Execution Guide

## Overview

This guide explains how to execute and verify multi-node distributed training using PyTorch DDP (Distributed Data Parallel) with the CPU backend (gloo).

## Quick Start

### Option 1: Using Helper Script (Recommended)

```bash
# Basic execution (2 nodes, 10 epochs)
./scripts/run_multinode.sh

# Custom parameters
./scripts/run_multinode.sh --nodes 4 --epochs 20 --partition gpu-node --account your-account
```

### Option 2: Direct SLURM Submission

```bash
# Submit the job
sbatch slurm/ddp_2node_cpu.sbatch

# Monitor
squeue -u $USER
tail -f results/ddp_cpu_*.out
```

## Verification

After the job completes, verify the results:

```bash
# Automatic verification
./scripts/verify_multinode.sh [job_id_or_results_dir]

# Example: Verify by job ID
./scripts/verify_multinode.sh 12345

# Example: Verify by results directory
./scripts/verify_multinode.sh results/ddp_cpu_12345
```

## What Gets Verified

The verification script checks:

1. **Required Files:**
   - `metrics.csv` - Training metrics per epoch
   - `sacct_summary.txt` - SLURM accounting information

2. **Job Status:**
   - Job completion status (COMPLETED/FAILED/CANCELLED)
   - Number of nodes used
   - Elapsed time

3. **Training Metrics:**
   - Number of epochs completed
   - Final training/validation loss
   - Final MAE and RMSE
   - Throughput (samples/second)

4. **DDP Configuration:**
   - DDP initialization confirmed
   - World size (number of nodes)
   - Backend (gloo for CPU)

## Expected Results

### Successful Execution

A successful multi-node run should produce:

```
results/ddp_cpu_<JOB_ID>/
├── metrics.csv              # Training metrics
├── sacct_summary.txt        # SLURM accounting
├── checkpoint_latest.pth    # Model checkpoint (if saved)
└── (optional monitoring files)
```

### Metrics CSV Format

The `metrics.csv` file contains:
- `epoch` - Epoch number
- `train_loss` - Training loss
- `val_loss` - Validation loss
- `val_mae` - Validation Mean Absolute Error (mph)
- `val_rmse` - Validation Root Mean Squared Error (mph)
- `epoch_time_sec` - Time per epoch (seconds)
- `throughput_samples_per_sec` - Training throughput
- `data_load_time_sec` - Data loading time
- `compute_time_sec` - Computation time
- `gpu_util_avg` - Average GPU utilization (if monitored)
- `cpu_util_avg` - Average CPU utilization (if monitored)

## Troubleshooting

### Job Fails to Start

**Issue:** Job stays in PENDING state
- **Solution:** Check partition availability: `sinfo -p gpu-node`
- **Solution:** Check account/partition settings in SLURM script

### DDP Initialization Fails

**Issue:** "Connection refused" or "Address already in use"
- **Solution:** Master port may be in use, try changing `MASTER_PORT` in script
- **Solution:** Check firewall/network settings between nodes

### Training Hangs

**Issue:** Job runs but no progress
- **Solution:** Check output file: `tail -f results/ddp_cpu_<JOB_ID>.out`
- **Solution:** Check error file: `cat results/ddp_cpu_<JOB_ID>.err`
- **Solution:** Verify all nodes can communicate: `srun --nodes=2 hostname`

### No Results Generated

**Issue:** Job completes but no metrics.csv
- **Solution:** Check if rank 0 process completed successfully
- **Solution:** Verify output directory permissions
- **Solution:** Check for errors in output file

## Technical Details

### CPU Backend (gloo)

The multi-node training uses the `gloo` backend for CPU-based distributed training. This is necessary when:
- GPU DDP (NCCL) has compatibility issues
- Running on CPU-only nodes
- Testing distributed training without GPU resources

### Key Differences from GPU DDP

1. **Backend:** `gloo` instead of `nccl`
2. **Device:** CPU instead of CUDA
3. **DDP Wrapping:** No `device_ids` parameter
4. **Performance:** Slower but functional for verification

### Environment Variables

The SLURM script sets:
- `BACKEND=gloo` - Use CPU backend
- `MASTER_ADDR` - Master node hostname
- `MASTER_PORT=29500` - Communication port
- `WORLD_SIZE` - Number of nodes
- `RANK` - Process rank (0, 1, 2, ...)
- `LOCAL_RANK` - Local rank (always 0 for 1 task per node)

## Performance Expectations

### CPU Training Performance

- **Throughput:** ~50-100 samples/second per node (CPU)
- **Scaling:** Near-linear up to 4 nodes (communication overhead is low)
- **Efficiency:** 85-95% parallel efficiency for CPU training

### Comparison to GPU

- CPU training is 10-50× slower than GPU
- But sufficient for verifying multi-node functionality
- Can be used as fallback when GPU DDP fails

## Next Steps

After successful multi-node execution:

1. **Document Results:** Add to `EXECUTION_SUMMARY.md`
2. **Run Profiling:** Execute profiling jobs (Task 2)
3. **Scaling Experiments:** Run strong/weak scaling (Task 6)
4. **Bottleneck Analysis:** Analyze profiling results (Task 2)

## References

- [PyTorch DDP Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- Project `docs/DDP_TROUBLESHOOTING.md` for GPU DDP issues

