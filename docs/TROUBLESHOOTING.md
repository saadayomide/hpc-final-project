# Troubleshooting Guide

## Common Issues and Solutions

### 1. DDP Initialization Failures

#### Error: "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"

**Cause:** Trying to use NCCL backend without CUDA/GPU access.

**Solutions:**
- Check GPU availability: `nvidia-smi`
- Verify container has GPU access: `apptainer exec --nv project.sif nvidia-smi`
- Use CPU training: `export BACKEND=gloo` before running
- Check SLURM GPU allocation: `srun --gres=gpu:1 nvidia-smi`

#### Error: "libcublas.so.11: cannot open shared object file"

**Cause:** CUDA libraries not available in container or environment.

**Solutions:**
- Rebuild container: `./run.sh build`
- Check LD_LIBRARY_PATH includes CUDA libs
- Verify container base image has CUDA support
- Use `--nv` flag with Apptainer: `apptainer exec --nv project.sif ...`

#### Error: "Timeout waiting for process group"

**Cause:** Network issues or nodes not communicating.

**Solutions:**
- Increase timeout in `setup_ddp()` call
- Check network connectivity between nodes
- Verify firewall settings
- Check SLURM node allocation: `squeue -j <job_id>`

### 2. Multi-Node Training Issues

#### Training hangs or times out

**Solutions:**
- Check all nodes are accessible: `srun --ntasks-per-node=1 hostname`
- Verify master node address is correct
- Check port availability (default 29500)
- Increase DDP timeout in code
- Check SLURM job output for errors

#### Inconsistent results across nodes

**Solutions:**
- Ensure same random seed on all nodes
- Verify data is properly distributed (check DistributedSampler)
- Check batch size consistency
- Verify all nodes use same model version

### 3. Performance Issues

#### Low GPU utilization

**Solutions:**
- Increase batch size
- Increase number of data loading workers
- Check for data loading bottlenecks
- Use mixed precision (bf16/fp16)
- Profile with monitoring tools

#### Poor scaling efficiency

**Solutions:**
- Check communication overhead (NCCL settings)
- Verify batch size scales with nodes
- Check for load imbalance
- Profile communication vs compute time
- Consider gradient accumulation for small batches

### 4. Data Loading Issues

#### "No such file or directory" for data files

**Solutions:**
- Generate sample data: `cd data && python generate_sample_data.py`
- Check data directory path in arguments
- Verify data files exist: `ls -la data/processed/`
- Check file permissions

### 5. Container Issues

#### Container build fails

**Solutions:**
- Check internet connectivity
- Verify base image is accessible
- Check disk space: `df -h`
- Try building with verbose output

#### Container runs but CUDA not available

**Solutions:**
- Use `--nv` flag: `apptainer exec --nv project.sif ...`
- Check host CUDA driver version
- Verify container CUDA version matches host
- Check `/dev/nvidia*` devices exist

## Debugging Tips

### Enable Verbose Logging

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
```

### Check Environment

```bash
# In container
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check backend
echo $BACKEND

# Check SLURM environment
env | grep SLURM
```

### Test DDP Setup

```bash
# Single node test
python src/train.py --epochs 1 --batch-size 4

# Multi-node test (2 nodes, 1 GPU each)
sbatch slurm/ddp_2node.sbatch
```

### Verify Results

```bash
# Check job output
cat results/ddp_*/out

# Check for errors
grep -i error results/ddp_*/*.err

# Verify metrics
cat results/baseline_*/metrics.csv
```

## Getting Help

1. Check job output files: `results/*.out` and `results/*.err`
2. Review SLURM accounting: `sacct -j <job_id> -l`
3. Check system logs if available
4. Review this troubleshooting guide
5. Check project documentation in `docs/`
