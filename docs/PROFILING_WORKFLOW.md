# Profiling Workflow Guide

## Overview

This guide explains the complete workflow for executing profiling runs and generating bottleneck analysis for the DCRNN training system.

## Quick Start

```bash
# 1. Configure profiling
./scripts/configure_profiling.sh

# 2. Run profiling jobs
source scripts/profiling_config.sh
./scripts/run_profiling.sh

# 3. Wait for jobs to complete
squeue -u $USER

# 4. Aggregate and analyze results
./scripts/aggregate_profiling_results.sh

# 5. Review bottleneck analysis
cat docs/BOTTLENECK_ANALYSIS.md
```

## Detailed Workflow

### Step 1: Configuration

First, configure your SLURM account and partition:

```bash
./scripts/configure_profiling.sh
```

Or manually:

```bash
export SLURM_ACCOUNT='your-account'
export SLURM_PARTITION='gpu-node'
echo "export SLURM_ACCOUNT='your-account'" > scripts/profiling_config.sh
echo "export SLURM_PARTITION='gpu-node'" >> scripts/profiling_config.sh
```

### Step 2: Submit Profiling Jobs

Submit profiling jobs for 1-node, 2-node, and 4-node configurations:

```bash
source scripts/profiling_config.sh
./scripts/run_profiling.sh
```

This will:
- Submit 3 profiling jobs (1-node, 2-node, 4-node)
- Save job IDs to `results/profiling/run_<timestamp>/job_ids.txt`
- Create results directories for each scenario

**Note:** For CPU DDP profiling (if GPU DDP fails), use:
```bash
sbatch slurm/profile_cpu_2node.sbatch
```

### Step 3: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output (replace JOB_ID with actual job ID)
tail -f results/profiling/profile_*_<JOB_ID>.out

# Wait for all jobs to complete
for job in $(cat results/profiling/run_*/job_ids.txt); do
    srun --jobid=$job echo "Job $job completed"
done
```

### Step 4: Analyze Individual Results

After each job completes, analyze its results:

```bash
# For 1-node
python scripts/analyze_profiling.py --results results/profiling/1node/run_<JOB_ID>

# For 2-node
python scripts/analyze_profiling.py --results results/profiling/2node/run_<JOB_ID>

# For 4-node
python scripts/analyze_profiling.py --results results/profiling/4node/run_<JOB_ID>
```

This generates:
- `profiling_report.txt` - Text summary
- `profiling_analysis.json` - Machine-readable data
- `profiling_analysis.png` - Visualization plots

### Step 5: Aggregate Results

After all jobs complete, aggregate results and generate bottleneck analysis:

```bash
./scripts/aggregate_profiling_results.sh [base_results_dir]
```

This will:
- Analyze all scenarios
- Generate aggregated bottleneck analysis document
- Create `docs/BOTTLENECK_ANALYSIS.md`

### Step 6: Review Results

```bash
# View bottleneck analysis
cat docs/BOTTLENECK_ANALYSIS.md

# View individual reports
cat results/profiling/*/run_*/profiling_report.txt

# View plots
ls results/profiling/*/run_*/profiling_analysis.png
```

## Output Files

### Per-Scenario Results

Each profiling run produces:

```
results/profiling/<scenario>/run_<JOB_ID>/
├── metrics.csv                    # Training metrics
├── gpu_monitor.csv                # GPU utilization (if GPU available)
├── cpu_monitor.csv                # CPU utilization
├── profiling_report.txt           # Text analysis report
├── profiling_analysis.json       # Machine-readable analysis
├── profiling_analysis.png         # Visualization plots
├── profiling_analysis.svg         # Vector plots
├── sacct_summary.txt              # SLURM accounting
├── job_metadata.txt               # Job configuration
└── nsys/                          # Nsight Systems traces (if GPU)
    ├── dcrnn_*node_profile.nsys-rep
    └── summary_stats.csv
```

### Aggregated Analysis

The bottleneck analysis document:

```
docs/BOTTLENECK_ANALYSIS.md
```

Contains:
- Executive summary
- Top bottlenecks identified
- Per-scenario analysis
- Time breakdown tables
- Scaling analysis
- Recommendations

## Understanding the Analysis

### Time Breakdown

The analysis breaks down training time into:

- **Compute:** Actual model computation (forward/backward pass)
- **Data Loading:** Data loading and preprocessing
- **Communication:** Gradient synchronization (NCCL/gloo)
- **Other Overhead:** Sync, I/O, and other overhead

### Bottleneck Types

1. **GPU Underutilization:** GPU < 70% utilized
   - Indicates: Data loading or communication bottleneck
   - Solution: Increase batch size, optimize data loading

2. **Data Loading Bottleneck:** > 30% time spent loading
   - Indicates: Data pipeline too slow
   - Solution: Increase num_workers, use pin_memory, prefetch

3. **Communication Overhead:** > 20% overhead
   - Indicates: Gradient sync taking too long
   - Solution: Gradient accumulation, optimize all-reduce

4. **Memory Pressure:** > 90% GPU memory used
   - Indicates: Risk of OOM errors
   - Solution: Reduce batch size, use mixed precision, gradient checkpointing

## Troubleshooting

### Jobs Fail to Submit

**Issue:** Configuration errors
- **Solution:** Run `./scripts/configure_profiling.sh` and verify account/partition

### No Profiling Data

**Issue:** Monitoring not enabled
- **Solution:** Ensure `--monitor-gpu` and `--monitor-cpu` flags are used

### Analysis Fails

**Issue:** Missing metrics files
- **Solution:** Check that training completed successfully
- **Solution:** Verify `metrics.csv` exists in results directory

### Nsight Systems Not Available

**Issue:** `nsys` command not found
- **Solution:** Profiling will still work with monitoring data
- **Solution:** Nsight Systems traces are optional (for detailed GPU analysis)

## CPU DDP Profiling

For CPU-based distributed training (gloo backend):

```bash
# Submit CPU profiling job
sbatch slurm/profile_cpu_2node.sbatch

# Analyze results
python scripts/analyze_profiling.py --results results/profiling/cpu_2node_<JOB_ID>
```

CPU profiling focuses on:
- CPU utilization
- Memory usage
- Communication overhead (gloo)
- Data loading performance

## Next Steps

After profiling:

1. **Review Bottleneck Analysis:** `docs/BOTTLENECK_ANALYSIS.md`
2. **Implement Optimizations:** Based on recommendations
3. **Re-profile:** Verify improvements
4. **Document:** Update paper with findings

## References

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Profiling Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- Project `docs/PHASE3_PROFILING.md` for detailed profiling methodology

