# Phase 3: Profiling & Bottleneck Identification

## Overview

Phase 3 focuses on identifying where time is actually spent in your DCRNN training application (CPU and/or GPU) to guide targeted optimizations.

**Goal:** Figure out what's slowing you down (compute vs memory vs I/O vs communication)

**Think of it as:** "We know how fast it is. Now we want to know what's slowing it down."

---

## Quick Start

### 1. Configure Account and Partition

**First time setup:**
```bash
cd /home/user42/hpc-final-project
./scripts/configure_profiling.sh
```

**Or manually set:**
```bash
export SLURM_ACCOUNT='def-sponsor00'
export SLURM_PARTITION='gpu-node'
```

### 2. Run Profiling Jobs

```bash
source scripts/profiling_config.sh
./scripts/run_profiling.sh
```

This submits profiling jobs for:
- 1-node run (baseline)
- 2-node run (scaling still good)  
- 4-node run (scaling degrades)

### 3. After Jobs Complete, Analyze

```bash
python scripts/analyze_profiling.py --results results/profiling
cat results/profiling/bottleneck_summary.txt
```

### 4. View Profiling Reports

- Download `.nsys-rep` files from cluster to local machine
- Open with Nsight Systems GUI
- Examine timeline views for GPU, CPU, and NCCL activity

---

## What Phase 3 Does

### 1. Profile Representative Scenarios

We profile representative runs:
- **1-node run** (baseline)
- **2-node run** (where scaling still looks good)
- **4-node run** (where scaling starts to degrade)

**Criteria:**
- Not too short (profilers need some work to record)
- Not too long (to avoid wasting time/budget)

### 2. Use Profiling Tools to Answer Questions

#### For GPU-heavy Workloads (PyTorch, DDP)

We use **Nsight Systems** to see:
- **GPU utilization** - How busy are the GPUs?
- **Time breakdown:**
  - Compute (kernels) - actual computation
  - Communication (NCCL all-reduce, etc.) - gradient synchronization
  - Data loading / host-side preprocessing - data pipeline
  - I/O (reading/writing data, checkpoints) - disk operations

**Nsight Systems provides:**
- Timeline view of GPU, CPU, and NCCL activity
- Identification of bottlenecks
- Memory transfer analysis
- Low overhead (suitable for full training runs)

#### For CPU-heavy Workloads (Optional)

We use **perf** to analyze:
- Time in computation vs communication (MPI) vs I/O
- Cache/memory behavior

### 3. Identify Bottleneck Patterns

| Bottleneck Type | Indicators | What to Look For |
|----------------|------------|------------------|
| **Compute-bound** | GPUs/CPUs nearly 100% busy, little idle time | High GPU utilization, minimal communication overhead |
| **Memory-bound** | Lots of stalls waiting for memory, low arithmetic intensity | High memory bandwidth utilization, kernel stalls |
| **I/O-bound** | Big gaps waiting for data to load or checkpoints to write | Large idle periods, high disk I/O time |
| **Communication-bound** | Large chunks of time in MPI/NCCL collectives, especially as node count grows | High NCCL time, grows with node count |
| **Data-loader-bound** | GPUs idle while data loader is decoding/augmenting/reading from disk | GPU idle between batches, low overlap with compute |

**Your job:** Turn profiler timelines into simple statements like:

> "At 4 nodes, 30–40% of the step time is spent in gradient synchronization (NCCL all-reduce), which limits strong scaling beyond 2 nodes."

OR

> "GPU utilization drops to 50% because the data loader cannot keep up; most of the idle time is between batches."

### 4. Capture Evidence in a Reusable Way

**Structure:**
```
results/profiling/
├── 1node/run_[JOB_ID]/
│   ├── nsys/dcrnn_1node_profile.nsys-rep
│   ├── nsys/summary_stats.csv
│   └── job_metadata.txt
├── 2node/run_[JOB_ID]/
└── 4node/run_[JOB_ID]/
```

**Summary document:**
- Fill out `results/profiling/BOTTLENECK_ANALYSIS_TEMPLATE.md`
- Answer:
  - What are the top 1-2 bottlenecks?
  - How do they change as you go from 1 → 2 → 4 nodes?
  - What seems to limit further scaling?

---

## Files and Scripts

### Profiling Scripts

- `slurm/profile_1node.sbatch` - 1-node baseline profiling
- `slurm/profile_2node.sbatch` - 2-node profiling
- `slurm/profile_4node.sbatch` - 4-node profiling
- `slurm/profile_cpu_1node.sbatch` - Optional CPU profiling

### Automation Scripts

- `scripts/run_profiling.sh` - Submits all profiling jobs
- `scripts/analyze_profiling.py` - Analyzes results and generates summaries
- `scripts/configure_profiling.sh` - Interactive configuration helper
- `scripts/profiling_config.sh` - Configuration file (auto-generated)

---

## Expected Outcomes

By the end of Phase 3, you should have:

1. ✅ **At least one GPU profile** for each scenario (1-node, 2-node, 4-node)
   - Nsight Systems `.nsys-rep` files
   - Summary statistics CSV files

2. ✅ **A clear written explanation** of what's slowing you down:
   - Compute vs Memory vs I/O vs Communication
   - Quantified time breakdown

3. ✅ **Concrete clues** about what to improve in Phase 4:
   - Top 1-2 bottlenecks identified
   - How bottlenecks change with node count
   - What limits further scaling

---

## Key Questions Phase 3 Answers

1. **What is the primary bottleneck?**
   - Compute-bound? Memory-bound? I/O-bound? Communication-bound?

2. **How does the bottleneck change with scaling?**
   - Does communication overhead increase?
   - Does data loading become a bottleneck?
   - Does memory bandwidth become a limitation?

3. **What limits scaling beyond X nodes?**
   - Is it communication overhead?
   - Is it data loading capacity?
   - Is it memory bandwidth?

4. **What should we optimize in Phase 4?**
   - Based on bottleneck analysis, what optimizations will have the biggest impact?

---

## Troubleshooting

### Jobs fail to submit

- Check SLURM account and partition are correct in `scripts/profiling_config.sh`
- Verify file permissions: `chmod +x scripts/run_profiling.sh`
- Check account is valid: `sacctmgr show user $USER`

### Nsight Systems not found

- Check if available: `which nsys`
- Try loading module: `module load nvidia-nsight-systems`
- Check if installed in container: `./run.sh which nsys`

### Profile reports are too large

- Reduce epochs in profiling scripts
- Profile fewer steps
- Use `--capture-range=cudaProfilerApi` to profile specific regions

### Analysis script can't find results

- Check directory structure: `ls -la results/profiling/`
- Verify job_metadata.txt exists: `find results/profiling -name job_metadata.txt`
- Manually specify results directory: `python scripts/analyze_profiling.py --results results/profiling/1node/run_*`

---

## Next Steps

After completing Phase 3:

1. **Phase 4:** Implement targeted optimizations based on bottleneck findings
2. **Documentation:** Fill out bottleneck analysis template with detailed findings
3. **Validation:** Re-run profiling after Phase 4 optimizations to verify improvements

---

## References

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Profiling](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NCCL Profiling](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/profiling.html)
