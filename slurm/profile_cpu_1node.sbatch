#!/bin/bash
#SBATCH --job-name=dcrnn-profile-cpu-1n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/project/user42/hpc-final-project/results/profiling/profile_cpu_1n_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/profiling/profile_cpu_1n_%j.err

# Phase 3: CPU Profiling (optional, for additional analysis)
# CPU profiling using perf for 1-node baseline

# Set environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

# Set filesystem paths
PROJECT_DIR="/project/user42/hpc-final-project"
SCRATCH_DIR="/scratch/user42/hpc-final-project"
HOME_DIR="${SLURM_SUBMIT_DIR}/.."

cd "${HOME_DIR}"

# Create directories
mkdir -p "${PROJECT_DIR}/results/profiling/1node"
mkdir -p "${PROJECT_DIR}/results/profiling/1node/perf"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${SCRATCH_DIR}/tmp"

# Results directory
PROFILE_DIR="${PROJECT_DIR}/results/profiling/1node"
RESULTS_DIR="${PROFILE_DIR}/run_cpu_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/perf"

DATA_DIR="${PROJECT_DIR}/data"

# Configuration
NUM_NODES=100
BATCH_SIZE=32
EPOCHS=3

echo "=========================================="
echo "Phase 3: CPU Profiling - 1 Node"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Profile directory: ${RESULTS_DIR}"
echo ""

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo "ERROR: perf not found. CPU profiling skipped."
    exit 1
fi

# CPU Profiling with perf
# Record CPU events: cycles, instructions, cache-misses, branch-misses
# Use perf record to capture samples

echo "Starting CPU profiling with perf..."
echo ""

# Profile the training run with perf
# Note: perf needs to be run on each rank separately
# We'll profile rank 0 as representative

srun --ntasks-per-node=1 --gpus-per-task=1 \
  perf record -o "${RESULTS_DIR}/perf/rank0.data" \
  -g --call-graph dwarf \
  -e cycles,instructions,cache-misses,cache-references,branch-misses,branches \
  -F 99 \
  -- \
  ./run.sh python -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  src/train.py \
  --data "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --num-nodes ${NUM_NODES} \
  --precision bf16 \
  --num-workers 6 \
  --results "${RESULTS_DIR}" \
  --seed 42 &

# Wait for profiling to complete
wait

if [ -f "${RESULTS_DIR}/perf/rank0.data" ]; then
    echo ""
    echo "Perf profiling completed successfully."
    echo "Perf data: ${RESULTS_DIR}/perf/rank0.data"
    
    # Generate reports
    echo "Generating perf reports..."
    
    # Top functions
    perf report -i "${RESULTS_DIR}/perf/rank0.data" > "${RESULTS_DIR}/perf/report_top.txt" 2>&1 || true
    
    # Statistics
    perf stat -i "${RESULTS_DIR}/perf/rank0.data" report > "${RESULTS_DIR}/perf/statistics.txt" 2>&1 || true
    
    # Flamegraph data (if available)
    perf script -i "${RESULTS_DIR}/perf/rank0.data" > "${RESULTS_DIR}/perf/script.out" 2>&1 || true
fi

# Save job metadata
cat > "${RESULTS_DIR}/job_metadata.txt" << EOF
Phase: 3 - Profiling & Bottleneck Identification
Type: CPU Profiling (perf)
Scenario: 1-node (baseline)
Nodes: 1
GPUs per node: 4
Total GPUs: 4
Problem size (num_nodes): ${NUM_NODES}
Batch size per GPU: ${BATCH_SIZE}
Total batch size: $((BATCH_SIZE * 4))
Epochs: ${EPOCHS}
Job ID: ${SLURM_JOB_ID}
Profiler: perf
Perf data: perf/rank0.data
Timestamp: $(date)
EOF

echo ""
echo "=========================================="
echo "CPU profiling complete!"
echo "Results: ${RESULTS_DIR}"
echo "=========================================="

