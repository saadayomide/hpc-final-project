#!/bin/bash
#SBATCH --job-name=dcrnn-profile-1n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/project/user42/hpc-final-project/results/profiling/profile_1n_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/profiling/profile_1n_%j.err

# Phase 3: Profiling & Bottleneck Identification
# 1-node run (baseline) with GPU profiling using Nsight Systems

# Set environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO  # More verbose for profiling
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export CUDA_LAUNCH_BLOCKING=0  # Allow async kernels (profiler handles sync)

# Set filesystem paths
PROJECT_DIR="/project/user42/hpc-final-project"
SCRATCH_DIR="/scratch/user42/hpc-final-project"
HOME_DIR="${SLURM_SUBMIT_DIR}/.."

cd "${HOME_DIR}"

# Create directories
mkdir -p "${PROJECT_DIR}/results/profiling/1node"
mkdir -p "${PROJECT_DIR}/results/profiling/1node/nsys"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${SCRATCH_DIR}/tmp"

# Results directory for this profiling run
PROFILE_DIR="${PROJECT_DIR}/results/profiling/1node"
RESULTS_DIR="${PROFILE_DIR}/run_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/nsys"

DATA_DIR="${PROJECT_DIR}/data"

# Configuration for profiling run
# Use enough epochs/steps for profiler to capture meaningful data
# But not too long to waste time
NUM_NODES=100        # Graph size
BATCH_SIZE=32        # Batch size per GPU
EPOCHS=3             # Enough for profiling, not too long
NUM_STEPS_PROFILE=50 # Number of steps to profile (to avoid huge traces)

echo "=========================================="
echo "Phase 3: Profiling - 1 Node (Baseline)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Problem size: ${NUM_NODES} graph nodes, batch_size=${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Profile directory: ${RESULTS_DIR}"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "WARNING: Nsight Systems (nsys) not found. Attempting to load module..."
    # Try to load module (adjust based on your HPC system)
    # module load nvidia-nsight-systems 2>/dev/null || true
fi

# GPU Profiling with Nsight Systems
# --trace=cuda,nvtx,nccl,cublas,cudnn - trace GPU kernels, NVTX ranges, NCCL, cuBLAS, cuDNN
# --trace=cpu - trace CPU activity
# --sample=cpu - CPU sampling
# --stats=true - print summary statistics
# --output= - output file path (will add .nsys-rep extension)
# --force-overwrite=true - overwrite existing files
# --capture-range=cudaProfilerApi - capture only the profiled region
# --stop-on-exit=true - stop profiling when process exits

echo "Starting Nsight Systems profiling..."
echo ""

# Profile the training run
# Use nsys profile to capture timeline of GPU, CPU, and NCCL activity
nsys profile \
  --trace=cuda,nvtx,nccl,cublas,cudnn,osrt \
  --trace=cpu \
  --sample=cpu \
  --stats=true \
  --output="${RESULTS_DIR}/nsys/dcrnn_1node_profile" \
  --force-overwrite=true \
  --stop-on-exit=true \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  srun --ntasks-per-node=4 --gpus-per-task=1 \
    ./run.sh python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    src/train.py \
    --data "${DATA_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --num-nodes ${NUM_NODES} \
    --precision bf16 \
    --num-workers 6 \
    --results "${RESULTS_DIR}" \
    --seed 42 \
    --profile-steps ${NUM_STEPS_PROFILE}

PROFILE_EXIT=$?

if [ $PROFILE_EXIT -eq 0 ]; then
    echo ""
    echo "Nsight Systems profiling completed successfully."
    echo "Profile report: ${RESULTS_DIR}/nsys/dcrnn_1node_profile.nsys-rep"
    
    # Generate summary report (if nsys stats is available)
    if [ -f "${RESULTS_DIR}/nsys/dcrnn_1node_profile.nsys-rep" ]; then
        echo "Generating summary statistics..."
        nsys stats --report gputrace --report cudaapisummary --report cudaapitime \
          --report ncclapisummary --report osrtrace --format csv \
          "${RESULTS_DIR}/nsys/dcrnn_1node_profile.nsys-rep" \
          > "${RESULTS_DIR}/nsys/summary_stats.csv" 2>&1 || true
    fi
else
    echo "WARNING: Profiling encountered an error (exit code: $PROFILE_EXIT)"
    echo "Training may have failed. Check logs."
fi

# Save job metadata
cat > "${RESULTS_DIR}/job_metadata.txt" << EOF
Phase: 3 - Profiling & Bottleneck Identification
Scenario: 1-node (baseline)
Nodes: 1
GPUs per node: 4
Total GPUs: 4
Problem size (num_nodes): ${NUM_NODES}
Batch size per GPU: ${BATCH_SIZE}
Total batch size: $((BATCH_SIZE * 4))
Epochs: ${EPOCHS}
Profile steps: ${NUM_STEPS_PROFILE}
Job ID: ${SLURM_JOB_ID}
Profiler: Nsight Systems (nsys)
Profile report: nsys/dcrnn_1node_profile.nsys-rep
Timestamp: $(date)
EOF

# Save sacct summary
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"

# Capture nvidia-smi output (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Capturing GPU information..."
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv > "${RESULTS_DIR}/gpu_info.txt" 2>&1 || true
fi

echo ""
echo "=========================================="
echo "1-node profiling complete!"
echo "Results: ${RESULTS_DIR}"
echo "Profile report: ${RESULTS_DIR}/nsys/dcrnn_1node_profile.nsys-rep"
echo "=========================================="

