#!/bin/bash
#SBATCH --job-name=dcrnn-profile-2n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/project/user42/hpc-final-project/results/profiling/profile_2n_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/profiling/profile_2n_%j.err

# Phase 3: Profiling & Bottleneck Identification
# 2-node run (where scaling still looks good) with GPU profiling using Nsight Systems

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
mkdir -p "${PROJECT_DIR}/results/profiling/2node"
mkdir -p "${PROJECT_DIR}/results/profiling/2node/nsys"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${SCRATCH_DIR}/tmp"

# Results directory for this profiling run
PROFILE_DIR="${PROJECT_DIR}/results/profiling/2node"
RESULTS_DIR="${PROFILE_DIR}/run_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/nsys"

DATA_DIR="${PROJECT_DIR}/data"

# Configuration for profiling run
NUM_NODES=100        # Same graph size as baseline (strong scaling)
BATCH_SIZE=32        # Same batch size per GPU
EPOCHS=3             # Enough for profiling
NUM_STEPS_PROFILE=50 # Number of steps to profile

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "=========================================="
echo "Phase 3: Profiling - 2 Nodes"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Master address: ${MASTER_ADDR}"
echo "Problem size: ${NUM_NODES} graph nodes, batch_size=${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Profile directory: ${RESULTS_DIR}"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "WARNING: Nsight Systems (nsys) not found. Attempting to load module..."
    # module load nvidia-nsight-systems 2>/dev/null || true
fi

# GPU Profiling with Nsight Systems
# For multi-node, we profile on rank 0 to avoid duplicate traces
# All ranks participate in training, but only rank 0 writes the profile

echo "Starting Nsight Systems profiling (2 nodes, 8 GPUs)..."
echo "NOTE: Profiling only on rank 0 to avoid duplicate traces."
echo ""

# Profile the training run
# For multi-node DDP, profile only on the master process (rank 0)
if [ "${SLURM_NODEID}" -eq 0 ] && [ "${SLURM_PROCID}" -eq 0 ]; then
    # This is rank 0 - run with profiling
    nsys profile \
      --trace=cuda,nvtx,nccl,cublas,cudnn,osrt \
      --trace=cpu \
      --sample=cpu \
      --stats=true \
      --output="${RESULTS_DIR}/nsys/dcrnn_2node_profile" \
      --force-overwrite=true \
      --stop-on-exit=true \
      --cuda-memory-usage=true \
      --cuda-um-cpu-page-faults=true \
      --cuda-um-gpu-page-faults=true \
      srun --ntasks-per-node=4 --gpus-per-task=1 \
        ./run.sh python -m torch.distributed.run \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=${SLURM_NODEID} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/train.py \
        --data "${DATA_DIR}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --num-nodes ${NUM_NODES} \
        --precision bf16 \
        --num-workers 6 \
        --results "${RESULTS_DIR}" \
        --seed 42 \
        --monitor-gpu \
        --monitor-cpu
else
    # Other ranks - run without profiling
    srun --ntasks-per-node=4 --gpus-per-task=1 \
      ./run.sh python -m torch.distributed.run \
      --nproc_per_node=4 \
      --nnodes=2 \
      --node_rank=${SLURM_NODEID} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      src/train.py \
      --data "${DATA_DIR}" \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --num-nodes ${NUM_NODES} \
      --precision bf16 \
      --num-workers 6 \
      --results "${RESULTS_DIR}" \
      --seed 42 \
      --monitor-gpu \
      --monitor-cpu
fi

PROFILE_EXIT=$?

if [ $PROFILE_EXIT -eq 0 ] && [ -f "${RESULTS_DIR}/nsys/dcrnn_2node_profile.nsys-rep" ]; then
    echo ""
    echo "Nsight Systems profiling completed successfully."
    echo "Profile report: ${RESULTS_DIR}/nsys/dcrnn_2node_profile.nsys-rep"
    
    # Generate summary report
    echo "Generating summary statistics..."
    nsys stats --report gputrace --report cudaapisummary --report cudaapitime \
      --report ncclapisummary --report osrtrace --format csv \
      "${RESULTS_DIR}/nsys/dcrnn_2node_profile.nsys-rep" \
      > "${RESULTS_DIR}/nsys/summary_stats.csv" 2>&1 || true
fi

# Save job metadata (only on rank 0)
if [ "${SLURM_NODEID}" -eq 0 ] && [ "${SLURM_PROCID}" -eq 0 ]; then
    cat > "${RESULTS_DIR}/job_metadata.txt" << EOF
Phase: 3 - Profiling & Bottleneck Identification
Scenario: 2-node (scaling still good)
Nodes: 2
GPUs per node: 4
Total GPUs: 8
Problem size (num_nodes): ${NUM_NODES}
Batch size per GPU: ${BATCH_SIZE}
Total batch size: $((BATCH_SIZE * 8))
Epochs: ${EPOCHS}
Profile steps: ${NUM_STEPS_PROFILE}
Job ID: ${SLURM_JOB_ID}
Master address: ${MASTER_ADDR}
Profiler: Nsight Systems (nsys)
Profile report: nsys/dcrnn_2node_profile.nsys-rep
Timestamp: $(date)
EOF

    # Save sacct summary
    sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"
fi

echo ""
echo "=========================================="
echo "2-node profiling complete!"
echo "Results: ${RESULTS_DIR}"
if [ -f "${RESULTS_DIR}/nsys/dcrnn_2node_profile.nsys-rep" ]; then
    echo "Profile report: ${RESULTS_DIR}/nsys/dcrnn_2node_profile.nsys-rep"
fi
echo "=========================================="

