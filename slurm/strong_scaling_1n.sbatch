#!/bin/bash
#SBATCH --job-name=dcrnn-strong-1n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/project/user42/hpc-final-project/results/strong_scaling_1n_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/strong_scaling_1n_%j.err

# Strong Scaling: Fixed problem size, vary number of nodes
# This script runs on 1 node (baseline)

# Set environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN  # Reduce verbosity for scaling runs

# Set filesystem paths
PROJECT_DIR="/project/user42/hpc-final-project"
SCRATCH_DIR="/scratch/user42/hpc-final-project"
HOME_DIR="${SLURM_SUBMIT_DIR}/.."

cd "${HOME_DIR}"

# Create directories
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${SCRATCH_DIR}/tmp"

# Results directory for this scaling experiment
RESULTS_DIR="${PROJECT_DIR}/results/strong_scaling_1n_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

DATA_DIR="${PROJECT_DIR}/data"

# Fixed problem size for strong scaling
NUM_NODES=100        # Fixed graph size
BATCH_SIZE=32        # Fixed batch size per GPU
EPOCHS=5             # Enough epochs to get stable timing

echo "=== Strong Scaling: 1 Node ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Problem size: ${NUM_NODES} graph nodes, batch_size=${BATCH_SIZE}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Run training (single node, 4 GPUs)
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
  --seed 42

# Save job metadata
cat > "${RESULTS_DIR}/job_metadata.txt" << EOF
Experiment: Strong Scaling
Nodes: 1
GPUs per node: 4
Total GPUs: 4
Problem size (num_nodes): ${NUM_NODES}
Batch size per GPU: ${BATCH_SIZE}
Total batch size: $((BATCH_SIZE * 4))
Epochs: ${EPOCHS}
Job ID: ${SLURM_JOB_ID}
EOF

# Save sacct summary
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"

echo "Strong scaling (1 node) complete. Results: ${RESULTS_DIR}"

