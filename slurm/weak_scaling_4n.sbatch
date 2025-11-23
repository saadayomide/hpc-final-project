#!/bin/bash
#SBATCH --job-name=dcrnn-weak-4n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/project/user42/hpc-final-project/results/weak_scaling_4n_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/weak_scaling_4n_%j.err

# Weak Scaling: Scale problem size with number of nodes
# This script runs on 4 nodes (4x problem size)

# Set environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

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
RESULTS_DIR="${PROJECT_DIR}/results/weak_scaling_4n_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

DATA_DIR="${PROJECT_DIR}/data"

# Weak scaling: problem size scales with number of nodes
# 4 nodes: 4x problem size (400 graph nodes)
NUM_NODES=400        # 4x base problem size
BATCH_SIZE=32        # Batch size per GPU (same per GPU)
EPOCHS=5

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "=== Weak Scaling: 4 Nodes ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Master node: ${MASTER_ADDR}"
echo "Problem size: ${NUM_NODES} graph nodes (4x base for 4 nodes)"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Run distributed training
srun --ntasks-per-node=4 --gpus-per-task=1 \
  ./run.sh python -m torch.distributed.run \
  --nproc_per_node=4 \
  --nnodes=${SLURM_NNODES} \
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
  --seed 42

# Save job metadata
cat > "${RESULTS_DIR}/job_metadata.txt" << EOF
Experiment: Weak Scaling
Nodes: 4
GPUs per node: 4
Total GPUs: 16
Problem size (num_nodes): ${NUM_NODES} (4x base)
Batch size per GPU: ${BATCH_SIZE}
Total batch size: $((BATCH_SIZE * 16))
Epochs: ${EPOCHS}
Job ID: ${SLURM_JOB_ID}
Master node: ${MASTER_ADDR}
EOF

# Save sacct summary
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"

echo "Weak scaling (4 nodes) complete. Results: ${RESULTS_DIR}"

