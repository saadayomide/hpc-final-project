#!/bin/bash
#SBATCH --job-name=dcrnn-param-workers
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/project/user42/hpc-final-project/results/param_workers_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/param_workers_%j.err
#SBATCH --array=0-3

# Parameter Sensitivity: Data Loader Workers Sweep
# This script runs multiple jobs with different num-workers values

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

# Define num-workers values to test
NUM_WORKERS_VALUES=(2 4 6 8)
NUM_WORKERS=${NUM_WORKERS_VALUES[$SLURM_ARRAY_TASK_ID]}

# Results directory for this parameter value
RESULTS_DIR="${PROJECT_DIR}/results/param_workers_${NUM_WORKERS}_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

DATA_DIR="${PROJECT_DIR}/data"

# Fixed parameters
NUM_NODES=100
BATCH_SIZE=32
EPOCHS=3  # Fewer epochs for parameter sweep

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "=== Parameter Sensitivity: Data Loader Workers ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num workers per GPU: ${NUM_WORKERS}"
echo "Nodes: ${SLURM_NNODES}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
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
  --num-workers ${NUM_WORKERS} \
  --results "${RESULTS_DIR}" \
  --seed 42

# Save job metadata
cat > "${RESULTS_DIR}/job_metadata.txt" << EOF
Experiment: Parameter Sensitivity - Data Loader Workers
Parameter: num_workers
Value: ${NUM_WORKERS}
Nodes: ${SLURM_NNODES}
GPUs per node: ${SLURM_GPUS_PER_NODE}
Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))
Batch size per GPU: ${BATCH_SIZE}
Problem size (num_nodes): ${NUM_NODES}
Epochs: ${EPOCHS}
Job ID: ${SLURM_JOB_ID}
Array Task ID: ${SLURM_ARRAY_TASK_ID}
EOF

# Save sacct summary
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"

echo "Parameter sweep (num_workers=${NUM_WORKERS}) complete. Results: ${RESULTS_DIR}"

