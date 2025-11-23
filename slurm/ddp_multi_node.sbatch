#!/bin/bash
#SBATCH --job-name=dcrnn-ddp
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/project/user42/hpc-final-project/results/ddp_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/ddp_%j.err

# Set environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0  # Uncomment if using InfiniBand

# Set filesystem paths following HPC guidelines:
# /home - code only (repo location)
# /project - persistent data (datasets, results, checkpoints)
# /scratch - temporary files (job-specific temp outputs)
PROJECT_DIR="/project/user42/hpc-final-project"
SCRATCH_DIR="/scratch/user42/hpc-final-project"
HOME_DIR="${SLURM_SUBMIT_DIR}/.."

# Set working directory to home (code location)
cd "${HOME_DIR}"

# Create directories in /project for persistent data
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${SCRATCH_DIR}/tmp"

# Create results directory for this job in /project
RESULTS_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Use /project for data directory (persistent datasets)
DATA_DIR="${PROJECT_DIR}/data"

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: ${MASTER_ADDR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks: ${SLURM_NTASKS}"

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
  --epochs 1 \
  --batch-size 64 \
  --precision bf16 \
  --num-workers 6 \
  --results "${RESULTS_DIR}" \
  --seed 42 \
  --monitor-gpu \
  --monitor-cpu

# Save sacct summary
echo "Saving sacct summary..."
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"

echo "Job completed. Results saved to ${RESULTS_DIR}"

