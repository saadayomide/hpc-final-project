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
#SBATCH --output=results/ddp_%j.out
#SBATCH --error=results/ddp_%j.err

# Set environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0  # Uncomment if using InfiniBand

# Set working directory
cd "${SLURM_SUBMIT_DIR}/.."

# Create results directory for this job
RESULTS_DIR="./results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

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
  --data ./data \
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

