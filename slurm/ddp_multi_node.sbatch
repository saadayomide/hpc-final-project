#!/bin/bash
#SBATCH --job-name=dcrnn-ddp
#SBATCH --account=def-sponsor00
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=../results/ddp_%j.out
#SBATCH --error=../results/ddp_%j.err

# ============================================================
# DCRNN Multi-Node DDP Training
# ============================================================

echo "=============================================="
echo "DCRNN Multi-Node DDP Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks: ${SLURM_NTASKS}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-4}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo ""

# Set environment variables
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# Uncomment if using InfiniBand
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5

# PyTorch distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo ""

# Create results directory
RESULTS_DIR="./results/ddp_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Log system info on each node
srun --ntasks-per-node=1 bash -c "
    echo \"Node \${SLURMD_NODENAME}:\"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ''
" | tee "${RESULTS_DIR}/gpu_info.txt"

# Run distributed training
echo ""
echo "Starting distributed training..."
echo "================================"

srun --ntasks-per-node=4 --gpus-per-task=1 \
    ./run.sh python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --data ./data \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-dim 64 \
    --num-layers 2 \
    --precision bf16 \
    --num-workers 4 \
    --results "${RESULTS_DIR}" \
    --seed 42 \
    --monitor-gpu \
    --monitor-cpu \
    --checkpoint-interval 10

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

# Save SLURM accounting information
echo ""
echo "SLURM Accounting Summary:"
sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList \
    | tee "${RESULTS_DIR}/sacct_summary.txt"

echo ""
echo "Results saved to: ${RESULTS_DIR}"
