#!/bin/bash
#SBATCH --job-name=dcrnn-prod-ddp
#SBATCH --account=def-sponsor00
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=results/production_ddp_%j.out
#SBATCH --error=results/production_ddp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=user@example.com

# ============================================================
# DCRNN Production Multi-Node DDP Training
# ============================================================

set -e  # Exit on error

echo "=============================================="
echo "DCRNN Production Multi-Node DDP Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks: ${SLURM_NTASKS}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo ""

# Production environment
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo ""

# Create results directory
RESULTS_DIR="./results/production_ddp_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Get master node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Verify GPU access on all nodes
echo "Verifying GPU access on all nodes..."
srun --ntasks-per-node=1 bash -c "
    echo \"Node \${SLURMD_NODENAME}:\"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ''
" | tee "${RESULTS_DIR}/gpu_info.txt"

# Verify container
if [ ! -f "env/project.sif" ]; then
    echo "ERROR: Container not found. Building..."
    ./run.sh build
fi

# Run distributed training
echo ""
echo "Starting production distributed training..."
echo "=========================================="

srun --ntasks-per-node=4 --gpus-per-task=1 \
    ./run.sh exec --nv python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --data ./data \
    --epochs 100 \
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
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Production training completed successfully"
else
    echo "❌ Production training failed with exit code: ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================="

# Save job information
sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList \
    > "${RESULTS_DIR}/job_info.txt" 2>/dev/null

exit $EXIT_CODE
