#!/bin/bash
#SBATCH --job-name=dcrnn-ddp-2n
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/ddp_%j.out
#SBATCH --error=results/ddp_%j.err

echo "=============================================="
echo "DCRNN Multi-Node DDP Training - 2 GPU Nodes"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"

# Set up environment
module purge
module load python/3.11 scipy-stack/2025a
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Project directory
PROJECT_DIR="/home/user42/hpc-final-project"
cd "${PROJECT_DIR}"

mkdir -p results
RESULTS_DIR="./results/ddp_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Install CUDA-enabled PyTorch if not already installed
echo "Setting up CUDA PyTorch on all nodes..."
srun --ntasks-per-node=1 bash -c '
    # Check if CUDA torch is already available
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Installing PyTorch with CUDA on $(hostname)..."
        pip uninstall -y torch 2>/dev/null
        pip install --user torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
    fi
    python -c "import torch; print(f\"$(hostname): CUDA={torch.cuda.is_available()}, Device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
'

# Get master node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=${SLURM_NNODES}

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""
echo "Starting distributed training..."
echo "================================"

srun --export=ALL bash -c '
export RANK=${SLURM_PROCID}
export LOCAL_RANK=0
echo "Node $(hostname): RANK=${RANK}, WORLD_SIZE=${WORLD_SIZE}"

python src/train.py \
    --data ./data \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --hidden-dim 64 \
    --num-layers 2 \
    --precision fp32 \
    --num-workers 2 \
    --results "'${RESULTS_DIR}'" \
    --seed 42 \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT}
'

echo ""
echo "=============================================="
echo "Training completed"
echo "End time: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================="

sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,Elapsed,MaxRSS > "${RESULTS_DIR}/sacct.txt" 2>/dev/null
