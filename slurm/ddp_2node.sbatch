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

# Check GPU availability and PyTorch setup
echo "Checking GPU and PyTorch setup on all nodes..."
srun --ntasks-per-node=1 bash -c '
    echo "Node $(hostname):"
    echo "  Checking nvidia-smi..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi failed or no GPUs found"
    
    echo "  Checking PyTorch..."
    if python -c "import torch; print(f\"PyTorch {torch.__version__}\"); assert torch.cuda.is_available(), \"CUDA not available\"" 2>/dev/null; then
        python -c "import torch; print(f\"  CUDA available: {torch.cuda.is_available()}\"); print(f\"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"None\\\"}\")"
    else
        echo "  WARNING: PyTorch CUDA check failed"
        echo "  Attempting CPU training with gloo backend"
        export BACKEND=gloo
    fi
    echo ""
'


# Get master node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=${SLURM_NNODES}

# Auto-detect backend if not set
if [ -z "$BACKEND" ]; then
    # Check if CUDA is available
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        export BACKEND=nccl
        echo "Auto-detected: Using NCCL backend (GPU available)"
    else
        export BACKEND=gloo
        echo "Auto-detected: Using Gloo backend (CPU training)"
    fi
fi


echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Backend: ${BACKEND}"
echo "World Size: ${WORLD_SIZE}"
echo ""
echo "Starting distributed training..."
echo "================================"

srun --export=ALL bash -c '
export RANK=${SLURM_PROCID}
export LOCAL_RANK=0
export BACKEND=${BACKEND:-nccl}

echo "Node $(hostname): RANK=${RANK}, WORLD_SIZE=${WORLD_SIZE}, BACKEND=${BACKEND}"


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
