#!/bin/bash
#SBATCH --job-name=dcrnn-profile-cpu-2n
#SBATCH --account=<account>
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=results/profiling/profile_cpu_2n_%j.out
#SBATCH --error=results/profiling/profile_cpu_2n_%j.err

# Phase 3: Profiling & Bottleneck Identification
# 2-node CPU DDP profiling (for CPU-based distributed training)

echo "=============================================="
echo "DCRNN CPU DDP Profiling - 2 Nodes"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Start time: $(date)"

module purge
module load python/3.11 scipy-stack/2025a
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

PROJECT_DIR="/home/user42/hpc-final-project"
cd "${PROJECT_DIR}"

mkdir -p results/profiling
RESULTS_DIR="./results/profiling/cpu_2node_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Install CPU-only PyTorch on all nodes
echo "Installing CPU-only PyTorch on all nodes..."
srun --ntasks-per-node=1 bash -c '
    pip uninstall -y torch 2>/dev/null
    pip install --user torch --extra-index-url https://download.pytorch.org/whl/cpu --quiet
    python -c "import torch; print(f\"$(hostname): PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}\")"
'

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=${SLURM_NNODES}
export BACKEND=gloo

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Using CPU backend (gloo) for distributed training"
echo ""

# Configuration
EPOCHS=3
BATCH_SIZE=32

echo "Starting CPU DDP training with monitoring..."
echo ""

srun --export=ALL bash -c '
export RANK=${SLURM_PROCID}
export LOCAL_RANK=0

python src/train.py \
    --data ./data \
    --epochs '${EPOCHS}' \
    --batch-size '${BATCH_SIZE}' \
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
    --master-port ${MASTER_PORT} \
    --monitor-cpu
'

echo ""
echo "Training completed. Results: ${RESULTS_DIR}"
echo "End time: $(date)"

# Save comprehensive SLURM accounting
echo "Saving SLURM accounting information..."
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,NNodes,NTasks,ReqNodes > "${RESULTS_DIR}/sacct_summary.txt" 2>/dev/null || true

# Verify results
if [ -f "${RESULTS_DIR}/metrics.csv" ]; then
    echo "✓ Metrics CSV found"
    echo "Results summary:"
    tail -n 1 "${RESULTS_DIR}/metrics.csv" | head -c 200
    echo ""
else
    echo "⚠ WARNING: metrics.csv not found"
fi

echo "=============================================="
echo "CPU DDP profiling complete!"
echo "Results: ${RESULTS_DIR}"
echo "=============================================="

