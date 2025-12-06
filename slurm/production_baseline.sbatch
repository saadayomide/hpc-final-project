#!/bin/bash
#SBATCH --job-name=dcrnn-prod-baseline
#SBATCH --account=def-sponsor00
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/production_baseline_%j.out
#SBATCH --error=results/production_baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@example.com

# ============================================================
# DCRNN Production Baseline Training - Single Node
# ============================================================

set -e  # Exit on error

echo "=============================================="
echo "DCRNN Production Baseline Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Set production environment
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo ""

# Create results directory
RESULTS_DIR="./results/production_baseline_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Log system info
echo "System Information:"
echo "  Hostname: $(hostname)"
echo "  CUDA Version: $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Verify container
if [ ! -f "env/project.sif" ]; then
    echo "ERROR: Container not found. Building..."
    ./run.sh build
fi

# Run training
echo "Starting production training..."
echo "================================"

./run.sh exec --nv python src/train.py \
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
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code: ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================="

# Save job information
sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList \
    > "${RESULTS_DIR}/job_info.txt" 2>/dev/null

exit $EXIT_CODE
