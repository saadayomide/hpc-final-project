#!/bin/bash
#SBATCH --job-name=dcrnn-baseline
#SBATCH --account=def-sponsor00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=../results/baseline_%j.out
#SBATCH --error=../results/baseline_%j.err

# ============================================================
# DCRNN Baseline Training - Single Node
# ============================================================

echo "=============================================="
echo "DCRNN Baseline Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Start time: $(date)"
echo ""

# Set environment variables
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Get project directory (parent of slurm folder)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo ""

# Create results directory
RESULTS_DIR="./results/baseline_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Log system info
echo "System Information:" | tee "${RESULTS_DIR}/system_info.txt"
echo "==================" | tee -a "${RESULTS_DIR}/system_info.txt"
nvidia-smi | tee -a "${RESULTS_DIR}/system_info.txt"
echo "" | tee -a "${RESULTS_DIR}/system_info.txt"

# Record git hash
echo "Git hash: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')" | tee -a "${RESULTS_DIR}/system_info.txt"

# Run training
echo ""
echo "Starting training..."
echo "===================="

./run.sh python src/train.py \
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
