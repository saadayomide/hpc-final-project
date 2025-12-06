#!/bin/bash
#SBATCH --job-name=dcrnn-quick-cpu
#SBATCH --account=def-sponsor00
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=results/quick_test_cpu_%j.out
#SBATCH --error=results/quick_test_cpu_%j.err

# Quick test run - CPU only (no GPU needed)
# This will work even without CUDA libraries

set -e

echo "=============================================="
echo "DCRNN Quick Test Run (CPU Only)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Load modules
module purge 2>/dev/null || true
module load python/3.11 scipy-stack/2025a 2>/dev/null || echo "Modules not available"

# Try to install/use PyTorch
echo "Checking for PyTorch..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Attempting to install CPU version..."
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5 || echo "Installation may have failed"
fi

# Set environment
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BACKEND=gloo  # Use CPU backend

# Project directory
PROJECT_DIR="/home/user42/hpc-final-project"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo ""

# Check Python
python3 -c "import sys; print(f'Python: {sys.version}')"

# Try PyTorch (will use CPU)
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('Using CPU mode (no GPU needed)')
" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

# Create results directory
RESULTS_DIR="./results/quick_test_cpu_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Run QUICK training (CPU mode)
echo ""
echo "Starting QUICK test run (CPU, 5 epochs)..."
echo "This will complete in ~15-20 minutes"
echo "================================"

python3 src/train.py \
    --data ./data \
    --epochs 5 \
    --batch-size 32 \
    --lr 0.001 \
    --hidden-dim 64 \
    --num-layers 2 \
    --precision fp32 \
    --num-workers 2 \
    --results "${RESULTS_DIR}" \
    --seed 42 \
    --monitor-cpu \
    --checkpoint-interval 5

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Quick test completed successfully!"
    echo "Results saved to: ${RESULTS_DIR}"
    echo "Check metrics: cat ${RESULTS_DIR}/metrics.csv"
else
    echo "❌ Quick test failed"
fi
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
