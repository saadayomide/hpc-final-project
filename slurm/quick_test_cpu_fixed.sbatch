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

# Quick test run - CPU only (installs CPU PyTorch if needed)

set -e

echo "=============================================="
echo "DCRNN Quick Test Run (CPU Only - Fixed)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Load modules
module purge 2>/dev/null || true
module load python/3.11 scipy-stack/2025a 2>/dev/null || echo "Modules not available"

# Set environment
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export BACKEND=gloo  # Use CPU backend

# Project directory
PROJECT_DIR="/home/user42/hpc-final-project"
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo ""

# Create and activate virtual environment
echo "Setting up virtual environment..."
VENV_DIR="${PROJECT_DIR}/.venv_cpu_${SLURM_JOB_ID}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Virtual environment created: ${VENV_DIR}"
echo "Python: $(which python)"
echo ""

# Upgrade pip
pip install --quiet --upgrade pip

# Install CPU-only PyTorch and dependencies
echo "Installing PyTorch CPU version and dependencies..."
pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    numpy pandas scipy matplotlib psutil tqdm h5py

# Verify PyTorch works
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✓ PyTorch CPU version working!')
" || {
    echo "ERROR: PyTorch installation failed"
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

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Use Python from virtual environment
python src/train.py \
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
