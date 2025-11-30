#!/bin/bash
#SBATCH --job-name=dcrnn-baseline
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/baseline_%j.out
#SBATCH --error=results/baseline_%j.err

echo "=============================================="
echo "DCRNN Baseline Training - Single GPU Node"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

# Set up environment
module purge
module load python/3.11 scipy-stack/2025a
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Project directory
PROJECT_DIR="/home/user42/hpc-final-project"
cd "${PROJECT_DIR}"

# Create results directory
mkdir -p results
RESULTS_DIR="./results/baseline_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Install PyTorch with CUDA if not already installed
echo "Checking PyTorch installation..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch with CUDA..."
    pip install --user torch --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
fi

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available on login node"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

# Run training
echo "Starting training..."
echo "===================="

python src/train.py \
    --data ./data \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --hidden-dim 64 \
    --num-layers 2 \
    --precision fp32 \
    --num-workers 2 \
    --results "${RESULTS_DIR}" \
    --seed 42

echo ""
echo "=============================================="
echo "Training completed"
echo "End time: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================="

# Save accounting
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,Elapsed,MaxRSS,AllocGRES > "${RESULTS_DIR}/sacct.txt" 2>/dev/null
