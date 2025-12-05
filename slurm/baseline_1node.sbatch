#!/bin/bash
#SBATCH --job-name=dcrnn-baseline
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/baseline_%j.out
#SBATCH --error=results/baseline_%j.err

###############################################################################
# DCRNN Baseline Training - Single GPU Node
# This script runs a baseline training run for performance comparison
###############################################################################

echo "=============================================="
echo "DCRNN Baseline Training - Single GPU Node"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

# Project directory
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/user42/hpc-final-project}"
cd "${PROJECT_DIR}"

# Create results directory
mkdir -p results
RESULTS_DIR="./results/baseline_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Load environment
source env/load_modules.sh

# Configuration
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
HIDDEN_DIM=64
NUM_LAYERS=2
PRECISION="fp32"  # Use fp32 for baseline, bf16 for speed
NUM_WORKERS=4
SEED=42

echo ""
echo "Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Precision: ${PRECISION}"
echo ""

###############################################################################
# GPU Detection and Setup
###############################################################################
echo "Checking GPU availability..."

# Method 1: Try using Apptainer container (preferred)
CONTAINER_PATH="${PROJECT_DIR}/env/project.sif"
USE_CONTAINER=false

if [ -f "${CONTAINER_PATH}" ]; then
    echo "Found Apptainer container: ${CONTAINER_PATH}"
    # Test container GPU access
    if apptainer exec --nv "${CONTAINER_PATH}" python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Container GPU access: OK"
        USE_CONTAINER=true
    else
        echo "Container GPU access: FAILED, trying direct Python"
    fi
else
    echo "No container found at ${CONTAINER_PATH}"
fi

# Method 2: Direct Python with CUDA
if [ "${USE_CONTAINER}" = false ]; then
    echo "Checking direct Python CUDA availability..."
    
    # Try to ensure PyTorch with CUDA is available
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "PyTorch CUDA not available, attempting to install..."
        pip install --user torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --quiet 2>/dev/null || true
    fi
    
    # Verify
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Direct Python GPU access: OK"
    else
        echo "WARNING: GPU not available, training will use CPU (slow!)"
    fi
fi

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo ""

###############################################################################
# Run Training
###############################################################################
echo "Starting training..."
echo "===================="
echo ""

TRAIN_CMD="python src/train.py \
    --data ./data \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --precision ${PRECISION} \
    --num-workers ${NUM_WORKERS} \
    --results ${RESULTS_DIR} \
    --seed ${SEED} \
    --monitor-gpu \
    --monitor-cpu"

if [ "${USE_CONTAINER}" = true ]; then
    echo "Running with Apptainer container..."
    apptainer exec --nv \
        --bind "${PROJECT_DIR}:/workspace" \
        --pwd /workspace \
        "${CONTAINER_PATH}" \
        ${TRAIN_CMD}
else
    echo "Running with direct Python..."
    ${TRAIN_CMD}
fi

TRAIN_EXIT_CODE=$?

###############################################################################
# Post-processing
###############################################################################
echo ""
echo "=============================================="
echo "Training completed with exit code: ${TRAIN_EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

# Save job accounting
echo ""
echo "Job Accounting:"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocGRES \
    2>/dev/null | tee "${RESULTS_DIR}/sacct.txt"

# Save environment info
cat > "${RESULTS_DIR}/environment.txt" << EOF
Job ID: ${SLURM_JOB_ID}
Node: ${SLURMD_NODENAME}
Start time: $(date)
Container used: ${USE_CONTAINER}
Container path: ${CONTAINER_PATH}
Python version: $(python --version 2>&1)
PyTorch version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA available: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
GPU name: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
EOF

echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "  - metrics.csv: Training metrics per epoch"
echo "  - gpu_monitor.csv: GPU utilization over time"
echo "  - cpu_monitor.csv: CPU utilization over time"
echo "  - checkpoint_*.pth: Model checkpoints"
echo "  - sacct.txt: SLURM accounting"
echo "  - environment.txt: Environment information"

exit ${TRAIN_EXIT_CODE}
