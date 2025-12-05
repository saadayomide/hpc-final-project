#!/bin/bash
#SBATCH --job-name=dcrnn-ddp
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/ddp_%j.out
#SBATCH --error=results/ddp_%j.err
#SBATCH --exclusive

###############################################################################
# DCRNN Distributed Training - Multi-Node DDP
# Uses PyTorch DistributedDataParallel across multiple GPU nodes
###############################################################################

echo "=============================================="
echo "DCRNN Multi-Node DDP Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "=============================================="

# Project directory
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/user42/hpc-final-project}"
cd "${PROJECT_DIR}"

# Create results directory
mkdir -p results
RESULTS_DIR="./results/ddp_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Load environment on all nodes
source env/load_modules.sh

###############################################################################
# DDP Configuration
###############################################################################

# Get master node (first node in allocation)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# Calculate world size
WORLD_SIZE=${SLURM_NNODES}
GPUS_PER_NODE=1  # Adjust if using multiple GPUs per node

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE

# NCCL optimization settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ""
echo "DDP Configuration:"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  World size: ${WORLD_SIZE}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo ""

# Training configuration
EPOCHS=50
BATCH_SIZE=32  # Per GPU
LEARNING_RATE=0.001
HIDDEN_DIM=64
NUM_LAYERS=2
PRECISION="fp32"
NUM_WORKERS=4
SEED=42

echo "Training Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: $((BATCH_SIZE * WORLD_SIZE))"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Precision: ${PRECISION}"
echo ""

###############################################################################
# Verify GPU Access on All Nodes
###############################################################################
echo "Verifying GPU access on all nodes..."

CONTAINER_PATH="${PROJECT_DIR}/env/project.sif"
USE_CONTAINER=false

if [ -f "${CONTAINER_PATH}" ]; then
    echo "Using Apptainer container: ${CONTAINER_PATH}"
    USE_CONTAINER=true
    
    # Verify on all nodes
    srun --ntasks-per-node=1 apptainer exec --nv "${CONTAINER_PATH}" \
        python -c "import torch; print(f'Node {torch.distributed.get_rank() if torch.distributed.is_initialized() else \"N/A\"}: GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null || true
else
    echo "No container found, using direct Python"
    
    # Verify Python/CUDA on all nodes
    srun --ntasks-per-node=1 bash -c '
        echo "Node $(hostname): $(python -c "import torch; print(f\"CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")" 2>/dev/null || echo "Python/PyTorch not available")'
fi

echo ""

###############################################################################
# Run Distributed Training
###############################################################################
echo "Starting distributed training..."
echo "================================"
echo ""

# Create the training command
TRAIN_SCRIPT="
import os
import sys
sys.path.insert(0, '/workspace' if os.path.exists('/workspace/src') else '${PROJECT_DIR}')

# Set rank from SLURM
rank = int(os.environ.get('SLURM_PROCID', 0))
local_rank = int(os.environ.get('SLURM_LOCALID', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

print(f'Starting rank {rank}/{world_size} on {os.environ.get(\"SLURMD_NODENAME\", \"unknown\")}')

# Set environment for this rank
os.environ['RANK'] = str(rank)
os.environ['LOCAL_RANK'] = str(local_rank)
os.environ['WORLD_SIZE'] = str(world_size)

# Now run training
import subprocess
cmd = [
    sys.executable, 'src/train.py',
    '--data', './data',
    '--epochs', '${EPOCHS}',
    '--batch-size', '${BATCH_SIZE}',
    '--lr', '${LEARNING_RATE}',
    '--hidden-dim', '${HIDDEN_DIM}',
    '--num-layers', '${NUM_LAYERS}',
    '--precision', '${PRECISION}',
    '--num-workers', '${NUM_WORKERS}',
    '--results', '${RESULTS_DIR}',
    '--seed', '${SEED}',
    '--rank', str(rank),
    '--world-size', str(world_size),
    '--master-addr', '${MASTER_ADDR}',
    '--master-port', '${MASTER_PORT}'
]
if rank == 0:
    cmd.extend(['--monitor-gpu', '--monitor-cpu'])

os.chdir('/workspace' if os.path.exists('/workspace/src') else '${PROJECT_DIR}')
subprocess.run(cmd, check=True)
"

if [ "${USE_CONTAINER}" = true ]; then
    echo "Running with Apptainer container..."
    srun --ntasks-per-node=1 --export=ALL \
        apptainer exec --nv \
        --bind "${PROJECT_DIR}:/workspace" \
        --pwd /workspace \
        "${CONTAINER_PATH}" \
        python -c "${TRAIN_SCRIPT}"
else
    echo "Running with direct Python..."
    srun --ntasks-per-node=1 --export=ALL \
        python -c "${TRAIN_SCRIPT}"
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
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocGRES,NodeList \
    2>/dev/null | tee "${RESULTS_DIR}/sacct.txt"

# Save DDP configuration
cat > "${RESULTS_DIR}/ddp_config.txt" << EOF
Job ID: ${SLURM_JOB_ID}
Nodes: ${SLURM_NNODES}
Node list: ${SLURM_JOB_NODELIST}
Master: ${MASTER_ADDR}:${MASTER_PORT}
World size: ${WORLD_SIZE}
GPUs per node: ${GPUS_PER_NODE}
Total GPUs: $((WORLD_SIZE * GPUS_PER_NODE))
Container used: ${USE_CONTAINER}
Batch size per GPU: ${BATCH_SIZE}
Effective batch size: $((BATCH_SIZE * WORLD_SIZE))
EOF

echo ""
echo "Results saved to: ${RESULTS_DIR}"

exit ${TRAIN_EXIT_CODE}
