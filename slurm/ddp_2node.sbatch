#!/bin/bash
#SBATCH --job-name=dcrnn-ddp-2n
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
# DCRNN DDP Training - 2 GPU Nodes
###############################################################################

echo "=============================================="
echo "DCRNN DDP Training - 2 Nodes"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
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

# Load environment
source env/load_modules.sh

# DDP Configuration
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
WORLD_SIZE=${SLURM_NNODES}

export MASTER_ADDR MASTER_PORT WORLD_SIZE
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ""
echo "DDP Configuration:"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  World size: ${WORLD_SIZE}"
echo ""

# Training parameters
EPOCHS=50
BATCH_SIZE=32
SEED=42

# Check container
CONTAINER_PATH="${PROJECT_DIR}/env/project.sif"
USE_CONTAINER=false
if [ -f "${CONTAINER_PATH}" ]; then
    USE_CONTAINER=true
    echo "Using container: ${CONTAINER_PATH}"
fi

echo "Starting distributed training..."
echo "================================"
echo ""

# Launch training on each node
srun --ntasks-per-node=1 --export=ALL bash -c '
    # Set rank from SLURM
    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=${SLURM_LOCALID:-0}
    
    echo "Node $(hostname): RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}"
    
    cd '"${PROJECT_DIR}"'
    
    # Build training command
    TRAIN_CMD="python src/train.py \
        --data ./data \
        --epochs '"${EPOCHS}"' \
        --batch-size '"${BATCH_SIZE}"' \
        --lr 0.001 \
        --hidden-dim 64 \
        --num-layers 2 \
        --precision fp32 \
        --num-workers 4 \
        --results '"${RESULTS_DIR}"' \
        --seed '"${SEED}"' \
        --rank ${RANK} \
        --world-size ${WORLD_SIZE} \
        --master-addr ${MASTER_ADDR} \
        --master-port ${MASTER_PORT}"
    
    # Add monitoring only for rank 0
    if [ ${RANK} -eq 0 ]; then
        TRAIN_CMD="${TRAIN_CMD} --monitor-gpu --monitor-cpu"
    fi
    
    # Execute with or without container
    if [ "'"${USE_CONTAINER}"'" = "true" ]; then
        apptainer exec --nv \
            --bind '"${PROJECT_DIR}"':/workspace \
            --pwd /workspace \
            '"${CONTAINER_PATH}"' \
            bash -c "${TRAIN_CMD}"
    else
        bash -c "${TRAIN_CMD}"
    fi
'

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

# Save accounting
sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocGRES,NodeList \
    2>/dev/null | tee "${RESULTS_DIR}/sacct.txt"

# Save configuration
cat > "${RESULTS_DIR}/config.txt" << EOF
experiment: ddp_2node
job_id: ${SLURM_JOB_ID}
nodes: ${SLURM_NNODES}
node_list: ${SLURM_JOB_NODELIST}
master: ${MASTER_ADDR}:${MASTER_PORT}
world_size: ${WORLD_SIZE}
epochs: ${EPOCHS}
batch_size_per_gpu: ${BATCH_SIZE}
effective_batch_size: $((BATCH_SIZE * WORLD_SIZE))
seed: ${SEED}
container: ${USE_CONTAINER}
EOF

echo ""
echo "Results saved to: ${RESULTS_DIR}"

exit ${EXIT_CODE}
