#!/bin/bash
###############################################################################
# Weak Scaling Experiment Script
# 
# Purpose: Measure throughput as we add more nodes while keeping the
#          work per GPU constant. Dataset size grows with node count.
#
# Usage: ./scripts/weak_scaling.sh [--dry-run]
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# Configuration - adjust for your cluster
PARTITION="${PARTITION:-gpu-node}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="02:00:00"
MEM_PER_NODE="32G"

# Experiment configuration
NODES_LIST=(1 2 4)  # Number of nodes to test
EPOCHS=20           # Same epochs for all
BASE_BATCH_SIZE=32  # Batch size per GPU (constant for weak scaling)
SEED=42

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="results/weak_scaling_${TIMESTAMP}"
mkdir -p "${RESULTS_BASE}"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - Jobs will not be submitted"
fi

echo "=============================================="
echo "Weak Scaling Experiment"
echo "=============================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Results directory: ${RESULTS_BASE}"
echo "Partition: ${PARTITION}"
echo "Nodes to test: ${NODES_LIST[*]}"
echo "Epochs per run: ${EPOCHS}"
echo "Batch size per GPU: ${BASE_BATCH_SIZE} (constant)"
echo "=============================================="
echo ""

# Generate sbatch scripts and submit jobs
JOBIDS=()
for NODES in "${NODES_LIST[@]}"; do
    RESULTS_DIR="${RESULTS_BASE}/${NODES}n"
    mkdir -p "${RESULTS_DIR}"
    
    SBATCH_FILE="${RESULTS_DIR}/submit.sbatch"
    
    # For weak scaling: total batch size grows with nodes
    # Each GPU processes the same amount of data
    EFFECTIVE_BATCH=$((BASE_BATCH_SIZE * NODES))
    
    cat > "${SBATCH_FILE}" << EOF
#!/bin/bash
#SBATCH --job-name=weak-${NODES}n
#SBATCH --partition=${PARTITION}
${ACCOUNT:+#SBATCH --account=${ACCOUNT}}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM_PER_NODE}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${RESULTS_DIR}/job_%j.out
#SBATCH --error=${RESULTS_DIR}/job_%j.err
#SBATCH --exclusive

echo "Weak Scaling: ${NODES} nodes"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Effective batch size: ${EFFECTIVE_BATCH}"
echo "Start: \$(date)"

cd "${PROJECT_DIR}"
source env/load_modules.sh

# Set up DDP environment
MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE=${NODES}
export NCCL_DEBUG=WARN

echo "Master: \${MASTER_ADDR}:\${MASTER_PORT}"

# Container or direct execution
CONTAINER_PATH="${PROJECT_DIR}/env/project.sif"

# Run training
if [ ${NODES} -eq 1 ]; then
    if [ -f "\${CONTAINER_PATH}" ]; then
        apptainer exec --nv --bind ${PROJECT_DIR}:/workspace --pwd /workspace \${CONTAINER_PATH} \\
            python src/train.py \\
            --data ./data \\
            --epochs ${EPOCHS} \\
            --batch-size ${BASE_BATCH_SIZE} \\
            --precision fp32 \\
            --num-workers 4 \\
            --results ${RESULTS_DIR} \\
            --seed ${SEED} \\
            --monitor-gpu \\
            --monitor-cpu
    else
        python src/train.py \\
            --data ./data \\
            --epochs ${EPOCHS} \\
            --batch-size ${BASE_BATCH_SIZE} \\
            --precision fp32 \\
            --num-workers 4 \\
            --results ${RESULTS_DIR} \\
            --seed ${SEED} \\
            --monitor-gpu \\
            --monitor-cpu
    fi
else
    srun --ntasks-per-node=1 --export=ALL bash -c '
        export RANK=\${SLURM_PROCID}
        export LOCAL_RANK=\${SLURM_LOCALID}
        
        cd ${PROJECT_DIR}
        
        TRAIN_CMD="python src/train.py \\
            --data ./data \\
            --epochs ${EPOCHS} \\
            --batch-size ${BASE_BATCH_SIZE} \\
            --precision fp32 \\
            --num-workers 4 \\
            --results ${RESULTS_DIR} \\
            --seed ${SEED} \\
            --rank \${RANK} \\
            --world-size ${NODES} \\
            --master-addr \${MASTER_ADDR} \\
            --master-port \${MASTER_PORT}"
        
        if [ \${RANK} -eq 0 ]; then
            TRAIN_CMD="\${TRAIN_CMD} --monitor-gpu --monitor-cpu"
        fi
        
        if [ -f "${PROJECT_DIR}/env/project.sif" ]; then
            apptainer exec --nv --bind ${PROJECT_DIR}:/workspace --pwd /workspace ${PROJECT_DIR}/env/project.sif \\
                bash -c "\${TRAIN_CMD}"
        else
            bash -c "\${TRAIN_CMD}"
        fi
    '
fi

echo "End: \$(date)"

# Save accounting
sacct -j \${SLURM_JOB_ID} --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocGRES \\
    > ${RESULTS_DIR}/sacct.txt 2>/dev/null

# Save metadata
cat > ${RESULTS_DIR}/metadata.txt << METADATA
experiment: weak_scaling
nodes: ${NODES}
epochs: ${EPOCHS}
batch_size_per_gpu: ${BASE_BATCH_SIZE}
effective_batch_size: ${EFFECTIVE_BATCH}
seed: ${SEED}
job_id: \${SLURM_JOB_ID}
METADATA
EOF

    chmod +x "${SBATCH_FILE}"
    
    if [ "${DRY_RUN}" = true ]; then
        echo "Would submit: ${SBATCH_FILE}"
    else
        JOBID=$(sbatch "${SBATCH_FILE}" | awk '{print $4}')
        JOBIDS+=("${JOBID}")
        echo "Submitted ${NODES}-node job: ${JOBID}"
    fi
done

if [ "${DRY_RUN}" = false ] && [ ${#JOBIDS[@]} -gt 0 ]; then
    echo ""
    echo "=============================================="
    echo "Submitted ${#JOBIDS[@]} jobs: ${JOBIDS[*]}"
    echo "Monitor with: squeue -u \$USER"
    echo "Results will be in: ${RESULTS_BASE}"
    echo ""
    echo "After completion, analyze with:"
    echo "  python scripts/analyze_scaling.py --results ${RESULTS_BASE} --type weak"
    echo "=============================================="
    
    echo "${JOBIDS[*]}" > "${RESULTS_BASE}/job_ids.txt"
fi
