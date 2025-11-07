#!/bin/bash
# Strong scaling experiment: fixed problem size, varying number of nodes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Configuration
ACCOUNT="<account>"
PARTITION="<partition>"
NODES_LIST=(1 2 4)  # Adjust based on quota
GPUS_PER_NODE=4
CPUS_PER_TASK=8
BATCH_SIZE=32  # Fixed per GPU
EPOCHS=1
PRECISION="bf16"
NUM_WORKERS=4

# Fixed problem size
NUM_NODES_GRAPH=50
SEQ_LEN=12
PRED_LEN=1

echo "Starting strong scaling experiments..."
echo "Nodes to test: ${NODES_LIST[@]}"
echo "Fixed batch size per GPU: ${BATCH_SIZE}"

# Create results directory
RESULTS_BASE="./results/strong_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_BASE}"

# Run experiments for each node count
for NODES in "${NODES_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running strong scaling: ${NODES} node(s)"
    echo "=========================================="
    
    JOB_NAME="dcrnn-strong-${NODES}n"
    RESULTS_DIR="${RESULTS_BASE}/${NODES}nodes"
    mkdir -p "${RESULTS_DIR}"
    
    # Create temporary sbatch script
    SBATCH_SCRIPT="${RESULTS_DIR}/job.sbatch"
    cat > "${SBATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=01:00:00
#SBATCH --output=${RESULTS_DIR}/slurm_%j.out
#SBATCH --error=${RESULTS_DIR}/slurm_%j.err

export OMP_NUM_THREADS=${CPUS_PER_TASK}
export NCCL_DEBUG=INFO

cd "${PROJECT_DIR}"

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

if [ ${NODES} -eq 1 ]; then
    # Single node: use DataParallel
    ./run.sh python src/train.py \
      --data ./data \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --precision ${PRECISION} \
      --num-workers ${NUM_WORKERS} \
      --results "${RESULTS_DIR}" \
      --seed 42 \
      --monitor-gpu \
      --monitor-cpu
else
    # Multi-node: use DDP
    srun --ntasks-per-node=4 --gpus-per-task=1 \
      ./run.sh python -m torch.distributed.run \
      --nproc_per_node=4 \
      --nnodes=${NODES} \
      --node_rank=\${SLURM_NODEID} \
      --master_addr=\${MASTER_ADDR} \
      --master_port=\${MASTER_PORT} \
      src/train.py \
      --data ./data \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --precision ${PRECISION} \
      --num-workers ${NUM_WORKERS} \
      --results "${RESULTS_DIR}" \
      --seed 42 \
      --monitor-gpu \
      --monitor-cpu
fi

sacct -j \${SLURM_JOB_ID} --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"
EOF
    
    chmod +x "${SBATCH_SCRIPT}"
    
    # Submit job
    echo "Submitting job for ${NODES} node(s)..."
    JOB_ID=$(sbatch "${SBATCH_SCRIPT}" | awk '{print $4}')
    echo "Job ID: ${JOB_ID}"
    echo "Results will be saved to: ${RESULTS_DIR}"
    
    # Wait for job to complete (optional - remove if you want to submit all at once)
    # echo "Waiting for job to complete..."
    # squeue -j ${JOB_ID} 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "All strong scaling jobs submitted!"
echo "Results base directory: ${RESULTS_BASE}"
echo "=========================================="
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To collect results after jobs complete, run:"
echo "  python scripts/analyze_scaling.py --results ${RESULTS_BASE} --type strong"

