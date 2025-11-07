#!/bin/bash
# Sensitivity sweep: varying batch size and num_workers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Configuration
ACCOUNT="<account>"
PARTITION="<partition>"
NODES=2  # Fixed number of nodes
GPUS_PER_NODE=4
CPUS_PER_TASK=8
EPOCHS=1
PRECISION="bf16"

# Sweep parameters
BATCH_SIZES=(32 64 128)
NUM_WORKERS_LIST=(2 4 8)

echo "Starting sensitivity sweep experiments..."
echo "Nodes: ${NODES}"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Num workers: ${NUM_WORKERS_LIST[@]}"

# Create results directory
RESULTS_BASE="./results/sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_BASE}"

# Run experiments for each combination
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for NUM_WORKERS in "${NUM_WORKERS_LIST[@]}"; do
        echo ""
        echo "=========================================="
        echo "Running: batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}"
        echo "=========================================="
        
        JOB_NAME="dcrnn-sens-b${BATCH_SIZE}-w${NUM_WORKERS}"
        RESULTS_DIR="${RESULTS_BASE}/batch${BATCH_SIZE}_workers${NUM_WORKERS}"
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

sacct -j \${SLURM_JOB_ID} --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"
EOF
        
        chmod +x "${SBATCH_SCRIPT}"
        
        # Submit job
        echo "Submitting job..."
        JOB_ID=$(sbatch "${SBATCH_SCRIPT}" | awk '{print $4}')
        echo "Job ID: ${JOB_ID}"
        echo "Results will be saved to: ${RESULTS_DIR}"
    done
done

echo ""
echo "=========================================="
echo "All sensitivity sweep jobs submitted!"
echo "Results base directory: ${RESULTS_BASE}"
echo "=========================================="
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To collect results after jobs complete, run:"
echo "  python scripts/analyze_scaling.py --results ${RESULTS_BASE} --type sensitivity"

