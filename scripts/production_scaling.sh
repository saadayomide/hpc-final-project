#!/bin/bash
# Production Scaling Experiments
# Runs strong and weak scaling experiments for production

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================="
echo "Production Scaling Experiments"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Create results directory
RESULTS_DIR="./results/production_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

echo "Results directory: ${RESULTS_DIR}"
echo ""

# Strong Scaling: 1, 2, 4, 8 nodes
echo "Starting Strong Scaling Experiments..."
echo "--------------------------------------"

for nodes in 1 2 4 8; do
    echo ""
    echo "Submitting strong scaling job: ${nodes} nodes"
    job_id=$(sbatch \
        --job-name=prod-strong-${nodes}n \
        --nodes=${nodes} \
        --ntasks-per-node=4 \
        --gpus-per-node=4 \
        --time=04:00:00 \
        --output=${RESULTS_DIR}/strong_${nodes}n_%j.out \
        --error=${RESULTS_DIR}/strong_${nodes}n_%j.err \
        --parsable \
        slurm/production_multi_node.sbatch)
    
    echo "  Job ID: ${job_id}"
    echo "  Waiting for completion..."
    
    # Wait for job to complete
    while squeue -j ${job_id} &>/dev/null; do
        sleep 30
    done
    
    # Check if job succeeded
    if [ -f "${RESULTS_DIR}/strong_${nodes}n_${job_id}.out" ]; then
        echo "  ✅ Job ${job_id} completed"
    else
        echo "  ⚠️  Job ${job_id} may have failed - check logs"
    fi
done

# Weak Scaling: 1, 2, 4, 8 nodes (with scaled data)
echo ""
echo "Starting Weak Scaling Experiments..."
echo "--------------------------------------"

for nodes in 1 2 4 8; do
    echo ""
    echo "Submitting weak scaling job: ${nodes} nodes"
    job_id=$(sbatch \
        --job-name=prod-weak-${nodes}n \
        --nodes=${nodes} \
        --ntasks-per-node=4 \
        --gpus-per-node=4 \
        --time=04:00:00 \
        --output=${RESULTS_DIR}/weak_${nodes}n_%j.out \
        --error=${RESULTS_DIR}/weak_${nodes}n_%j.err \
        --parsable \
        slurm/production_multi_node.sbatch)
    
    echo "  Job ID: ${job_id}"
done

echo ""
echo "=============================================="
echo "All scaling jobs submitted"
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: ${RESULTS_DIR}"
echo "=============================================="
