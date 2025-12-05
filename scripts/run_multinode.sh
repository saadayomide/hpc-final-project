#!/bin/bash
# Helper script to execute multi-node DDP training
# Usage: ./scripts/run_multinode.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Default values
PARTITION="gpu-node"
NODES=2
EPOCHS=10
ACCOUNT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --partition PARTITION  SLURM partition (default: gpu-node)"
            echo "  --nodes N              Number of nodes (default: 2)"
            echo "  --epochs N             Number of epochs (default: 10)"
            echo "  --account ACCOUNT      SLURM account (optional)"
            echo "  --help                 Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Multi-Node DDP Training - Execution Helper"
echo "=============================================="
echo "Partition: ${PARTITION}"
echo "Nodes: ${NODES}"
echo "Epochs: ${EPOCHS}"
echo ""

# Check if SLURM script exists
SLURM_SCRIPT="slurm/ddp_2node_cpu.sbatch"
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}"
    exit 1
fi

# Create temporary SLURM script with custom parameters
TEMP_SCRIPT=$(mktemp)
cp "${SLURM_SCRIPT}" "${TEMP_SCRIPT}"

# Update parameters in temp script
sed -i.bak "s/#SBATCH --partition=.*/#SBATCH --partition=${PARTITION}/" "${TEMP_SCRIPT}"
sed -i.bak "s/#SBATCH --nodes=.*/#SBATCH --nodes=${NODES}/" "${TEMP_SCRIPT}"
sed -i.bak "s/--epochs [0-9]*/--epochs ${EPOCHS}/" "${TEMP_SCRIPT}"

if [ -n "${ACCOUNT}" ]; then
    # Add account line if not present
    if ! grep -q "#SBATCH --account" "${TEMP_SCRIPT}"; then
        sed -i.bak "/#SBATCH --partition/a\\
#SBATCH --account=${ACCOUNT}
" "${TEMP_SCRIPT}"
    else
        sed -i.bak "s/#SBATCH --account=.*/#SBATCH --account=${ACCOUNT}/" "${TEMP_SCRIPT}"
    fi
fi

# Clean up backup file
rm -f "${TEMP_SCRIPT}.bak"

echo "Submitting job..."
JOB_ID=$(sbatch "${TEMP_SCRIPT}" | awk '{print $4}')
echo "Job submitted: ${JOB_ID}"
echo ""

# Clean up temp script
rm -f "${TEMP_SCRIPT}"

# Show monitoring commands
echo "Monitor job status:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "View output (wait a few seconds first):"
echo "  tail -f results/ddp_cpu_${JOB_ID}.out"
echo ""
echo "After completion, verify results:"
echo "  ./scripts/verify_multinode.sh ${JOB_ID}"
echo ""

# Wait a moment and show initial status
sleep 2
echo "Current job status:"
squeue -j "${JOB_ID}" 2>/dev/null || echo "Job may have completed or not yet visible"

