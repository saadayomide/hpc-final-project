#!/bin/bash
###############################################################################
# Run All Experiments Script
#
# Submits all required experiments for the HPC Final Project:
# 1. Baseline training (1 node)
# 2. Strong scaling (1, 2, 4 nodes)
# 3. Weak scaling (1, 2, 4 nodes)
# 4. Profiling run
#
# Usage: ./scripts/run_all_experiments.sh [--dry-run]
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - Jobs will not be submitted"
fi

echo "=============================================="
echo "HPC Final Project - Full Experiment Suite"
echo "=============================================="
echo "Project: ${PROJECT_DIR}"
echo "Date: $(date)"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check for container or working Python
if [ -f "env/project.sif" ]; then
    echo "✓ Container found: env/project.sif"
elif python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch available (no container)"
else
    echo "✗ No container and PyTorch not available"
    echo "  Build container: ./run.sh build"
    echo "  Or install PyTorch: pip install torch"
    exit 1
fi

# Check for data
if [ -f "data/processed/train.npz" ]; then
    echo "✓ Training data found"
else
    echo "✗ Training data not found"
    echo "  Generate data: cd data && python generate_sample_data.py"
    exit 1
fi

echo ""

###############################################################################
# Submit Experiments
###############################################################################

ALL_JOBS=()

submit_job() {
    local script=$1
    local name=$2
    
    if [ ! -f "$script" ]; then
        echo "  ✗ Script not found: $script"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "  Would submit: $script"
    else
        JOBID=$(sbatch "$script" 2>&1 | awk '{print $4}')
        if [ -n "$JOBID" ]; then
            echo "  ✓ Submitted $name: Job $JOBID"
            ALL_JOBS+=("$JOBID")
        else
            echo "  ✗ Failed to submit $name"
        fi
    fi
}

# 1. Baseline training
echo "1. Submitting baseline training (1 node)..."
submit_job "slurm/baseline_1node.sbatch" "Baseline"
echo ""

# Wait a bit to avoid port conflicts
sleep 2

# 2. Strong scaling experiments
echo "2. Submitting strong scaling experiments..."

# Check for individual scaling scripts or generate them
if [ -f "slurm/strong_scaling_1n.sbatch" ]; then
    submit_job "slurm/strong_scaling_1n.sbatch" "Strong-1n"
    sleep 2
    submit_job "slurm/strong_scaling_2n.sbatch" "Strong-2n"
    sleep 2
    submit_job "slurm/strong_scaling_4n.sbatch" "Strong-4n"
else
    # Use the shell script to generate and submit
    if [ "$DRY_RUN" = true ]; then
        ./scripts/strong_scaling.sh --dry-run
    else
        echo "  Running strong_scaling.sh to generate and submit jobs..."
        ./scripts/strong_scaling.sh 2>&1 | grep -E "Submitted|Would submit"
    fi
fi
echo ""

sleep 2

# 3. Weak scaling experiments
echo "3. Submitting weak scaling experiments..."

if [ -f "slurm/weak_scaling_1n.sbatch" ]; then
    submit_job "slurm/weak_scaling_1n.sbatch" "Weak-1n"
    sleep 2
    submit_job "slurm/weak_scaling_2n.sbatch" "Weak-2n"
    sleep 2
    submit_job "slurm/weak_scaling_4n.sbatch" "Weak-4n"
else
    if [ "$DRY_RUN" = true ]; then
        ./scripts/weak_scaling.sh --dry-run
    else
        echo "  Running weak_scaling.sh to generate and submit jobs..."
        ./scripts/weak_scaling.sh 2>&1 | grep -E "Submitted|Would submit"
    fi
fi
echo ""

# 4. DDP test (2 nodes)
echo "4. Submitting DDP test (2 nodes)..."
submit_job "slurm/ddp_2node.sbatch" "DDP-2n"
echo ""

###############################################################################
# Summary
###############################################################################

echo "=============================================="
echo "Experiment Submission Summary"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - No jobs submitted"
else
    echo "Jobs submitted: ${#ALL_JOBS[@]}"
    if [ ${#ALL_JOBS[@]} -gt 0 ]; then
        echo "Job IDs: ${ALL_JOBS[*]}"
    fi
fi

echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check outputs:"
echo "  tail -f results/*.out"
echo ""
echo "After all jobs complete, analyze results:"
echo "  python scripts/analyze_scaling.py --results results/strong_scaling_* --type strong"
echo "  python scripts/analyze_scaling.py --results results/weak_scaling_* --type weak"
echo "  python scripts/analyze_profiling.py --results results/baseline_*/"
echo ""
echo "=============================================="
