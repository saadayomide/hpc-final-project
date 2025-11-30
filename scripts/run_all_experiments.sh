#!/bin/bash
# Master script to run all experiments
# This script coordinates the full experimental workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================="
echo "DCRNN HPC Experiment Suite"
echo "=============================================="
echo "Project directory: ${PROJECT_DIR}"
echo ""

# Check for required files
if [ ! -f "run.sh" ]; then
    echo "Error: run.sh not found. Are you in the project root?"
    exit 1
fi

# Parse arguments
RUN_BASELINE=false
RUN_STRONG=false
RUN_WEAK=false
RUN_SENSITIVITY=false
RUN_ALL=false
GENERATE_DATA=false
BUILD_CONTAINER=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Run all experiments"
    echo "  --baseline      Run baseline (1 node) experiment"
    echo "  --strong        Run strong scaling experiments"
    echo "  --weak          Run weak scaling experiments"
    echo "  --sensitivity   Run sensitivity sweep"
    echo "  --data          Generate sample data"
    echo "  --build         Build container"
    echo "  --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --build --data --baseline"
    echo "  $0 --all"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --baseline)
            RUN_BASELINE=true
            shift
            ;;
        --strong)
            RUN_STRONG=true
            shift
            ;;
        --weak)
            RUN_WEAK=true
            shift
            ;;
        --sensitivity)
            RUN_SENSITIVITY=true
            shift
            ;;
        --data)
            GENERATE_DATA=true
            shift
            ;;
        --build)
            BUILD_CONTAINER=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if $RUN_ALL; then
    BUILD_CONTAINER=true
    GENERATE_DATA=true
    RUN_BASELINE=true
    RUN_STRONG=true
    RUN_WEAK=true
    RUN_SENSITIVITY=true
fi

# Step 1: Build container
if $BUILD_CONTAINER; then
    echo ""
    echo "Step 1: Building container..."
    echo "=============================="
    ./run.sh build
fi

# Step 2: Generate data
if $GENERATE_DATA; then
    echo ""
    echo "Step 2: Generating sample data..."
    echo "=================================="
    cd data
    ./run.sh python generate_sample_data.py || python3 generate_sample_data.py
    cd ..
fi

# Step 3: Run baseline
if $RUN_BASELINE; then
    echo ""
    echo "Step 3: Submitting baseline experiment..."
    echo "=========================================="
    cd slurm
    JOB_ID=$(sbatch baseline_1node.sbatch | awk '{print $4}')
    echo "Baseline job submitted: ${JOB_ID}"
    cd ..
fi

# Step 4: Run strong scaling
if $RUN_STRONG; then
    echo ""
    echo "Step 4: Submitting strong scaling experiments..."
    echo "================================================="
    ./scripts/strong_scaling.sh
fi

# Step 5: Run weak scaling
if $RUN_WEAK; then
    echo ""
    echo "Step 5: Submitting weak scaling experiments..."
    echo "==============================================="
    ./scripts/weak_scaling.sh
fi

# Step 6: Run sensitivity sweep
if $RUN_SENSITIVITY; then
    echo ""
    echo "Step 6: Submitting sensitivity sweep..."
    echo "========================================"
    ./scripts/sensitivity_sweep.sh
fi

echo ""
echo "=============================================="
echo "Experiment submission complete!"
echo "=============================================="
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "After completion, analyze results:"
echo "  python scripts/analyze_scaling.py --results results/strong_scaling_* --type strong"
echo "  python scripts/analyze_scaling.py --results results/weak_scaling_* --type weak"
echo "  python scripts/analyze_scaling.py --results results/sensitivity_* --type sensitivity"
