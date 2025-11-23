#!/bin/bash
# Script to run all Phase 3 profiling scenarios (1-node, 2-node, 4-node)
# This submits profiling jobs for all three scenarios

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "Phase 3: Profiling & Bottleneck Identification"
echo "=========================================="
echo ""
echo "This script will submit profiling jobs for:"
echo "  1. 1-node run (baseline)"
echo "  2. 2-node run (scaling still good)"
echo "  3. 4-node run (scaling degrades)"
echo ""

# Configuration - Try to auto-detect or use environment variables
# Check if config file exists
CONFIG_FILE="${SCRIPT_DIR}/profiling_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
fi

# Check if account/partition are set via environment
ACCOUNT="${SLURM_ACCOUNT}"
PARTITION="${SLURM_PARTITION}"

# Try to auto-detect account from SLURM (if not set)
if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "<account>" ]; then
    # Try to get account from sacctmgr
    ACCOUNT=$(sacctmgr show user $USER -p 2>/dev/null | head -1 | cut -d'|' -f2 2>/dev/null || echo "")
    
    # If still empty, try to get from running jobs
    if [ -z "$ACCOUNT" ]; then
        ACCOUNT=$(squeue -u $USER -o "%a" -h 2>/dev/null | head -1 || echo "")
    fi
fi

# Try to auto-detect partition (if not set)
if [ -z "$PARTITION" ] || [ "$PARTITION" = "<partition>" ]; then
    # Try to get partition from running jobs
    PARTITION=$(squeue -u $USER -o "%P" -h 2>/dev/null | head -1 || echo "")
    
    # If still empty, check for GPU partitions
    if [ -z "$PARTITION" ]; then
        if sinfo -p gpu-node 2>/dev/null | grep -q "gpu-node"; then
            PARTITION="gpu-node"
        elif sinfo -p node 2>/dev/null | grep -q "node"; then
            PARTITION="node"
        fi
    fi
fi

# Validate configuration
if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "<account>" ]; then
    echo "=========================================="
    echo "ERROR: SLURM account not configured."
    echo "=========================================="
    echo ""
    echo "Please set it using one of these methods:"
    echo ""
    echo "Method 1 (Recommended): Use configuration helper"
    echo "  ./scripts/configure_profiling.sh"
    echo ""
    echo "Method 2: Export environment variable"
    echo "  export SLURM_ACCOUNT='your_account'"
    echo "  export SLURM_PARTITION='your_partition'"
    echo ""
    echo "Method 3: Create config file"
    echo "  echo \"export SLURM_ACCOUNT='your_account'\" > scripts/profiling_config.sh"
    echo "  echo \"export SLURM_PARTITION='your_partition'\" >> scripts/profiling_config.sh"
    echo ""
    echo "Check your account with:"
    echo "  sacctmgr show user $USER"
    echo ""
    echo "Check available partitions with:"
    echo "  sinfo"
    echo ""
    exit 1
fi

if [ -z "$PARTITION" ] || [ "$PARTITION" = "<partition>" ]; then
    echo "=========================================="
    echo "ERROR: SLURM partition not configured."
    echo "=========================================="
    echo ""
    echo "Please set it using one of these methods:"
    echo ""
    echo "Method 1 (Recommended): Use configuration helper"
    echo "  ./scripts/configure_profiling.sh"
    echo ""
    echo "Method 2: Export environment variable"
    echo "  export SLURM_PARTITION='your_partition'"
    echo ""
    echo "Method 3: Create config file"
    echo "  echo \"export SLURM_PARTITION='your_partition'\" >> scripts/profiling_config.sh"
    echo ""
    echo "Check available partitions with:"
    echo "  sinfo"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Account: ${ACCOUNT}"
echo "  Partition: ${PARTITION}"
echo ""

# Create results directory
RESULTS_BASE="${PROJECT_DIR}/results/profiling"
mkdir -p "${RESULTS_BASE}"
PROFILE_RUN_DIR="${RESULTS_BASE}/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROFILE_RUN_DIR}"

# Store job IDs
JOB_IDS=()

echo "Results will be saved to: ${PROFILE_RUN_DIR}"
echo ""

# Function to submit profiling job
submit_profile_job() {
    local nodes=$1
    local script_name=$2
    
    echo "----------------------------------------"
    echo "Submitting profiling job: ${nodes} node(s)"
    echo "Script: ${script_name}"
    echo "----------------------------------------"
    
    # Update account and partition in script (create temporary copy)
    TEMP_SCRIPT="${PROFILE_RUN_DIR}/temp_${script_name}"
    sed "s|<account>|${ACCOUNT}|g; s|<partition>|${PARTITION}|g" \
        "${PROJECT_DIR}/slurm/${script_name}" > "${TEMP_SCRIPT}"
    
    # Submit job (with timeout to prevent hanging)
    echo "Submitting job..."
    OUTPUT=$(timeout 30 sbatch "${TEMP_SCRIPT}" 2>&1)
    EXIT_CODE=$?
    
    # Clean up temp script immediately after submission
    rm -f "${TEMP_SCRIPT}"
    
    if [ $EXIT_CODE -ne 0 ]; then
        if [ $EXIT_CODE -eq 124 ]; then
            echo "ERROR: Job submission timed out (>30 seconds)"
            echo "Check SLURM configuration and try again"
        else
            echo "ERROR: Failed to submit job (exit code: $EXIT_CODE)"
            echo "$OUTPUT"
        fi
        return 1
    fi
    
    JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')
    
    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Failed to extract job ID from output:"
        echo "$OUTPUT"
        return 1
    fi
    
    # Verify JOB_ID looks like a number
    if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid job ID extracted: '$JOB_ID'"
        echo "Full output: $OUTPUT"
        return 1
    fi
    
    JOB_IDS+=(${JOB_ID})
    
    echo "Job ID: ${JOB_ID}"
    echo "Job submitted successfully!"
    echo ""
    
    # Clean up temp script
    rm -f "${TEMP_SCRIPT}"
}

# Submit profiling jobs
echo "Starting job submissions..."
echo ""

# 1-node profiling
submit_profile_job 1 "profile_1node.sbatch"

# 2-node profiling
submit_profile_job 2 "profile_2node.sbatch"

# 4-node profiling
submit_profile_job 4 "profile_4node.sbatch"

# Check if any jobs were submitted successfully
if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "=========================================="
    echo "ERROR: No jobs were submitted successfully!"
    echo "=========================================="
    echo "Please check the errors above and fix the configuration."
    exit 1
fi

# Save job IDs to file
JOB_IDS_FILE="${PROFILE_RUN_DIR}/job_ids.txt"
printf "%s\n" "${JOB_IDS[@]}" > "${JOB_IDS_FILE}"

echo "=========================================="
echo "All profiling jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Job IDs:"
for job_id in "${JOB_IDS[@]}"; do
    echo "  - ${job_id}"
done
echo ""
echo "Job IDs saved to: ${JOB_IDS_FILE}"
echo "Results base directory: ${PROFILE_RUN_DIR}"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To wait for all jobs to complete:"
echo "  for job in \$(cat ${JOB_IDS_FILE}); do srun --jobid=\$job echo \"Job \$job completed\"; done"
echo ""
echo "After jobs complete, analyze results with:"
echo "  python scripts/analyze_profiling.py --results ${PROFILE_RUN_DIR}"
echo ""
echo "Or manually review the profiling outputs in:"
echo "  ${PROJECT_DIR}/results/profiling/"
echo ""

