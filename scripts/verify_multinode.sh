#!/bin/bash
# Verification script for multi-node DDP training results
# Usage: ./scripts/verify_multinode.sh [job_id_or_results_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================="
echo "Multi-Node DDP Training Verification"
echo "=============================================="
echo ""

# Determine results directory
if [ $# -eq 0 ]; then
    # Find most recent ddp_cpu result
    RESULTS_DIR=$(find results -type d -name "ddp_cpu_*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "${RESULTS_DIR}" ]; then
        echo "ERROR: No ddp_cpu results found. Please specify job ID or results directory."
        echo "Usage: $0 [job_id_or_results_dir]"
        exit 1
    fi
    echo "Found most recent results: ${RESULTS_DIR}"
elif [ -d "$1" ]; then
    RESULTS_DIR="$1"
elif [ -d "results/ddp_cpu_$1" ]; then
    RESULTS_DIR="results/ddp_cpu_$1"
else
    echo "ERROR: Results directory not found: $1"
    exit 1
fi

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: Results directory does not exist: ${RESULTS_DIR}"
    exit 1
fi

echo "Verifying: ${RESULTS_DIR}"
echo ""

# Extract job ID if possible
JOB_ID=$(basename "${RESULTS_DIR}" | sed 's/ddp_cpu_//')
echo "Job ID: ${JOB_ID}"
echo ""

# Verification checklist
PASSED=0
FAILED=0

check_file() {
    local file="$1"
    local desc="$2"
    if [ -f "${RESULTS_DIR}/${file}" ]; then
        echo "✓ ${desc}: ${file}"
        ((PASSED++))
        return 0
    else
        echo "✗ ${desc}: ${file} MISSING"
        ((FAILED++))
        return 1
    fi
}

check_content() {
    local file="$1"
    local pattern="$2"
    local desc="$3"
    if [ -f "${RESULTS_DIR}/${file}" ] && grep -q "${pattern}" "${RESULTS_DIR}/${file}" 2>/dev/null; then
        echo "✓ ${desc}"
        ((PASSED++))
        return 0
    else
        echo "✗ ${desc}"
        ((FAILED++))
        return 1
    fi
}

# Check required files
echo "=== Required Files ==="
check_file "metrics.csv" "Metrics CSV"
check_file "sacct_summary.txt" "SLURM accounting summary"

# Check output files
if [ -f "results/ddp_cpu_${JOB_ID}.out" ]; then
    echo "✓ Job output file: results/ddp_cpu_${JOB_ID}.out"
    ((PASSED++))
else
    echo "⚠ Job output file not found (may be in different location)"
fi

if [ -f "results/ddp_cpu_${JOB_ID}.err" ]; then
    ERR_SIZE=$(stat -f%z "results/ddp_cpu_${JOB_ID}.err" 2>/dev/null || stat -c%s "results/ddp_cpu_${JOB_ID}.err" 2>/dev/null || echo "0")
    if [ "${ERR_SIZE}" -gt 0 ]; then
        echo "⚠ Error file has content: results/ddp_cpu_${JOB_ID}.err"
        echo "  Check for errors: tail results/ddp_cpu_${JOB_ID}.err"
    else
        echo "✓ Error file is empty (no errors)"
        ((PASSED++))
    fi
fi

echo ""

# Check metrics content
if [ -f "${RESULTS_DIR}/metrics.csv" ]; then
    echo "=== Metrics Analysis ==="
    NUM_EPOCHS=$(tail -n +2 "${RESULTS_DIR}/metrics.csv" | wc -l | tr -d ' ')
    echo "Number of epochs completed: ${NUM_EPOCHS}"
    
    if [ "${NUM_EPOCHS}" -ge 1 ]; then
        echo "✓ At least 1 epoch completed"
        ((PASSED++))
        
        # Show last epoch metrics
        echo ""
        echo "Last epoch metrics:"
        tail -n 1 "${RESULTS_DIR}/metrics.csv" | awk -F',' '{
            printf "  Epoch: %s\n", $1
            printf "  Train Loss: %s\n", $2
            printf "  Val Loss: %s\n", $3
            printf "  Val MAE: %s mph\n", $4
            printf "  Val RMSE: %s mph\n", $5
            printf "  Epoch Time: %s s\n", $6
            printf "  Throughput: %s samples/s\n", $7
        }'
    else
        echo "✗ No epochs completed"
        ((FAILED++))
    fi
    echo ""
fi

# Check SLURM accounting
if [ -f "${RESULTS_DIR}/sacct_summary.txt" ]; then
    echo "=== SLURM Accounting ==="
    if grep -q "COMPLETED" "${RESULTS_DIR}/sacct_summary.txt" 2>/dev/null; then
        echo "✓ Job completed successfully"
        ((PASSED++))
        
        # Extract key info
        NODES=$(grep -E "^${JOB_ID}" "${RESULTS_DIR}/sacct_summary.txt" | head -1 | awk '{print $8}' || echo "N/A")
        ELAPSED=$(grep -E "^${JOB_ID}" "${RESULTS_DIR}/sacct_summary.txt" | head -1 | awk '{print $5}' || echo "N/A")
        echo "  Nodes used: ${NODES}"
        echo "  Elapsed time: ${ELAPSED}"
    elif grep -q "FAILED\|CANCELLED\|TIMEOUT" "${RESULTS_DIR}/sacct_summary.txt" 2>/dev/null; then
        echo "✗ Job did not complete successfully"
        grep -E "FAILED\|CANCELLED\|TIMEOUT" "${RESULTS_DIR}/sacct_summary.txt" | head -1
        ((FAILED++))
    else
        echo "⚠ Could not determine job status from sacct"
    fi
    echo ""
fi

# Check for DDP indicators in output
if [ -f "results/ddp_cpu_${JOB_ID}.out" ]; then
    echo "=== DDP Verification ==="
    if grep -q "Using DDP" "results/ddp_cpu_${JOB_ID}.out" 2>/dev/null; then
        echo "✓ DDP initialization detected"
        ((PASSED++))
        
        # Extract DDP info
        DDP_INFO=$(grep "Using DDP" "results/ddp_cpu_${JOB_ID}.out" | head -1)
        echo "  ${DDP_INFO}"
        
        if grep -q "World Size: 2" "results/ddp_cpu_${JOB_ID}.out" 2>/dev/null; then
            echo "✓ 2-node configuration confirmed"
            ((PASSED++))
        fi
    else
        echo "✗ DDP initialization not found in output"
        ((FAILED++))
    fi
    echo ""
fi

# Summary
echo "=============================================="
echo "Verification Summary"
echo "=============================================="
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo ""

if [ ${FAILED} -eq 0 ]; then
    echo "✓ All checks passed! Multi-node training verified."
    exit 0
else
    echo "⚠ Some checks failed. Review the output above."
    exit 1
fi

