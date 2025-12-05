#!/bin/bash
# Script to aggregate profiling results from multiple node configurations
# Usage: ./scripts/aggregate_profiling_results.sh [base_results_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================="
echo "Aggregating Profiling Results"
echo "=============================================="
echo ""

# Determine base results directory
if [ $# -eq 0 ]; then
    # Default: look for profiling results
    BASE_DIR="${PROJECT_DIR}/results/profiling"
else
    BASE_DIR="$1"
fi

if [ ! -d "${BASE_DIR}" ]; then
    echo "ERROR: Results directory not found: ${BASE_DIR}"
    echo "Usage: $0 [base_results_dir]"
    exit 1
fi

echo "Base directory: ${BASE_DIR}"
echo ""

# Find all profiling result directories
SCENARIOS=("1node" "2node" "4node")
FOUND_SCENARIOS=()

for scenario in "${SCENARIOS[@]}"; do
    # Look for scenario directories
    SCENARIO_DIRS=$(find "${BASE_DIR}" -type d -name "*${scenario}*" 2>/dev/null | head -1)
    
    if [ -n "${SCENARIO_DIRS}" ]; then
        # Find most recent run in scenario
        LATEST_RUN=$(find "${SCENARIO_DIRS}" -type d -name "run_*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        
        if [ -n "${LATEST_RUN}" ] && [ -d "${LATEST_RUN}" ]; then
            FOUND_SCENARIOS+=("${scenario}:${LATEST_RUN}")
            echo "Found ${scenario}: ${LATEST_RUN}"
        fi
    fi
done

if [ ${#FOUND_SCENARIOS[@]} -eq 0 ]; then
    echo "ERROR: No profiling results found in ${BASE_DIR}"
    echo "Please run profiling jobs first."
    exit 1
fi

echo ""
echo "Found ${#FOUND_SCENARIOS[@]} scenario(s) with results"
echo ""

# Analyze each scenario
echo "Analyzing individual scenarios..."
echo ""

for scenario_info in "${FOUND_SCENARIOS[@]}"; do
    IFS=':' read -r scenario result_dir <<< "${scenario_info}"
    
    echo "----------------------------------------"
    echo "Analyzing: ${scenario}"
    echo "Directory: ${result_dir}"
    echo "----------------------------------------"
    
    # Run analysis if metrics exist
    if [ -f "${result_dir}/metrics.csv" ] || [ -f "${result_dir}/gpu_monitor.csv" ] || [ -f "${result_dir}/cpu_monitor.csv" ]; then
        python scripts/analyze_profiling.py --results "${result_dir}" --output "${result_dir}" || {
            echo "Warning: Analysis failed for ${scenario}"
        }
    else
        echo "Warning: No metrics found in ${result_dir}"
    fi
    
    echo ""
done

# Generate aggregated bottleneck analysis
echo "=============================================="
echo "Generating Bottleneck Analysis Document"
echo "=============================================="
echo ""

OUTPUT_DOC="${PROJECT_DIR}/docs/BOTTLENECK_ANALYSIS.md"

python scripts/generate_bottleneck_analysis.py \
    --results "${BASE_DIR}" \
    --output "${OUTPUT_DOC}" || {
    echo "Warning: Failed to generate bottleneck analysis document"
    exit 1
}

echo ""
echo "=============================================="
echo "Aggregation Complete"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Individual analyses: ${BASE_DIR}"
echo "  - Bottleneck analysis: ${OUTPUT_DOC}"
echo ""
echo "Next steps:"
echo "  1. Review ${OUTPUT_DOC}"
echo "  2. Check individual profiling reports in each scenario directory"
echo "  3. View Nsight Systems traces if available"
echo ""

