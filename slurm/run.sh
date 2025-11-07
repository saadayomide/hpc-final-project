#!/bin/bash
# SLURM job submission wrapper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOB_SCRIPT="${1:-${SCRIPT_DIR}/single_node.sbatch}"

if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "Error: Job script not found: ${JOB_SCRIPT}"
    exit 1
fi

echo "Submitting job: ${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"

