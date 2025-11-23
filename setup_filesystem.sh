#!/bin/bash
# Setup script to initialize filesystem structure following HPC guidelines
# This script creates the necessary directories in /project and /scratch

set -e

USERNAME=$(whoami)
PROJECT_DIR="/project/${USERNAME}/hpc-final-project"
SCRATCH_DIR="/scratch/${USERNAME}/hpc-final-project"
HOME_DIR="${HOME}/hpc-final-project"

echo "Setting up filesystem structure for HPC cluster..."
echo ""

# Create project directory structure (persistent data)
echo "Creating /project directories..."
mkdir -p "${PROJECT_DIR}/data/train"
mkdir -p "${PROJECT_DIR}/data/val"
mkdir -p "${PROJECT_DIR}/data/test"
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/checkpoints"

# Create scratch directory structure (temporary files)
echo "Creating /scratch directories..."
mkdir -p "${SCRATCH_DIR}/tmp"
mkdir -p "${SCRATCH_DIR}/.apptainer_cache"
mkdir -p "${SCRATCH_DIR}/.apptainer_tmp"

echo ""
echo "Filesystem structure created:"
echo ""
echo "/home/${USERNAME}/hpc-final-project/  (code only)"
echo "  ├── src/           (source code)"
echo "  ├── slurm/         (job scripts)"
echo "  ├── env/           (container definition)"
echo "  └── run.sh         (wrapper script)"
echo ""
echo "${PROJECT_DIR}/  (persistent data)"
echo "  ├── data/          (datasets)"
echo "  ├── results/       (training results, logs)"
echo "  └── checkpoints/   (model checkpoints)"
echo ""
echo "${SCRATCH_DIR}/  (temporary files)"
echo "  ├── tmp/           (job-specific temp files)"
echo "  └── .apptainer_*   (container cache)"
echo ""
echo "Next steps:"
echo "1. Copy datasets to: ${PROJECT_DIR}/data/"
echo "2. Run: ./run.sh build  (to build container)"
echo "3. Submit jobs from: ${HOME_DIR}/slurm/"
echo ""
echo "To check disk usage:"
echo "  du -sh ${PROJECT_DIR}/*"
echo "  du -sh ${SCRATCH_DIR}/*"

