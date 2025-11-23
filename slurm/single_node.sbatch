#!/bin/bash
#SBATCH --job-name=dcrnn_train
#SBATCH --output=/project/user42/hpc-final-project/results/slurm_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Load modules if needed
# module load apptainer

# Set filesystem paths following HPC guidelines:
# /home - code only (repo location)
# /project - persistent data (datasets, results, checkpoints)
# /scratch - temporary files (job-specific temp outputs)
PROJECT_DIR="/project/user42/hpc-final-project"
SCRATCH_DIR="/scratch/user42/hpc-final-project"
HOME_DIR="${SLURM_SUBMIT_DIR}/.."

# Set working directory to home (code location)
cd "${HOME_DIR}"

# Create directories in /project for persistent data
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${SCRATCH_DIR}/tmp"

# Use /project for data directory (persistent datasets)
DATA_DIR="${PROJECT_DIR}/data"
RESULTS_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Run the container with training script
./run.sh python src/train.py \
  --data "${DATA_DIR}" \
  --results "${RESULTS_DIR}"

