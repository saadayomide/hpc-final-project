#!/bin/bash
#SBATCH --job-name=dcrnn_multi
#SBATCH --output=../results/slurm_%j.out
#SBATCH --error=../results/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

# Load modules if needed
# module load apptainer
# module load openmpi

# Set working directory
cd "${SLURM_SUBMIT_DIR}/.."

# Run the container with multi-node training
# Adjust for your distributed training setup
./run.sh python src/train.py --distributed --nodes=${SLURM_NNODES}

