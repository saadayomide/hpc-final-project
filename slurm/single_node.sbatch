#!/bin/bash
#SBATCH --job-name=dcrnn_train
#SBATCH --output=../results/slurm_%j.out
#SBATCH --error=../results/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Load modules if needed
# module load apptainer

# Set working directory
cd "${SLURM_SUBMIT_DIR}/.."

# Run the container with training script
./run.sh python src/train.py

