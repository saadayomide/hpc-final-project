#!/bin/bash
#SBATCH --job-name=dcrnn_profile
#SBATCH --output=../results/slurm_%j.out
#SBATCH --error=../results/slurm_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Load modules if needed
# module load apptainer
# module load nvidia-nsight-systems  # or your profiler

# Set working directory
cd "${SLURM_SUBMIT_DIR}/.."

# Run profiling (adjust profiler command as needed)
# Example with nsys:
# nsys profile --output=../results/profile ./run.sh python src/train.py --profile

# Or basic profiling:
./run.sh python src/train.py --profile

