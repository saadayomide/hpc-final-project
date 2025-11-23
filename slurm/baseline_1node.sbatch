#!/bin/bash
#SBATCH --job-name=dcrnn-1n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/project/user42/hpc-final-project/results/1n_%j.out
#SBATCH --error=/project/user42/hpc-final-project/results/1n_%j.err

# Load modules if needed
# module load apptainer
# module load cuda/12.1

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

# Create results directory for this job in /project
RESULTS_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Use /project for data directory (persistent datasets)
DATA_DIR="${PROJECT_DIR}/data"

# Run training
./run.sh python -m src.train \
  --data "${DATA_DIR}" \
  --epochs 1 \
  --batch-size 32 \
  --precision bf16 \
  --num-workers 4 \
  --results "${RESULTS_DIR}" \
  --seed 42 \
  --monitor-gpu \
  --monitor-cpu

# Save sacct summary
echo "Job ID: ${SLURM_JOB_ID}"
echo "Saving sacct summary..."
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUs,AllocGRES,NodeList > "${RESULTS_DIR}/sacct_summary.txt"

echo "Job completed. Results saved to ${RESULTS_DIR}"

