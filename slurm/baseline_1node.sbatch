#!/bin/bash
#SBATCH --job-name=dcrnn-1n
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=results/1n_%j.out
#SBATCH --error=results/1n_%j.err

# Load modules if needed
# module load apptainer
# module load cuda/12.1

# Set working directory
cd "${SLURM_SUBMIT_DIR}/.."

# Create results directory for this job
RESULTS_DIR="./results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Run training
./run.sh python -m src.train \
  --data ./data \
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

