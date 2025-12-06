#!/bin/bash
# Production Deployment Script
# Validates environment and prepares for production deployment

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================="
echo "Production Deployment Validation"
echo "=============================================="
echo ""

# Check 1: Container
echo "1. Checking container..."
if [ -f "env/project.sif" ]; then
    echo "   ✓ Container exists: env/project.sif"
    SIZE=$(du -h env/project.sif | cut -f1)
    echo "   ✓ Container size: ${SIZE}"
else
    echo "   ✗ Container not found. Building..."
    ./run.sh build
fi

# Check 2: Data
echo ""
echo "2. Checking data..."
if [ -d "data/processed" ] && [ -n "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "   ✓ Processed data exists"
    ls -lh data/processed/*.npz 2>/dev/null | head -3
else
    echo "   ⚠️  No processed data found"
    echo "   Run: cd data && python generate_sample_data.py"
fi

# Check 3: SLURM scripts
echo ""
echo "3. Checking SLURM scripts..."
SLURM_SCRIPTS=(
    "slurm/production_baseline.sbatch"
    "slurm/production_multi_node.sbatch"
)
for script in "${SLURM_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "   ✓ $script"
    else
        echo "   ✗ $script not found"
    fi
done

# Check 4: Results directory
echo ""
echo "4. Checking results directory..."
mkdir -p results
echo "   ✓ Results directory ready"

# Check 5: Python syntax
echo ""
echo "5. Validating Python code..."
if python3 -m py_compile src/train.py src/data.py src/model/dcrnn.py 2>/dev/null; then
    echo "   ✓ All Python files compile"
else
    echo "   ✗ Python syntax errors found"
    exit 1
fi

# Check 6: SLURM account
echo ""
echo "6. Checking SLURM configuration..."
if [ -n "$SLURM_ACCOUNT" ] || grep -q "account=" slurm/production_*.sbatch 2>/dev/null; then
    echo "   ✓ SLURM account configured"
else
    echo "   ⚠️  Update SLURM account in production scripts"
fi

# Check 7: GPU access
echo ""
echo "7. Checking GPU access..."
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        echo "   ✓ GPU access available"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    else
        echo "   ⚠️  nvidia-smi failed (may need --nv flag in container)"
    fi
else
    echo "   ⚠️  nvidia-smi not found (may be normal on login node)"
fi

echo ""
echo "=============================================="
echo "Deployment Validation Complete"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Review PRODUCTION_DEPLOYMENT.md"
echo "  2. Update SLURM account in scripts"
echo "  3. Submit baseline job: sbatch slurm/production_baseline.sbatch"
echo "  4. Monitor: squeue -u \$USER"
echo ""
