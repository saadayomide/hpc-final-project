#!/bin/bash
###############################################################################
# Release Preparation Script
# 
# Prepares the repository for final submission:
# 1. Validates required files exist
# 2. Generates PDFs from markdown documentation
# 3. Creates release tag
# 4. Generates final checklist
#
# Usage: ./scripts/prepare_release.sh [--tag v1.0-course-teamname]
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# Default tag name
TAG_NAME="${1:-v1.0-course-teamname}"

echo "=============================================="
echo "HPC Final Project - Release Preparation"
echo "=============================================="
echo "Project: ${PROJECT_DIR}"
echo "Tag: ${TAG_NAME}"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }

###############################################################################
# Step 1: Validate Required Files
###############################################################################
echo "Step 1: Validating required files..."
echo "--------------------------------------"

ERRORS=0

# Core code
for f in src/train.py src/data.py src/model/dcrnn.py; do
    if [ -f "$f" ]; then pass "$f"; else fail "$f"; ((ERRORS++)); fi
done

# Environment
for f in env/project.def env/load_modules.sh run.sh; do
    if [ -f "$f" ]; then pass "$f"; else fail "$f"; ((ERRORS++)); fi
done

# Slurm scripts
if ls slurm/*.sbatch 1> /dev/null 2>&1; then
    pass "slurm/*.sbatch scripts"
else
    fail "No slurm scripts found"; ((ERRORS++))
fi

# Data
for f in data/README.md; do
    if [ -f "$f" ]; then pass "$f"; else fail "$f"; ((ERRORS++)); fi
done

if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
    pass "data/processed/ (has data)"
else
    warn "data/processed/ (empty or missing - needs data generation)"
fi

# Documentation
for f in docs/paper.md docs/eurohpc_proposal.md docs/pitch.md; do
    if [ -f "$f" ]; then pass "$f"; else fail "$f"; ((ERRORS++)); fi
done

for f in reproduce.md SYSTEM.md README.md; do
    if [ -f "$f" ]; then pass "$f"; else fail "$f"; ((ERRORS++)); fi
done

# Results
if [ -d "results" ] && ls results/*.csv 1> /dev/null 2>&1; then
    pass "results/ (has CSV files)"
else
    warn "results/ (no CSV files - need to run experiments)"
fi

if ls results/scaling/*.png 1> /dev/null 2>&1; then
    pass "results/scaling/ (has plots)"
else
    warn "results/scaling/ (no plots - need to run analysis)"
fi

echo ""
if [ $ERRORS -gt 0 ]; then
    fail "Found $ERRORS missing required files!"
    echo "Please create missing files before release."
    echo ""
fi

###############################################################################
# Step 2: Generate PDFs from Markdown
###############################################################################
echo "Step 2: Generating PDFs..."
echo "--------------------------------------"

# Check for pandoc
if command -v pandoc &> /dev/null; then
    pass "pandoc found"
    
    # Paper PDF
    echo "  Generating paper.pdf..."
    if pandoc docs/paper.md -o docs/paper.pdf \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        --highlight-style=tango 2>/dev/null; then
        pass "docs/paper.pdf generated"
    else
        # Try with pdflatex
        if pandoc docs/paper.md -o docs/paper.pdf \
            --pdf-engine=pdflatex \
            -V geometry:margin=1in 2>/dev/null; then
            pass "docs/paper.pdf generated (pdflatex)"
        else
            # Fallback: create without PDF engine
            warn "Could not generate paper.pdf - missing LaTeX engine"
            echo "  Install: apt install texlive-xetex OR texlive-latex-base"
        fi
    fi
    
    # Proposal PDF
    echo "  Generating eurohpc_proposal.pdf..."
    if pandoc docs/eurohpc_proposal.md -o docs/eurohpc_proposal.pdf \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=11pt 2>/dev/null; then
        pass "docs/eurohpc_proposal.pdf generated"
    elif pandoc docs/eurohpc_proposal.md -o docs/eurohpc_proposal.pdf \
        --pdf-engine=pdflatex \
        -V geometry:margin=1in 2>/dev/null; then
        pass "docs/eurohpc_proposal.pdf generated (pdflatex)"
    else
        warn "Could not generate eurohpc_proposal.pdf"
    fi
    
    # Pitch slides (if reveal.js or beamer available)
    echo "  Generating pitch slides..."
    if pandoc docs/pitch.md -o docs/pitch.pdf \
        -t beamer \
        -V theme:Madrid \
        --pdf-engine=xelatex 2>/dev/null; then
        pass "docs/pitch.pdf generated (beamer)"
    else
        warn "Could not generate pitch.pdf - consider using reveal.js or Google Slides"
    fi
    
else
    warn "pandoc not found - cannot generate PDFs"
    echo "  Install: apt install pandoc texlive-xetex"
    echo "  Or use online converter: https://pandoc.org/try/"
fi

echo ""

###############################################################################
# Step 3: Update SYSTEM.md with actual system info
###############################################################################
echo "Step 3: Recording system information..."
echo "--------------------------------------"

# Capture current module list if available
if command -v module &> /dev/null; then
    echo "Recording module list..."
    module list 2>&1 > env/modules_used.txt 2>&1 || true
    pass "Saved module list to env/modules_used.txt"
fi

# Capture Python/CUDA versions
cat > env/versions.txt << EOF
# Environment Versions (captured $(date))

Python: $(python --version 2>&1 || echo "N/A")
PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA (PyTorch): $(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
numpy: $(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "N/A")

nvidia-smi:
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")
EOF
pass "Saved versions to env/versions.txt"

echo ""

###############################################################################
# Step 4: Run final checks
###############################################################################
echo "Step 4: Running final checks..."
echo "--------------------------------------"

# Check for uncommitted changes
if command -v git &> /dev/null && [ -d ".git" ]; then
    if [ -n "$(git status --porcelain)" ]; then
        warn "Uncommitted changes detected"
        echo "  Run: git add -A && git commit -m 'Final release preparation'"
    else
        pass "No uncommitted changes"
    fi
    
    # Check if tag exists
    if git rev-parse "${TAG_NAME}" >/dev/null 2>&1; then
        warn "Tag ${TAG_NAME} already exists"
    else
        pass "Tag ${TAG_NAME} is available"
    fi
fi

# Check for large files
LARGE_FILES=$(find . -type f -size +50M -not -path "./.git/*" -not -path "./env/*.sif" 2>/dev/null)
if [ -n "$LARGE_FILES" ]; then
    warn "Large files detected (>50MB):"
    echo "$LARGE_FILES" | sed 's/^/    /'
    echo "  Consider adding to .gitignore if not needed"
else
    pass "No large files outside container"
fi

echo ""

###############################################################################
# Step 5: Generate Submission Checklist
###############################################################################
echo "Step 5: Generating submission checklist..."
echo "--------------------------------------"

CHECKLIST_FILE="SUBMISSION_CHECKLIST.md"
cat > "${CHECKLIST_FILE}" << 'EOF'
# Submission Checklist

## Code & Repository
- [ ] `src/` contains all source code
- [ ] `env/project.def` Apptainer recipe is complete
- [ ] `slurm/` contains working job scripts
- [ ] `run.sh` wrapper script works
- [ ] Code runs on ≥2 nodes under SLURM

## Reproducibility
- [ ] `reproduce.md` has exact commands
- [ ] Random seeds are fixed (42)
- [ ] Versions documented in `SYSTEM.md`
- [ ] Container recipe (`env/project.def`) is complete

## Performance Evidence
- [ ] Strong scaling results (1, 2, 4+ nodes)
- [ ] Weak scaling results
- [ ] Plots in `results/scaling/` (PNG + SVG)
- [ ] `sacct` logs saved for jobs
- [ ] Profiling data collected

## Documentation
- [ ] Paper (4-6 pages) in `docs/paper.pdf`
- [ ] EuroHPC proposal (6-8 pages) in `docs/eurohpc_proposal.pdf`
- [ ] 5-slide pitch in `docs/` (PDF or slides)
- [ ] `SYSTEM.md` with node types, modules, versions

## Final Steps
- [ ] All changes committed
- [ ] Release tag created: `git tag -a v1.0-course-teamname -m "Final submission"`
- [ ] Tag pushed: `git push origin v1.0-course-teamname`
- [ ] Verify tag on GitHub/GitLab

## Optional but Recommended
- [ ] Container `.sif` file built and tested
- [ ] Sample data in `data/` for quick testing
- [ ] Screenshots of successful runs
EOF

pass "Created ${CHECKLIST_FILE}"

echo ""

###############################################################################
# Summary
###############################################################################
echo "=============================================="
echo "Release Preparation Complete"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Review and complete SUBMISSION_CHECKLIST.md"
echo "  2. Commit all changes:"
echo "     git add -A && git commit -m 'Final release preparation'"
echo ""
echo "  3. Create release tag:"
echo "     git tag -a ${TAG_NAME} -m 'HPC Final Project Submission'"
echo ""
echo "  4. Push to remote:"
echo "     git push origin main"
echo "     git push origin ${TAG_NAME}"
echo ""
echo "  5. Verify on GitHub/GitLab that tag appears in Releases"
echo ""
echo "=============================================="


