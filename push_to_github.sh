#!/bin/bash
# Script to push changes to GitHub
# This requires authentication setup

echo "=============================================="
echo "Pushing HPC Final Project to GitHub"
echo "=============================================="

cd /home/user42/hpc-final-project

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: ${CURRENT_BRANCH}"

# Show status
echo ""
echo "Git status:"
git status --short

# Show commits to push
echo ""
echo "Commits to push:"
git log origin/main..HEAD --oneline 2>/dev/null || git log --oneline -5

echo ""
echo "=============================================="
echo "To push, choose one of these methods:"
echo ""
echo "Method 1: Using Personal Access Token (HTTPS)"
echo "  git remote set-url origin https://github.com/saadayomide/hpc-final-project.git"
echo "  git push origin main"
echo "  (When prompted, use your GitHub username and a Personal Access Token as password)"
echo ""
echo "Method 2: Using SSH Key"
echo "  1. Generate SSH key: ssh-keygen -t ed25519 -C 'your_email@example.com'"
echo "  2. Add to GitHub: cat ~/.ssh/id_ed25519.pub"
echo "  3. Then: git remote set-url origin git@github.com:saadayomide/hpc-final-project.git"
echo "  4. Then: git push origin main"
echo ""
echo "Method 3: Manual push from local machine"
echo "  git clone https://github.com/saadayomide/hpc-final-project.git"
echo "  cd hpc-final-project"
echo "  git remote add cluster user42@login1.int.hpcie.labs.faculty.ie.edu:/home/user42/hpc-final-project"
echo "  git fetch cluster"
echo "  git merge cluster/main"
echo "  git push origin main"
echo "=============================================="
