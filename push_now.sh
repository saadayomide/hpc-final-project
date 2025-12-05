#!/bin/bash
# Quick push script - run this after you have your GitHub token

echo "=============================================="
echo "Pushing to GitHub"
echo "=============================================="

cd /home/user42/hpc-final-project

# Set remote to HTTPS
git remote set-url origin https://github.com/saadayomide/hpc-final-project.git

echo ""
echo "Ready to push! You'll be prompted for:"
echo "  Username: saadayomide"
echo "  Password: [Your Personal Access Token]"
echo ""
echo "Press Enter to continue, or Ctrl+C to cancel..."
read

# Push
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "Verify at: https://github.com/saadayomide/hpc-final-project"
else
    echo ""
    echo "❌ Push failed. Check your token and try again."
    echo "See PUSH_INSTRUCTIONS.md for help."
fi
