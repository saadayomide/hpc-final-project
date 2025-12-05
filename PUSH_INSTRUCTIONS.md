# How to Push to GitHub - Step by Step

## Method 1: Personal Access Token (EASIEST)

### Step 1: Create GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "HPC Final Project"
4. Select expiration: 90 days (or No expiration)
5. Check the box: **repo** (this gives full repository access)
6. Click "Generate token"
7. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
   - It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 2: Push from Cluster

Run these commands on the cluster:

```bash
cd /home/user42/hpc-final-project

# Set remote to HTTPS
git remote set-url origin https://github.com/saadayomide/hpc-final-project.git

# Push (you'll be prompted for credentials)
git push origin main
```

When prompted:
- **Username**: `saadayomide`
- **Password**: Paste your Personal Access Token (NOT your GitHub password)

### Alternative: Push from Your Local Machine

If you prefer to push from your own computer:

```bash
# On your local machine
git clone https://github.com/saadayomide/hpc-final-project.git
cd hpc-final-project

# Add cluster as remote
git remote add cluster user42@login1.int.hpcie.labs.faculty.ie.edu:/home/user42/hpc-final-project

# Fetch from cluster
git fetch cluster

# Merge cluster's main branch
git merge cluster/main

# Push to GitHub
git push origin main
```

## Method 2: SSH Key (More Permanent)

If you want to set up SSH for future use:

1. Generate SSH key on cluster:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default location
   # Press Enter twice for no passphrase (or set one)
   ```

2. Display your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. Add to GitHub:
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste the key content
   - Save

4. Push:
   ```bash
   git remote set-url origin git@github.com:saadayomide/hpc-final-project.git
   git push origin main
   ```

## Quick Check

After pushing, verify:
```bash
git log origin/main --oneline
```

You should see your commits there!
