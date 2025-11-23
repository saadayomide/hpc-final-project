# Filesystem Usage Guidelines

This project follows the HPC cluster filesystem guidelines to ensure stable operation for all users.

## Filesystem Layout

### `/home/<username>/hpc-final-project/` - Code Only
**Purpose:** Small, shared filesystem for code and configuration

**Contains:**
- Source code (`src/`)
- Job scripts (`slurm/`)
- Container definition (`env/project.def`)
- Wrapper scripts (`run.sh`)
- Documentation and README files

**Do NOT store here:**
- ❌ Large datasets
- ❌ Model checkpoints
- ❌ Training results/logs
- ❌ Conda/virtualenv installations
- ❌ Large output files

### `/project/<username>/hpc-final-project/` - Persistent Data
**Purpose:** Long-term storage for project data

**Contains:**
- `data/` - Datasets (train/val/test splits)
- `results/` - Training results, metrics, logs
- `checkpoints/` - Saved model checkpoints

**Usage:**
```bash
# Copy datasets here
cp -r /path/to/dataset /project/user42/hpc-final-project/data/

# Results are automatically saved here by SLURM scripts
# Checkpoints are saved here during training
```

### `/scratch/<username>/hpc-final-project/` - Temporary Files
**Purpose:** High-volume temporary data that can be recomputed

**Contains:**
- `tmp/` - Job-specific temporary outputs
- `.apptainer_cache/` - Apptainer build cache
- `.apptainer_tmp/` - Apptainer temporary build files

**Note:** Files in `/scratch` may be periodically cleaned up. Do not rely on it for long-term storage.

## Setup

Run the setup script to create the directory structure:

```bash
./setup_filesystem.sh
```

This will create all necessary directories in `/project` and `/scratch`.

## SLURM Scripts

All SLURM scripts (`slurm/*.sbatch`) have been configured to:
- Write results to `/project/user42/hpc-final-project/results/`
- Read data from `/project/user42/hpc-final-project/data/`
- Use `/scratch` for temporary files

## Container Build

The `run.sh` script automatically uses `/scratch` for Apptainer cache to avoid filling up `/home`:

```bash
./run.sh build
```

The container image itself (`env/project.sif`) stays in the repo directory (it's code-related and relatively small).

## Checking Disk Usage

Monitor your disk usage regularly:

```bash
# Check /home usage
du -sh ~/hpc-final-project/*

# Check /project usage
du -sh /project/user42/hpc-final-project/*

# Check /scratch usage
du -sh /scratch/user42/hpc-final-project/*
```

## Moving Existing Data

If you have large files in `/home`, move them:

```bash
# Move datasets
mv ~/hpc-final-project/data/large_dataset /project/user42/hpc-final-project/data/

# Move results
mv ~/hpc-final-project/results/* /project/user42/hpc-final-project/results/

# Move checkpoints
mv ~/hpc-final-project/checkpoints/* /project/user42/hpc-final-project/checkpoints/
```

## Best Practices

1. **Keep `/home` small** - Only code and small config files
2. **Use `/project` for persistent data** - Datasets, results, checkpoints
3. **Use `/scratch` for temporary files** - Job outputs, cache, temp data
4. **Monitor disk usage** - Check regularly to avoid filling shared storage
5. **Clean up `/scratch`** - Remove temporary files when done

## Troubleshooting

**"No space left on device" error:**
- Check if `/home` is full: `df -h /home`
- Move large files to `/project` or `/scratch`
- Clean up Apptainer cache: `rm -rf /scratch/user42/.apptainer_cache/*`

**Container build fails:**
- The build script automatically uses `/scratch` for cache
- If issues persist, check `/scratch` has space: `df -h /scratch`

**Can't find data/results:**
- Ensure you've run `./setup_filesystem.sh`
- Check paths in SLURM scripts match your username
- Verify data is in `/project/user42/hpc-final-project/data/`
