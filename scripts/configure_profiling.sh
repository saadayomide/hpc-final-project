#!/bin/bash
# Helper script to configure account and partition for profiling scripts

echo "=========================================="
echo "Phase 3: Profiling Configuration"
echo "=========================================="
echo ""

# Check for available partitions
echo "Available partitions:"
sinfo -o "%P %A %G" 2>/dev/null | head -10
echo ""
echo "Note: For GPU profiling, you'll typically want a partition with GPU resources (gpu:1)."
echo "Recommended: 'gpu-node' for GPU workloads"
echo ""

# Prompt for account
if [ -z "$SLURM_ACCOUNT" ]; then
    echo "Please enter your SLURM account:"
    read -r SLURM_ACCOUNT
fi

# Prompt for partition
if [ -z "$SLURM_PARTITION" ]; then
    echo "Please enter your SLURM partition (e.g., gpu-node):"
    read -r SLURM_PARTITION
fi

# Validate
if [ -z "$SLURM_ACCOUNT" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "ERROR: Account and partition must be set."
    exit 1
fi

# Check if partition exists (more robust check)
PARTITION_EXISTS=false
if sinfo -p "$SLURM_PARTITION" 2>/dev/null | grep -q "$SLURM_PARTITION"; then
    PARTITION_EXISTS=true
fi

# Also check if it's a partial match (for wildcards)
if [ "$PARTITION_EXISTS" = false ]; then
    if sinfo -o "%P" 2>/dev/null | grep -q "$SLURM_PARTITION"; then
        PARTITION_EXISTS=true
    fi
fi

if [ "$PARTITION_EXISTS" = false ]; then
    echo ""
    echo "WARNING: Partition '$SLURM_PARTITION' does not appear to exist."
    echo "Available partitions:"
    sinfo -o "%P" -h 2>/dev/null | sort -u
    echo ""
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        echo "Cancelled. Please run this script again and select a valid partition."
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Account: $SLURM_ACCOUNT"
echo "  Partition: $SLURM_PARTITION"
echo ""

# Export for current session
export SLURM_ACCOUNT
export SLURM_PARTITION

echo "To use this configuration, run:"
echo "  export SLURM_ACCOUNT='$SLURM_ACCOUNT'"
echo "  export SLURM_PARTITION='$SLURM_PARTITION'"
echo "  ./scripts/run_profiling.sh"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  export SLURM_ACCOUNT='$SLURM_ACCOUNT'"
echo "  export SLURM_PARTITION='$SLURM_PARTITION'"
echo ""

# Create a config file for reference
CONFIG_FILE="scripts/profiling_config.sh"
cat > "$CONFIG_FILE" << EOF
#!/bin/bash
# Profiling configuration (auto-generated)
# Source this file before running profiling scripts:
#   source scripts/profiling_config.sh

export SLURM_ACCOUNT='$SLURM_ACCOUNT'
export SLURM_PARTITION='$SLURM_PARTITION'
EOF

chmod +x "$CONFIG_FILE"

echo "Configuration saved to: $CONFIG_FILE"
echo "To use: source $CONFIG_FILE && ./scripts/run_profiling.sh"

