# System Information

This document describes the cluster system configuration where experiments are run.

## Node Types

### Compute Nodes
- **Type**: [To be filled - e.g., GPU nodes with V100/A100]
- **CPU**: [To be filled]
- **Memory**: [To be filled]
- **GPU**: [To be filled - model, count, memory]
- **Interconnect**: [To be filled - e.g., InfiniBand, Ethernet]

### Login Nodes
- **Type**: [To be filled]
- **CPU**: [To be filled]
- **Memory**: [To be filled]

## Software Environment

### Modules
List of loaded modules:
```
# module list
```

Key modules:
- **Apptainer/Singularity**: [Version]
- **CUDA**: [Version]
- **cuDNN**: [Version]
- **OpenMPI**: [Version] (if using distributed training)

### Container Runtime
- **Apptainer Version**: [To be filled]
- **Base Image**: `nvcr.io/nvidia/pytorch:24.04-py3`
- **Python Version**: 3.10 (from base image)

### Driver Versions
- **NVIDIA Driver**: [Version]
- **CUDA Runtime**: [Version]

### Python Packages
See `env/project.def` for complete list. Key packages:
- PyTorch: [Version from base image]
- NumPy, SciPy, Pandas
- torch-geometric, torch-sparse, torch-scatter
- DGL, dgllife

## SLURM Configuration

### Partitions
- **Partition**: [To be filled]
- **Default Time Limit**: [To be filled]
- **Max Nodes**: [To be filled]

### Resource Limits
- **Max GPUs per job**: [To be filled]
- **Max Memory per node**: [To be filled]
- **Max Walltime**: [To be filled]

## Network Configuration

- **Interconnect**: [To be filled]
- **Bandwidth**: [To be filled]
- **Latency**: [To be filled]

## Storage

- **Home Directory**: [Path]
- **Scratch Directory**: [Path]
- **Shared Data**: [Path]

## Notes

- Update this file with actual system information from your cluster
- Use `module list` to capture module versions
- Use `nvidia-smi` to capture GPU and driver information
- Use `sinfo` to capture partition information

