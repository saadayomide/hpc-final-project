# HPC Final Project: DCRNN Scalability and Performance

## Project Abstract

This project investigates the scalability and performance optimization of Diffusion Convolutional Recurrent Neural Networks (DCRNN) for spatio-temporal graph data on high-performance computing systems. DCRNNs have shown promise in traffic forecasting, epidemiology, and sensor network applications, but their computational demands and memory requirements pose significant challenges for large-scale datasets and real-time inference.

Our approach focuses on developing a containerized, reproducible framework that enables efficient distributed training across multiple GPU nodes. We implement the DCRNN architecture using PyTorch with distributed data parallel and model parallel strategies, optimized for cluster environments with NVIDIA GPUs. The project employs Apptainer containers for reproducibility, ensuring consistent execution across different HPC systems. We leverage SLURM for job scheduling and resource management, with profiling tools to identify performance bottlenecks.

Success metrics include: (1) achieving linear scaling efficiency above 80% when scaling from 1 to 8 GPU nodes, (2) reducing training time by at least 50% compared to single-node baseline through optimized data loading and communication patterns, (3) demonstrating the framework's reproducibility by successfully running on multiple HPC clusters with minimal configuration changes, and (4) maintaining model accuracy within 2% of the baseline implementation. The project will produce a complete, documented codebase that can be easily deployed and reproduced by other researchers.

## Repository Structure

```
repo/
├── src/            # DCRNN code: train.py, data.py, model/, utils/
├── env/            # Apptainer recipe (project.def)
├── slurm/          # sbatch scripts (1N, multi-N, profiling), run.sh wrapper
├── data/           # Dataset + fetch_data.sh + README
├── results/        # CSV, PNG/SVG plots, sacct logs, profiler outputs
├── docs/           # paper.pdf, eurohpc_proposal.pdf, slides.pdf, README.md
├── reproduce.md    # Exact commands + seeds + git hash
├── SYSTEM.md       # Node types, module list, driver/runtime versions
└── run.sh          # Container build and execution wrapper
```

## Quick Start

### Prerequisites
- Apptainer/Singularity installed
- Access to an HPC cluster with GPU nodes
- SLURM workload manager

### Build Container
```bash
./run.sh build
```

### Verify Environment
```bash
./run.sh python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`

### Run Training
```bash
./run.sh python src/train.py
```

### Submit SLURM Job
```bash
cd slurm
./run.sh single_node.sbatch
```

## Documentation

- **reproduce.md**: Detailed reproduction instructions
- **SYSTEM.md**: System configuration and requirements
- **data/README.md**: Dataset information and fetching instructions
- **docs/README.md**: Project documentation

## License

[To be specified]
