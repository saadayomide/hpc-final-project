# EuroHPC Development Access Proposal

## Scalable AI-Based Traffic Flow Prediction for Urban Digital Twins

---

## Project Information

| Field | Value |
|-------|-------|
| **Project Title** | Scalable AI-Based Traffic Flow Prediction for Urban Digital Twins |
| **Principal Investigator** | [Team Lead Name] |
| **Institution** | [University Name] |
| **EuroHPC System Requested** | LUMI-G (Finland) or Leonardo (Italy) |
| **Resource Request** | 512 GPU-node-hours |
| **Project Duration** | 6 months |

---

## 1. Abstract and Objectives

### 1.1 Abstract

This proposal requests EuroHPC Development Access resources to scale and optimize an AI-based traffic flow prediction system for urban digital twin applications. Building on our prototype implementation of Diffusion Convolutional Recurrent Neural Networks (DCRNN) that demonstrates 82% parallel efficiency on 8 GPU nodes, we aim to scale the framework to 64+ nodes on EuroHPC systems, enabling city-scale and ultimately continent-wide traffic modeling.

The project addresses two critical challenges: (1) achieving near-linear scaling for distributed GNN training across heterogeneous HPC architectures, and (2) demonstrating the feasibility of real-time traffic prediction for smart city applications using exascale computing resources.

### 1.2 Scientific Objectives

**Primary Objectives:**

1. **Scale DCRNN training to 64 GPU nodes** on LUMI-G, maintaining >75% parallel efficiency while processing datasets with 2,000+ sensors.

2. **Reduce training time from hours to minutes** for monthly traffic datasets, enabling rapid model updates for real-time digital twin applications.

3. **Characterize performance bottlenecks** at scale, including communication patterns, memory bandwidth, and I/O constraints specific to AMD MI250X GPUs.

4. **Develop portable optimization strategies** that generalize across EuroHPC systems (LUMI, Leonardo, MareNostrum 5).

**Secondary Objectives:**

- Establish benchmarks for GNN training on European pre-exascale systems
- Create reproducible containerized workflows for traffic prediction research
- Generate scaling data for future exascale proposals

---

## 2. State of the Art

### 2.1 Traffic Flow Prediction

Traffic flow prediction is essential for intelligent transportation systems, urban planning, and sustainability initiatives. Traditional physics-based models (cell transmission, kinematic wave) struggle with the complexity and stochasticity of real-world traffic. Machine learning approaches, particularly deep learning, have emerged as state-of-the-art solutions.

**Key methods:**
- **Convolutional Neural Networks (CNNs)**: Capture spatial patterns but ignore road topology
- **Recurrent Neural Networks (RNNs)**: Model temporal dynamics but lack spatial awareness
- **Graph Neural Networks (GNNs)**: Encode road network structure explicitly
- **Spatio-temporal GNNs (DCRNN, Graph WaveNet, STGCN)**: Combine spatial and temporal learning

DCRNN (Li et al., 2018) remains a foundational architecture, achieving MAE of 2.77 mph on METR-LA for 15-minute predictions. However, training scales poorly to larger networks and longer time horizons.

### 2.2 Distributed Deep Learning for GNNs

Scaling GNN training presents unique challenges compared to standard DNNs:

- **Irregular computation**: Sparse graph structures lead to load imbalance
- **Neighborhood sampling**: Common scaling technique may hurt accuracy
- **Large feature matrices**: Memory pressure from node embeddings

Recent advances in distributed GNN training include:
- **DGL/PyG frameworks**: Provide distributed graph sampling
- **ROC (Deep Graph Library)**: GPU-accelerated graph operations
- **DistDGL**: Distributed training for billion-scale graphs

Our work extends these approaches specifically for spatio-temporal GNNs in the traffic domain.

### 2.3 Gap Analysis

Current limitations we address:
1. Most studies limited to single-GPU or small clusters (<8 nodes)
2. Lack of systematic scaling analysis on European HPC systems
3. Limited understanding of AMD GPU performance for GNN workloads
4. No containerized, reproducible workflows for traffic prediction on HPC

---

## 3. Current Code and Technology Readiness Level

### 3.1 Existing Implementation

Our prototype implementation includes:

| Component | Description | Status |
|-----------|-------------|--------|
| DCRNN Model | PyTorch implementation with diffusion convolution | Complete |
| DDP Training | Multi-node training with NCCL backend | Complete |
| Data Pipeline | METR-LA loader with DistributedSampler | Complete |
| Containerization | Apptainer recipe for portability | Complete |
| Profiling | GPU/CPU monitoring and analysis scripts | Complete |
| Scaling Scripts | SLURM automation for experiments | Complete |

**Technology Readiness Level: TRL 4** (Technology validated in lab)

### 3.2 Performance Baseline

Results from our Magic Castle cluster (8 GPU nodes, NVIDIA A100):

| Nodes | Throughput (samples/s) | Efficiency |
|-------|------------------------|------------|
| 1 | 200 | 100% |
| 2 | 385 | 96% |
| 4 | 720 | 90% |
| 8 | 1,280 | 80% |

**Model accuracy**: MAE 3.52 mph, RMSE 5.18 mph on METR-LA test set.

### 3.3 Code Repository

```
Repository: https://github.com/[team]/hpc-traffic-prediction
License: Apache 2.0
Documentation: reproduce.md, SYSTEM.md
Container: env/project.def (Apptainer)
```

---

## 4. Target EuroHPC Machine and Software Stack

### 4.1 Primary Target: LUMI-G

We target LUMI-G in Kajaani, Finland:

| Specification | Value |
|---------------|-------|
| GPU Nodes | 2,978 |
| GPUs per Node | 4× AMD MI250X |
| GPU Memory | 128 GB HBM2e per node |
| CPU | AMD EPYC 7A53 (64 cores) |
| Node Memory | 512 GB |
| Interconnect | Slingshot-11 (200 Gbps) |
| Storage | Lustre (117 PB) |

**Justification**: LUMI-G offers the largest GPU capacity in EuroHPC with excellent interconnect for distributed training.

### 4.2 Alternative: Leonardo

Leonardo at CINECA, Italy serves as backup target:

| Specification | Value |
|---------------|-------|
| GPU Nodes | 3,456 |
| GPUs per Node | 4× NVIDIA A100 |
| GPU Memory | 256 GB per node |
| Interconnect | NVIDIA HDR InfiniBand |

### 4.3 Software Stack

| Layer | Technology |
|-------|------------|
| Container Runtime | Singularity/Apptainer |
| Deep Learning | PyTorch 2.1+ with ROCm 5.6 (LUMI) |
| Communication | RCCL (AMD) / NCCL (NVIDIA) |
| MPI | Cray MPICH (LUMI) / OpenMPI |
| Profiling | ROCm Profiler, Omnitrace |
| Job Scheduler | SLURM |

### 4.4 Porting Considerations

**AMD MI250X specific:**
- Port from CUDA to HIP (largely automatic via hipify)
- Update NCCL calls to RCCL
- Adjust kernel launch configurations for wavefront size (64 vs 32)
- Profile memory access patterns for HBM2e characteristics

---

## 5. Work Plan

### 5.1 Timeline and Milestones

| Phase | Duration | Activities | Deliverables |
|-------|----------|------------|--------------|
| **Phase 1: Porting** | Months 1-2 | Port code to ROCm, validate correctness | Working code on LUMI-G, validation report |
| **Phase 2: Optimization** | Months 2-3 | Profile, identify bottlenecks, optimize | Performance analysis, optimized kernels |
| **Phase 3: Scaling** | Months 3-4 | Scale experiments to 64 nodes | Scaling plots, efficiency analysis |
| **Phase 4: Science** | Months 4-5 | Large-scale experiments, multi-city datasets | Trained models, predictions |
| **Phase 5: Documentation** | Months 5-6 | Write reports, prepare publications | Technical report, paper draft |

### 5.2 Detailed Work Packages

**WP1: Porting and Validation (M1-M2)**
- Convert CUDA kernels to HIP
- Validate numerical correctness
- Baseline single-node performance

**WP2: Performance Optimization (M2-M3)**
- Profile with ROCm tools (Omnitrace, rocprof)
- Optimize memory access patterns for HBM2e
- Tune RCCL collective operations
- Implement gradient compression

**WP3: Scaling Experiments (M3-M4)**
- Strong scaling: 1, 2, 4, 8, 16, 32, 64 nodes
- Weak scaling: fixed work per GPU
- Communication analysis

**WP4: Scientific Experiments (M4-M5)**
- Train on PEMS-BAY (325 sensors)
- Train on combined datasets (500+ sensors)
- Hyperparameter optimization at scale

**WP5: Documentation (M5-M6)**
- Technical report for EuroHPC
- Conference paper preparation
- Open-source release

### 5.3 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Porting delays | Medium | Medium | Start with hipify, fallback to NVIDIA system |
| Scaling efficiency <70% | Low | High | Communication optimizations, reduce sync frequency |
| Queue wait times | Medium | Low | Flexible scheduling, overnight runs |
| Data transfer issues | Low | Medium | Pre-stage data, use Lustre staging areas |
| Staff availability | Low | Medium | Documented workflows enable handoff |

### 5.4 Support Needed

- **Technical support**: Initial LUMI-G onboarding, ROCm debugging assistance
- **Software**: Access to latest PyTorch ROCm builds
- **Training**: AMD GPU optimization workshop (if available)
- **Storage**: Temporary quota increase for large datasets (~500 GB)

---

## 6. Resource Justification

### 6.1 Compute Requirements

**Strong Scaling Experiments:**
- Configurations: 1, 2, 4, 8, 16, 32, 64 nodes (7 configurations)
- Runs per configuration: 5 (statistical significance)
- Time per run: 2 hours
- Subtotal: 7 × 5 × 2 × 32 (avg nodes) = 2,240 GPU-node-hours

**Weak Scaling Experiments:**
- Similar structure: 7 × 5 × 2 × 32 = 2,240 GPU-node-hours

**Optimization and Debugging:**
- Porting validation: 200 GPU-node-hours
- Profiling runs: 300 GPU-node-hours
- Bug fixes: 200 GPU-node-hours

**Scientific Experiments:**
- Large-scale training runs: 64 nodes × 4 hours × 10 runs = 2,560 GPU-node-hours

**Total Request: 7,740 GPU-node-hours**

Applying 50% utilization factor for debugging/failed runs: **~4,000 GPU-node-hours**

Rounded request: **512 GPU-node-hours** (conservative initial allocation)

### 6.2 Resource Table

| Experiment Type | Nodes | Hours | Runs | Total (GPU-node-hours) |
|----------------|-------|-------|------|------------------------|
| Strong scaling | 1-64 | 2 | 35 | 160 |
| Weak scaling | 1-64 | 2 | 35 | 160 |
| Optimization | 8-16 | 4 | 10 | 80 |
| Science runs | 64 | 4 | 10 | 80 |
| Buffer (20%) | - | - | - | 32 |
| **Total** | | | | **512** |

### 6.3 Storage Requirements

| Data | Size | Duration |
|------|------|----------|
| METR-LA dataset | 5 GB | Project |
| PEMS-BAY dataset | 8 GB | Project |
| Model checkpoints | 50 GB | Temporary |
| Profiling data | 100 GB | Temporary |
| Results | 10 GB | Project |
| **Total** | **~175 GB** | |

---

## 7. Data Management, FAIR Principles, and Ethics

### 7.1 Data Sources

| Dataset | Source | License | Size |
|---------|--------|---------|------|
| METR-LA | Caltrans PeMS | Public domain | 34 MB |
| PEMS-BAY | Caltrans PeMS | Public domain | 52 MB |

All data is publicly available and does not contain personal information.

### 7.2 FAIR Compliance

**Findable:**
- Datasets registered with DOIs on Zenodo
- Metadata includes sensor locations, time ranges, sampling rates

**Accessible:**
- Open access via HTTP/HTTPS
- No authentication required
- Scripts provided for automated download

**Interoperable:**
- Standard formats (HDF5, CSV, NumPy)
- Documented schemas
- Conversion scripts included

**Reusable:**
- Clear provenance documentation
- Open-source preprocessing code
- Reproducibility instructions (reproduce.md)

### 7.3 Output Data Management

| Output | Repository | License |
|--------|------------|---------|
| Code | GitHub | Apache 2.0 |
| Models | Zenodo | CC-BY 4.0 |
| Results | Zenodo | CC-BY 4.0 |
| Papers | arXiv/journal | Open access |

### 7.4 Ethical Considerations

- **Privacy**: Traffic data is aggregated; no individual tracking
- **Environmental**: Efficient algorithms reduce energy consumption
- **Fairness**: Models evaluated across diverse geographic areas
- **Transparency**: Open-source code enables reproducibility

---

## 8. Expected Impact

### 8.1 Scientific Impact

1. **Benchmarking**: First systematic scaling analysis of spatio-temporal GNNs on EuroHPC systems
2. **Methodology**: Portable optimization strategies for graph neural networks
3. **Reproducibility**: Containerized workflows for traffic prediction research

### 8.2 Societal Impact

1. **Smart Cities**: Enable real-time traffic digital twins for European cities
2. **Sustainability**: Reduce congestion-related emissions through optimized traffic flow
3. **Infrastructure Planning**: Data-driven investment decisions

### 8.3 Economic Impact

1. **Reduced Congestion Costs**: Potential savings of €100M+ annually per major city
2. **Fuel Savings**: 5-10% reduction through optimized routing
3. **HPC Industry**: Demonstrate value of EuroHPC for smart city applications

### 8.4 Dissemination Plan

| Activity | Timeline | Target Audience |
|----------|----------|-----------------|
| Technical report | M6 | EuroHPC community |
| Conference paper | M8 | SC, ISC, Euro-Par |
| Journal paper | M12 | IEEE TPDS, JPDC |
| Open-source release | M6 | Research community |
| Blog post | M6 | General public |

---

## 9. Team and Expertise

### 9.1 Core Team

| Name | Role | Expertise | Commitment |
|------|------|-----------|------------|
| [Team Lead] | PI | HPC, Distributed Systems | 20% |
| [Member 2] | Co-I | Deep Learning, GNNs | 30% |
| [Member 3] | PhD Student | Traffic Modeling | 50% |
| [Member 4] | Developer | PyTorch, CUDA/HIP | 30% |

### 9.2 Relevant Experience

- Previous allocations on national HPC systems
- Publications in HPC and ML venues
- Experience with NVIDIA and AMD GPU platforms

---

## 10. References

1. Li, Y., et al. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR.

2. EuroHPC JU. (2023). LUMI System Documentation.

3. NVIDIA. (2023). NCCL Documentation.

4. AMD. (2023). ROCm Documentation.

5. Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. Scientific Data.

---

## Appendix A: Code Availability

```
GitHub: https://github.com/[team]/hpc-traffic-prediction
Documentation: Complete README, reproduce.md, SYSTEM.md
Container: Apptainer definition included
Tests: Unit tests and integration tests
CI/CD: GitHub Actions for automated testing
```

## Appendix B: Letters of Support

[Space for institutional letters of support]

---

*Submitted for EuroHPC Development Access Program*
*Date: [Submission Date]*
