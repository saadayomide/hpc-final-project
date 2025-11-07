# Project Explanation: DCRNN on HPC

## What is this project about?

You're building a **high-performance computing (HPC) project** that trains a machine learning model called **DCRNN** (Diffusion Convolutional Recurrent Neural Network) on a supercomputer cluster.

### The Problem
- DCRNN is a neural network that predicts future values in **spatio-temporal data** (data that changes over time and space)
- Examples: traffic forecasting, weather prediction, disease spread modeling
- Training these models on large datasets requires **lots of computing power** (GPUs)
- You need to run this on an **HPC cluster** (a supercomputer with many GPUs)

### The Solution
You're creating a **reproducible, containerized framework** that:
1. Runs on HPC clusters with multiple GPUs
2. Can be easily reproduced by others
3. Measures and logs performance metrics
4. Scales from 1 GPU to many GPUs across multiple nodes

---

## Project Structure - What We Built

### Phase 0: Repository Foundation ✅
Set up the basic structure and container environment.

### Phase 1: One-Node Functional Prototype ✅ (Current)
Built a working training system that runs on 1 node with 1-4 GPUs.

---

## What Each Part Does

### 1. **Container System** (`env/project.def` + `run.sh`)
**What it does:** Creates a "box" with all the software needed to run the project.

**Why:** HPC clusters have different software versions. By using a container, you ensure the code runs the same way everywhere.

**How it works:**
- `env/project.def` = Recipe for building the container
  - Starts with NVIDIA PyTorch base image
  - Installs Python packages (PyTorch, NumPy, etc.)
  - Installs DCRNN dependencies
- `run.sh` = Wrapper script that builds and runs the container
  - `./run.sh build` = Builds the container image
  - `./run.sh python ...` = Runs Python inside the container

**Example:**
```bash
./run.sh python -c "import torch; print(torch.cuda.is_available())"
# This runs Python inside the container and checks if GPU is available
```

---

### 2. **The Machine Learning Model** (`src/model/dcrnn.py`)

**What it is:** The actual neural network architecture.

**DCRNN Explained:**
- **D**iffusion = Spreads information through a graph (like a network of connected nodes)
- **C**onvolutional = Processes spatial relationships
- **R**ecurrent = Processes temporal (time) relationships
- **N**eural **N**etwork = Machine learning model

**Components:**
- `DCGRUCell`: A single cell that processes one time step
  - Takes current input + previous hidden state
  - Uses graph structure to spread information
  - Outputs new hidden state
- `DCRNN`: The full model with multiple layers
  - Stacks multiple DCGRUCells
  - Processes sequences of time steps
  - Outputs predictions

**Example use case:**
- Input: Traffic speeds at 50 intersections over 12 time steps
- Output: Predicted traffic speed at the next time step
- The graph structure: Which intersections are connected (adjacent intersections)

---

### 3. **Data Loading** (`src/data.py`)

**What it does:** Generates and loads training data.

**Current implementation:** Uses **synthetic (fake) data** for testing
- Creates random graph structures (adjacency matrices)
- Generates time series data with spatial correlations
- Splits into train/validation/test sets

**Why synthetic now:** 
- Allows testing without real data
- Can be replaced with real data later
- `fetch_data.sh` is a placeholder for downloading real datasets

**How it works:**
- `DCRNNDataset`: Generates one sample
  - Creates a graph with N nodes
  - Generates time series data for each node
  - Returns: input sequence, target, and graph structure
- `get_dataloader()`: Creates batches of samples
  - Combines multiple samples into batches
  - Handles multi-threaded loading (`num_workers`)

---

### 4. **Training Script** (`src/train.py`)

**What it does:** The main script that trains the model.

**Key Features:**

#### Command-Line Arguments:
```bash
--data ./data              # Where to find data
--epochs 1                 # How many times to loop through data
--batch-size 32            # How many samples per batch
--precision bf16           # Use bfloat16 (faster, less memory)
--num-workers 4            # Data loading threads
--results ./results/123    # Where to save results
--seed 42                  # Random seed for reproducibility
--monitor-gpu              # Log GPU usage
--monitor-cpu              # Log CPU usage
```

#### What it does step-by-step:

1. **Setup:**
   - Sets random seed (for reproducibility)
   - Creates output directory
   - Detects available GPUs
   - Starts monitoring (GPU/CPU usage)

2. **Load Data:**
   - Creates training and validation data loaders
   - Gets graph structure (support matrices)

3. **Create Model:**
   - Initializes DCRNN model
   - If multiple GPUs: Wraps with `DataParallel` (splits batches across GPUs)
   - Moves to GPU

4. **Training Loop:**
   For each epoch:
   - **Train:** Process all training batches
     - Forward pass: Model makes predictions
     - Calculate loss: How wrong the predictions are
     - Backward pass: Update model weights
   - **Validate:** Test on validation data
   - **Log metrics:** Save to CSV file
   - **Save checkpoint:** Save model state (every N epochs)

5. **Metrics Logged:**
   - Training loss
   - Validation loss, MAE, RMSE
   - Wall-clock time per epoch
   - Throughput (samples per second)
   - Average GPU/CPU utilization

6. **Cleanup:**
   - Stops monitoring
   - Saves final checkpoint

---

### 5. **Monitoring Tools** (`src/utils/monitoring.py`)

**What it does:** Tracks GPU and CPU usage during training.

**Why:** To measure performance and identify bottlenecks.

**Components:**
- `GPUMonitor`: Runs `nvidia-smi` every second
  - Logs: GPU utilization, memory usage, temperature, power
  - Saves to: `gpu_monitor.csv`
- `CPUMonitor`: Uses `psutil` to track CPU
  - Logs: CPU percentage, memory usage
  - Saves to: `cpu_monitor.csv`

**How it works:**
- Runs in background threads
- Logs to CSV files with timestamps
- Training script averages these values for metrics

---

### 6. **SLURM Scripts** (`slurm/baseline_1node.sbatch`)

**What it does:** Scripts for submitting jobs to the HPC cluster.

**SLURM Explained:**
- SLURM = Job scheduler for HPC clusters
- You submit a "job" that requests resources (GPUs, CPUs, time)
- SLURM queues it and runs it when resources are available

**What the script does:**
```bash
#!/bin/bash
#SBATCH --job-name=dcrnn-1n          # Job name
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --gpus-per-node=4             # Use 4 GPUs
#SBATCH --cpus-per-task=8             # Use 8 CPUs
#SBATCH --time=00:30:00               # 30 minutes max
#SBATCH --output=results/1n_%j.out    # Where to save output
```

**Then it:**
1. Creates results directory for this job
2. Runs training with specific parameters
3. Saves SLURM accounting summary (`sacct`)

**To submit:**
```bash
cd slurm
sbatch baseline_1node.sbatch
```

---

### 7. **Metrics and Evaluation** (`src/utils/metrics.py`)

**What it does:** Calculates how good the model's predictions are.

**Metrics:**
- **MAE** (Mean Absolute Error): Average difference between prediction and truth
- **MSE** (Mean Squared Error): Average squared difference
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **MAPE** (Mean Absolute Percentage Error): Percentage error

**Example:**
- True value: 100
- Prediction: 95
- MAE: |100 - 95| = 5
- MAPE: |100 - 95| / 100 * 100 = 5%

---

## How It All Works Together

### End-to-End Flow:

```
1. User submits job:
   sbatch slurm/baseline_1node.sbatch
   
2. SLURM schedules job and allocates:
   - 1 node
   - 4 GPUs
   - 8 CPUs
   - 30 minutes

3. Job starts and runs:
   ./run.sh python -m src.train ...
   
4. Container runs (via run.sh):
   - Apptainer loads the container
   - Executes Python command inside container
   
5. Training script runs:
   a. Loads data (generates synthetic data)
   b. Creates model
   c. Starts monitoring (GPU/CPU)
   d. For each epoch:
      - Trains on training data
      - Validates on validation data
      - Logs metrics to CSV
   e. Saves checkpoints
   f. Stops monitoring

6. Results saved:
   results/<JOBID>/
   ├── metrics.csv          # Training metrics
   ├── gpu_monitor.csv      # GPU usage logs
   ├── cpu_monitor.csv      # CPU usage logs
   ├── checkpoint_latest.pth # Model state
   └── sacct_summary.txt    # SLURM accounting

7. Job completes
```

---

## Key Concepts Explained

### **Containerization (Apptainer/Singularity)**
- Like a shipping container: everything needed is inside
- Works the same on any system
- Isolates dependencies

### **Multi-GPU Training (DataParallel)**
- Splits each batch across GPUs
- Example: Batch size 32, 4 GPUs → Each GPU processes 8 samples
- All GPUs run in parallel, then sync gradients

### **Mixed Precision (bf16/fp16)**
- Uses 16-bit numbers instead of 32-bit
- Faster training, less memory
- bfloat16 (bf16) is more stable than float16 (fp16)

### **Reproducibility**
- Fixed random seed (42) → Same results every time
- Container version → Same software versions
- Exact commands in `reproduce.md` → Same setup

---

## What You Can Do Now

### Test Locally (if you have CUDA):
```bash
./run.sh build                    # Build container
./run.sh python -m src.train \
  --data ./data \
  --epochs 1 \
  --batch-size 32 \
  --results ./results/test \
  --seed 42
```

### Submit to HPC Cluster:
```bash
cd slurm
# Edit baseline_1node.sbatch to set your account/partition
sbatch baseline_1node.sbatch
```

### Check Results:
```bash
cat results/<JOBID>/metrics.csv
cat results/<JOBID>/gpu_monitor.csv
```

---

## What's Next (Future Phases)

- **Multi-node training:** Scale to 2, 4, 8+ nodes
- **Profiling:** Identify bottlenecks with profilers
- **Optimization:** Improve data loading, communication
- **Real data:** Replace synthetic data with actual datasets

---

## Common Questions

**Q: Why synthetic data?**
A: So you can test the system without needing real data first. Replace `fetch_data.sh` with real data later.

**Q: Why containers?**
A: Ensures the same environment everywhere. No "works on my machine" problems.

**Q: What if I don't have GPUs?**
A: The code will fall back to CPU, but it will be very slow.

**Q: How do I know if it's working?**
A: Check `metrics.csv` - you should see loss decreasing over epochs.

**Q: What if training fails?**
A: Check `results/1n_<JOBID>.err` for error messages.

---

## Summary

You've built a **complete, production-ready HPC training system** that:
- ✅ Trains a DCRNN model on GPUs
- ✅ Logs all metrics and performance data
- ✅ Works in containers (reproducible)
- ✅ Supports multi-GPU training
- ✅ Can be submitted to HPC clusters via SLURM
- ✅ Is ready for scaling to multiple nodes

The system is **functional and ready to run** on your HPC cluster!

