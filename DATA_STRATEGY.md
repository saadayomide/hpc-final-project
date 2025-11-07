# Data Strategy: Synthetic vs Real Data

## Current Status

**Current implementation:** Uses **synthetic (generated) data** for testing

**Why synthetic data now:**
- ✅ Allows testing the system without needing real data first
- ✅ Focuses Phase 1 on getting the HPC infrastructure working
- ✅ Can be easily replaced with real data later
- ✅ Demonstrates the system is functional

---

## What You Need to Decide

### Option 1: Keep Synthetic Data (Recommended for Phase 1)

**Pros:**
- ✅ **Phase 1 complete** - Focus is on HPC infrastructure, not data
- ✅ **No data acquisition needed** - Can test immediately
- ✅ **Reproducible** - Same data every time
- ✅ **Fast iteration** - Test system changes quickly

**Cons:**
- ❌ Not realistic - Results won't reflect real-world performance
- ❌ Can't validate model accuracy on real tasks
- ❌ May need real data later for final project/demo

**When to use:**
- Phase 1 (current): Getting infrastructure working
- Testing: System changes, scaling tests
- Development: Rapid iteration

---

### Option 2: Use Real Data (For Full Project)

**Pros:**
- ✅ **Realistic results** - Actual performance metrics
- ✅ **Publishable** - Can write papers with real results
- ✅ **Validates model** - Shows if DCRNN works for your use case
- ✅ **Complete project** - Demonstrates full pipeline

**Cons:**
- ❌ **Data acquisition** - Need to find/download/prepare data
- ❌ **Preprocessing** - May need significant data preparation
- ❌ **Storage** - Real datasets can be large
- ❌ **Time** - More setup time before testing

**When to use:**
- Final project submission
- Research paper/publication
- Real-world application
- Performance validation

---

## Recommendation for Your Project

### Phase 1 (Current): **Keep Synthetic Data** ✅

**Reason:** Phase 1 is about demonstrating:
- ✅ System works end-to-end
- ✅ Training runs on HPC cluster
- ✅ Metrics/logging work correctly
- ✅ Multi-GPU training functions
- ✅ Reproducibility works

**What you demonstrate:**
- Infrastructure works
- Code is functional
- System is ready for real data

### Future Phases: **Consider Real Data** (if needed)

**Reason:** For scaling tests (Phase 2+) and final project:
- Real data may be needed for meaningful results
- Depends on your project requirements
- Depends on what you're trying to demonstrate

---

## How to Switch to Real Data (When Ready)

The system is designed to be flexible. Here's how to add real data:

### Step 1: Get Your Data

**Common DCRNN datasets:**
- **Traffic forecasting:** METR-LA, PEMS-BAY
- **Epidemiology:** COVID-19 data, disease spread networks
- **Sensor networks:** IoT sensor data
- **Weather:** Climate prediction datasets

**Where to find:**
- Papers with code: https://paperswithcode.com/
- UCI Machine Learning Repository
- Kaggle datasets
- Domain-specific repositories

### Step 2: Update `fetch_data.sh`

```bash
# Example: Download METR-LA traffic dataset
wget -O data/metr_la.tar.gz "https://example.com/metr_la.tar.gz"
tar -xzf data/metr_la.tar.gz -C data/
```

### Step 3: Update `src/data.py`

**Option A: Modify `DCRNNDataset` class**
```python
def __init__(self, data_dir, split='train', ...):
    # Check if real data exists
    if os.path.exists(os.path.join(data_dir, f'{split}_data.npz')):
        # Load real data
        data = np.load(os.path.join(data_dir, f'{split}_data.npz'))
        self.x = data['x']  # [num_samples, seq_len, num_nodes, input_dim]
        self.y = data['y']  # [num_samples, pred_len, num_nodes, output_dim]
        self.adj_matrix = data['adj_matrix']
    else:
        # Fall back to synthetic
        self._generate_synthetic_data()
```

**Option B: Create new class**
```python
class RealDCRNNDataset(Dataset):
    """Loads real data from files"""
    def __init__(self, data_dir, split='train'):
        # Load your actual data files
        ...
```

### Step 4: Update Training Script

```python
# In src/train.py, add flag to choose data source
parser.add_argument('--use-real-data', action='store_true',
                    help='Use real data instead of synthetic')
```

---

## What Your Project Needs

### Check Your Project Requirements:

1. **Is this for a course/assignment?**
   - Check if real data is required
   - Some courses want synthetic data for testing
   - Some want real data for final results

2. **Is this for research/publication?**
   - Real data usually needed
   - Synthetic data acceptable for infrastructure testing
   - Real data needed for final results

3. **What are you trying to demonstrate?**
   - **HPC scaling:** Synthetic data is fine
   - **Model performance:** Real data needed
   - **Infrastructure:** Synthetic data is fine

4. **What's your timeline?**
   - **Short timeline:** Stick with synthetic for now
   - **Long timeline:** Can add real data later

---

## My Recommendation

### For Phase 1: **Stick with Synthetic Data** ✅

**Why:**
1. Phase 1 goal is "functional prototype" - synthetic data is fine
2. Focus should be on getting HPC infrastructure working
3. You can validate the system works without real data
4. Easy to add real data later when needed

### For Final Project: **Depends on Requirements**

**If you need real results:**
- Add real data in Phase 2 or 3
- Update `src/data.py` to load real data
- Run full training with real dataset

**If infrastructure is enough:**
- Synthetic data is fine
- Focus on scaling, performance, reproducibility
- Document that synthetic data was used

---

## Summary

**Current state:** ✅ Synthetic data - works for Phase 1

**Future state:** 🤔 Real data - depends on your needs

**Easy to switch:** ✅ System is designed to handle both

**Recommendation:** Keep synthetic for Phase 1, add real data later if needed for final project/paper

---

## Questions to Ask Yourself

1. **What does my project require?**
   - Check assignment/course requirements
   - Check if real data is specified

2. **What am I trying to demonstrate?**
   - HPC infrastructure → Synthetic is fine
   - Model performance → Real data needed

3. **What's my timeline?**
   - Short → Stick with synthetic
   - Long → Can add real data later

4. **What's the focus?**
   - Scaling/performance → Synthetic is fine
   - Accuracy/results → Real data needed

