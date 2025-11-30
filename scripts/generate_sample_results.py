#!/usr/bin/env python3
"""
Generate Sample Results and Plots

Creates realistic sample results for demonstration and documentation purposes.
This simulates the expected output from scaling experiments and profiling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


def generate_strong_scaling_data():
    """Generate sample strong scaling data"""
    nodes = np.array([1, 2, 4, 8])
    
    # Realistic scaling with some overhead
    base_time = 120.0  # seconds for 1 node
    efficiency_decay = 0.92  # efficiency drops slightly with more nodes
    
    times = []
    for n in nodes:
        # Ideal would be base_time/n, but there's communication overhead
        ideal = base_time / n
        efficiency = efficiency_decay ** (n - 1)
        actual = ideal / efficiency
        times.append(actual)
    
    times = np.array(times)
    throughput = 5000 / times  # samples per second
    speedup = times[0] / times
    efficiency = speedup / nodes
    
    return pd.DataFrame({
        'num_nodes': nodes,
        'epoch_time_sec': times,
        'throughput_samples_per_sec': throughput,
        'speedup': speedup,
        'efficiency': efficiency,
        'ideal_speedup': nodes.astype(float),
        'train_loss': 0.1 - 0.01 * np.log(nodes),  # slightly better with more data
        'val_loss': 0.12 - 0.01 * np.log(nodes),
        'val_mae': 3.5 - 0.2 * np.log(nodes),
        'val_rmse': 5.2 - 0.3 * np.log(nodes),
        'gpu_util_avg': 85 - 3 * np.log2(nodes),
        'data_load_time_sec': 5 + 2 * np.log(nodes),
        'compute_time_sec': times - 5 - 2 * np.log(nodes) - 5
    })


def generate_weak_scaling_data():
    """Generate sample weak scaling data"""
    nodes = np.array([1, 2, 4, 8])
    
    # For weak scaling, time should stay roughly constant
    base_time = 120.0
    comm_overhead = 0.05  # 5% overhead per doubling
    
    times = base_time * (1 + comm_overhead * np.log2(nodes))
    throughput = 5000 * nodes / times  # total throughput scales with nodes
    efficiency = base_time / times
    
    return pd.DataFrame({
        'num_nodes': nodes,
        'epoch_time_sec': times,
        'throughput_samples_per_sec': throughput,
        'time_ratio': times / times[0],
        'efficiency': efficiency,
        'samples_per_node': np.full_like(nodes, 5000, dtype=float),
        'total_samples': 5000 * nodes,
        'train_loss': np.array([0.102, 0.100, 0.098, 0.097]),
        'val_loss': np.array([0.122, 0.119, 0.117, 0.116]),
        'gpu_util_avg': np.array([87, 85, 83, 81]),
    })


def generate_sensitivity_data():
    """Generate sample sensitivity sweep data"""
    batch_sizes = [32, 64, 128, 256]
    num_workers = [2, 4, 8]
    
    results = []
    for bs in batch_sizes:
        for nw in num_workers:
            # Throughput increases with batch size up to a point
            base_throughput = 100
            bs_factor = np.sqrt(bs / 32)
            nw_factor = 1 + 0.1 * np.log2(nw)
            throughput = base_throughput * bs_factor * nw_factor
            
            # Time decreases with throughput
            time = 5000 / throughput
            
            # GPU util increases with batch size
            gpu_util = min(95, 60 + 10 * np.log2(bs / 32))
            
            results.append({
                'batch_size': bs,
                'num_workers': nw,
                'epoch_time_sec': time,
                'throughput_samples_per_sec': throughput,
                'gpu_util_avg': gpu_util,
                'memory_used_gb': 8 + 6 * (bs / 256),
                'data_load_time_sec': time * (0.3 - 0.05 * np.log2(nw)),
            })
    
    return pd.DataFrame(results)


def generate_profiling_data():
    """Generate sample GPU and CPU profiling data"""
    # Simulate 10 minutes of profiling at 1-second intervals
    n_samples = 600
    timestamps = np.arange(n_samples)
    
    # GPU data with realistic patterns
    gpu_util_base = 75 + 10 * np.sin(timestamps / 50)  # periodic variation
    gpu_util = gpu_util_base + np.random.randn(n_samples) * 5
    gpu_util = np.clip(gpu_util, 0, 100)
    
    memory_util = 65 + np.random.randn(n_samples) * 3
    memory_used = 12000 + np.random.randn(n_samples) * 500
    temperature = 65 + 0.01 * timestamps + np.random.randn(n_samples) * 2
    power = 250 + 20 * (gpu_util / 100) + np.random.randn(n_samples) * 10
    
    gpu_df = pd.DataFrame({
        'timestamp': timestamps,
        'gpu_id': 0,
        'utilization_gpu': gpu_util,
        'utilization_memory': memory_util,
        'memory_used_mb': memory_used,
        'memory_total_mb': 16384,
        'temperature': temperature,
        'power_draw_w': power
    })
    
    # CPU data
    cpu_percent = 45 + 15 * np.sin(timestamps / 30) + np.random.randn(n_samples) * 5
    cpu_percent = np.clip(cpu_percent, 0, 100)
    memory_percent = 35 + np.random.randn(n_samples) * 2
    
    cpu_df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_used_gb': 32 * memory_percent / 100,
        'memory_total_gb': 32
    })
    
    return gpu_df, cpu_df


def generate_training_metrics():
    """Generate sample training metrics over epochs"""
    epochs = np.arange(1, 51)
    
    # Realistic loss curves with diminishing returns
    train_loss = 0.5 * np.exp(-epochs / 15) + 0.05 + np.random.randn(50) * 0.01
    val_loss = 0.55 * np.exp(-epochs / 15) + 0.06 + np.random.randn(50) * 0.01
    
    # MAE and RMSE follow similar patterns
    val_mae = 8 * np.exp(-epochs / 20) + 3.5 + np.random.randn(50) * 0.1
    val_rmse = 12 * np.exp(-epochs / 20) + 5.0 + np.random.randn(50) * 0.2
    
    # Throughput stabilizes after warm-up
    throughput = 380 + 20 * (1 - np.exp(-epochs / 5)) + np.random.randn(50) * 10
    
    epoch_time = 5000 / throughput
    data_load_time = epoch_time * 0.15 + np.random.randn(50) * 0.2
    compute_time = epoch_time * 0.75 + np.random.randn(50) * 0.3
    
    gpu_util = 82 + 3 * (1 - np.exp(-epochs / 5)) + np.random.randn(50) * 2
    cpu_util = 45 + np.random.randn(50) * 5
    
    return pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'epoch_time_sec': epoch_time,
        'throughput_samples_per_sec': throughput,
        'data_load_time_sec': data_load_time,
        'compute_time_sec': compute_time,
        'gpu_util_avg': gpu_util,
        'cpu_util_avg': cpu_util
    })


def create_scaling_plots(strong_df, weak_df, output_dir):
    """Create scaling analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Strong scaling - Time
    ax = axes[0, 0]
    ax.plot(strong_df['num_nodes'], strong_df['epoch_time_sec'], 'o-', 
            linewidth=2, markersize=8, label='Actual')
    ideal_time = strong_df['epoch_time_sec'].iloc[0] / strong_df['num_nodes']
    ax.plot(strong_df['num_nodes'], ideal_time, '--', linewidth=2, alpha=0.7, label='Ideal')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Time per Epoch (s)', fontsize=12)
    ax.set_title('Strong Scaling: Time vs Nodes', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(strong_df['num_nodes'])
    
    # Strong scaling - Speedup
    ax = axes[0, 1]
    ax.plot(strong_df['num_nodes'], strong_df['speedup'], 'o-', 
            linewidth=2, markersize=8, label='Actual')
    ax.plot(strong_df['num_nodes'], strong_df['ideal_speedup'], '--', 
            linewidth=2, alpha=0.7, label='Ideal')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Strong Scaling: Speedup', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(strong_df['num_nodes'])
    
    # Strong scaling - Efficiency
    ax = axes[0, 2]
    ax.plot(strong_df['num_nodes'], strong_df['efficiency'] * 100, 'o-', 
            linewidth=2, markersize=8, color='green')
    ax.axhline(y=80, linestyle='--', color='red', alpha=0.7, label='80% threshold')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('Strong Scaling: Efficiency', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 110])
    ax.set_xticks(strong_df['num_nodes'])
    
    # Weak scaling - Time
    ax = axes[1, 0]
    ax.plot(weak_df['num_nodes'], weak_df['epoch_time_sec'], 'o-', 
            linewidth=2, markersize=8, label='Actual')
    ax.axhline(y=weak_df['epoch_time_sec'].iloc[0], linestyle='--', 
               color='red', alpha=0.7, label='Ideal (constant)')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Time per Epoch (s)', fontsize=12)
    ax.set_title('Weak Scaling: Time vs Nodes', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weak_df['num_nodes'])
    
    # Weak scaling - Throughput
    ax = axes[1, 1]
    ax.plot(weak_df['num_nodes'], weak_df['throughput_samples_per_sec'], 'o-', 
            linewidth=2, markersize=8, label='Actual')
    ideal_throughput = weak_df['throughput_samples_per_sec'].iloc[0] * weak_df['num_nodes']
    ax.plot(weak_df['num_nodes'], ideal_throughput, '--', linewidth=2, alpha=0.7, label='Ideal')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title('Weak Scaling: Total Throughput', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weak_df['num_nodes'])
    
    # Weak scaling - Efficiency
    ax = axes[1, 2]
    ax.plot(weak_df['num_nodes'], weak_df['efficiency'] * 100, 'o-', 
            linewidth=2, markersize=8, color='green')
    ax.axhline(y=80, linestyle='--', color='red', alpha=0.7, label='80% threshold')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('Weak Scaling: Efficiency', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 110])
    ax.set_xticks(weak_df['num_nodes'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scaling_analysis.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved scaling plots to {output_dir}")


def create_sensitivity_plots(sens_df, output_dir):
    """Create sensitivity analysis plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Throughput vs Batch Size
    ax = axes[0]
    for nw in sens_df['num_workers'].unique():
        subset = sens_df[sens_df['num_workers'] == nw].sort_values('batch_size')
        ax.plot(subset['batch_size'], subset['throughput_samples_per_sec'], 
                'o-', linewidth=2, markersize=8, label=f'{nw} workers')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title('Throughput vs Batch Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # GPU Utilization vs Batch Size
    ax = axes[1]
    for nw in sens_df['num_workers'].unique():
        subset = sens_df[sens_df['num_workers'] == nw].sort_values('batch_size')
        ax.plot(subset['batch_size'], subset['gpu_util_avg'], 
                'o-', linewidth=2, markersize=8, label=f'{nw} workers')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization vs Batch Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heatmap
    ax = axes[2]
    pivot = sens_df.pivot(index='num_workers', columns='batch_size', 
                          values='throughput_samples_per_sec')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Number of Workers', fontsize=12)
    ax.set_title('Throughput Heatmap', fontsize=14)
    plt.colorbar(im, ax=ax, label='Throughput (samples/s)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sensitivity_analysis.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved sensitivity plots to {output_dir}")


def create_training_plots(metrics_df, output_dir):
    """Create training progress plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(metrics_df['epoch'], metrics_df['train_loss'], 'o-', 
            linewidth=2, markersize=4, label='Train Loss', alpha=0.8)
    ax.plot(metrics_df['epoch'], metrics_df['val_loss'], 's-', 
            linewidth=2, markersize=4, label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE and RMSE
    ax = axes[0, 1]
    ax.plot(metrics_df['epoch'], metrics_df['val_mae'], 'o-', 
            linewidth=2, markersize=4, label='MAE', alpha=0.8)
    ax.plot(metrics_df['epoch'], metrics_df['val_rmse'], 's-', 
            linewidth=2, markersize=4, label='RMSE', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error (mph)', fontsize=12)
    ax.set_title('Validation MAE and RMSE', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Throughput
    ax = axes[1, 0]
    ax.plot(metrics_df['epoch'], metrics_df['throughput_samples_per_sec'], 
            linewidth=2, alpha=0.8)
    ax.fill_between(metrics_df['epoch'], metrics_df['throughput_samples_per_sec'], alpha=0.3)
    ax.axhline(y=metrics_df['throughput_samples_per_sec'].mean(), 
               linestyle='--', color='red', label=f"Avg: {metrics_df['throughput_samples_per_sec'].mean():.1f}")
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title('Training Throughput', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # GPU Utilization
    ax = axes[1, 1]
    ax.plot(metrics_df['epoch'], metrics_df['gpu_util_avg'], 
            linewidth=2, alpha=0.8, color='green')
    ax.axhline(y=80, linestyle='--', color='red', alpha=0.7, label='80% target')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('Average GPU Utilization', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_progress.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved training plots to {output_dir}")


def main():
    # Set up output directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    print("Generating Sample Results")
    print("=" * 50)
    
    # Generate data
    print("\n1. Generating scaling data...")
    strong_df = generate_strong_scaling_data()
    weak_df = generate_weak_scaling_data()
    sens_df = generate_sensitivity_data()
    
    # Save CSVs
    (results_dir / 'scaling').mkdir(exist_ok=True)
    strong_df.to_csv(results_dir / 'scaling' / 'strong_scaling_analysis.csv', index=False)
    weak_df.to_csv(results_dir / 'scaling' / 'weak_scaling_analysis.csv', index=False)
    sens_df.to_csv(results_dir / 'scaling' / 'sensitivity_analysis.csv', index=False)
    print(f"   Saved to {results_dir / 'scaling'}")
    
    print("\n2. Generating profiling data...")
    gpu_df, cpu_df = generate_profiling_data()
    (results_dir / 'profiling').mkdir(exist_ok=True)
    gpu_df.to_csv(results_dir / 'profiling' / 'gpu_monitor.csv', index=False)
    cpu_df.to_csv(results_dir / 'profiling' / 'cpu_monitor.csv', index=False)
    print(f"   Saved to {results_dir / 'profiling'}")
    
    print("\n3. Generating training metrics...")
    metrics_df = generate_training_metrics()
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
    metrics_df.to_csv(results_dir / 'profiling' / 'metrics.csv', index=False)
    print(f"   Saved to {results_dir}")
    
    print("\n4. Creating plots...")
    create_scaling_plots(strong_df, weak_df, results_dir / 'scaling')
    create_sensitivity_plots(sens_df, results_dir / 'scaling')
    create_training_plots(metrics_df, results_dir)
    
    print("\n" + "=" * 50)
    print("Sample results generated successfully!")
    print(f"Results directory: {results_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
