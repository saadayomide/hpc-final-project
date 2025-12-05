#!/usr/bin/env python3
"""
Scaling Analysis Script

Analyzes strong and weak scaling experiment results and generates plots.

Usage:
    python scripts/analyze_scaling.py --results results/strong_scaling_* --type strong
    python scripts/analyze_scaling.py --results results/weak_scaling_* --type weak
"""

import argparse
import os
import sys
import glob
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Try to import matplotlib, provide helpful error if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will not be generated")


def parse_metrics_csv(csv_path: str) -> Dict:
    """Parse a metrics.csv file and return summary statistics."""
    if not os.path.exists(csv_path):
        return None
    
    epochs = []
    epoch_times = []
    throughputs = []
    train_losses = []
    val_losses = []
    val_maes = []
    gpu_utils = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get('epoch', 0)))
                epoch_times.append(float(row.get('epoch_time_sec', 0)))
                throughputs.append(float(row.get('throughput_samples_per_sec', 0)))
                train_losses.append(float(row.get('train_loss', 0)))
                val_losses.append(float(row.get('val_loss', 0)))
                val_maes.append(float(row.get('val_mae', 0)))
                gpu_utils.append(float(row.get('gpu_util_avg', 0)))
            except (ValueError, KeyError) as e:
                continue
    
    if not epoch_times:
        return None
    
    # Skip first epoch (warmup) for timing statistics
    if len(epoch_times) > 1:
        steady_times = epoch_times[1:]
        steady_throughputs = throughputs[1:]
    else:
        steady_times = epoch_times
        steady_throughputs = throughputs
    
    return {
        'total_epochs': len(epochs),
        'avg_epoch_time': np.mean(steady_times),
        'std_epoch_time': np.std(steady_times),
        'avg_throughput': np.mean(steady_throughputs),
        'std_throughput': np.std(steady_throughputs),
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_val_loss': val_losses[-1] if val_losses else 0,
        'final_val_mae': val_maes[-1] if val_maes else 0,
        'avg_gpu_util': np.mean(gpu_utils) if gpu_utils else 0,
        'epoch_times': epoch_times,
        'throughputs': throughputs,
    }


def parse_metadata(metadata_path: str) -> Dict:
    """Parse a metadata.txt file."""
    if not os.path.exists(metadata_path):
        return {}
    
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()
    return metadata


def collect_scaling_results(results_dir: str) -> List[Dict]:
    """Collect results from all node configurations in a scaling experiment."""
    results = []
    
    # Handle both glob patterns and directory paths
    if '*' in results_dir:
        base_dirs = glob.glob(results_dir)
    else:
        base_dirs = [results_dir]
    
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
        
        # Look for subdirectories like "1n", "2n", "4n"
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            # Try to extract node count from directory name
            if subdir.endswith('n'):
                try:
                    num_nodes = int(subdir[:-1])
                except ValueError:
                    continue
            else:
                continue
            
            # Parse metrics
            metrics_path = os.path.join(subdir_path, 'metrics.csv')
            metadata_path = os.path.join(subdir_path, 'metadata.txt')
            
            metrics = parse_metrics_csv(metrics_path)
            metadata = parse_metadata(metadata_path)
            
            if metrics:
                results.append({
                    'num_nodes': num_nodes,
                    'metrics': metrics,
                    'metadata': metadata,
                    'path': subdir_path,
                })
    
    # Sort by node count
    results.sort(key=lambda x: x['num_nodes'])
    return results


def analyze_strong_scaling(results: List[Dict], output_dir: str) -> Dict:
    """Analyze strong scaling results."""
    if not results:
        print("No results to analyze")
        return {}
    
    # Get baseline (1 node) results
    baseline = next((r for r in results if r['num_nodes'] == 1), None)
    if not baseline:
        print("Warning: No baseline (1 node) results found, using first result")
        baseline = results[0]
    
    baseline_time = baseline['metrics']['avg_epoch_time']
    
    analysis = {
        'experiment_type': 'strong_scaling',
        'baseline_time': baseline_time,
        'results': []
    }
    
    for r in results:
        nodes = r['num_nodes']
        epoch_time = r['metrics']['avg_epoch_time']
        throughput = r['metrics']['avg_throughput']
        
        speedup = baseline_time / epoch_time if epoch_time > 0 else 0
        efficiency = speedup / nodes if nodes > 0 else 0
        ideal_speedup = nodes
        
        entry = {
            'num_nodes': nodes,
            'epoch_time_sec': epoch_time,
            'throughput_samples_per_sec': throughput,
            'speedup': speedup,
            'efficiency': efficiency,
            'ideal_speedup': ideal_speedup,
            'train_loss': r['metrics']['final_train_loss'],
            'val_loss': r['metrics']['final_val_loss'],
            'val_mae': r['metrics']['final_val_mae'],
            'gpu_util_avg': r['metrics']['avg_gpu_util'],
        }
        analysis['results'].append(entry)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'strong_scaling_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        if analysis['results']:
            writer = csv.DictWriter(f, fieldnames=analysis['results'][0].keys())
            writer.writeheader()
            writer.writerows(analysis['results'])
    print(f"Saved: {csv_path}")
    
    return analysis


def analyze_weak_scaling(results: List[Dict], output_dir: str) -> Dict:
    """Analyze weak scaling results."""
    if not results:
        print("No results to analyze")
        return {}
    
    baseline = next((r for r in results if r['num_nodes'] == 1), None)
    if not baseline:
        baseline = results[0]
    
    baseline_time = baseline['metrics']['avg_epoch_time']
    baseline_throughput = baseline['metrics']['avg_throughput']
    
    analysis = {
        'experiment_type': 'weak_scaling',
        'baseline_time': baseline_time,
        'baseline_throughput': baseline_throughput,
        'results': []
    }
    
    for r in results:
        nodes = r['num_nodes']
        epoch_time = r['metrics']['avg_epoch_time']
        throughput = r['metrics']['avg_throughput']
        
        # For weak scaling: time should stay constant, throughput should scale linearly
        time_ratio = epoch_time / baseline_time if baseline_time > 0 else 0
        efficiency = (baseline_throughput * nodes) / throughput if throughput > 0 else 0
        # Alternative efficiency: 1 / time_ratio (time should be constant)
        time_efficiency = baseline_time / epoch_time if epoch_time > 0 else 0
        
        entry = {
            'num_nodes': nodes,
            'epoch_time_sec': epoch_time,
            'throughput_samples_per_sec': throughput,
            'time_ratio': time_ratio,
            'efficiency': min(time_efficiency, 1.0),  # Cap at 100%
            'samples_per_node': throughput / nodes if nodes > 0 else 0,
            'total_samples': throughput,  # Approximate
            'train_loss': r['metrics']['final_train_loss'],
            'val_loss': r['metrics']['final_val_loss'],
            'gpu_util_avg': r['metrics']['avg_gpu_util'],
        }
        analysis['results'].append(entry)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'weak_scaling_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        if analysis['results']:
            writer = csv.DictWriter(f, fieldnames=analysis['results'][0].keys())
            writer.writeheader()
            writer.writerows(analysis['results'])
    print(f"Saved: {csv_path}")
    
    return analysis


def plot_strong_scaling(analysis: Dict, output_dir: str):
    """Generate strong scaling plots."""
    if not HAS_MATPLOTLIB:
        return
    
    results = analysis.get('results', [])
    if not results:
        return
    
    nodes = [r['num_nodes'] for r in results]
    speedups = [r['speedup'] for r in results]
    efficiencies = [r['efficiency'] * 100 for r in results]  # Convert to percentage
    ideal_speedups = [r['ideal_speedup'] for r in results]
    throughputs = [r['throughput_samples_per_sec'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Strong Scaling Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Speedup
    ax1 = axes[0, 0]
    ax1.plot(nodes, speedups, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.plot(nodes, ideal_speedups, 'r--', linewidth=2, label='Ideal (linear)')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Nodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(nodes)
    
    # Plot 2: Efficiency
    ax2 = axes[0, 1]
    ax2.bar(nodes, efficiencies, color='steelblue', edgecolor='black')
    ax2.axhline(y=80, color='orange', linestyle='--', label='80% target')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Parallel Efficiency (%)')
    ax2.set_title('Parallel Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(nodes)
    ax2.set_ylim(0, 110)
    
    # Plot 3: Throughput
    ax3 = axes[1, 0]
    ax3.plot(nodes, throughputs, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Throughput (samples/sec)')
    ax3.set_title('Training Throughput')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(nodes)
    
    # Plot 4: Time breakdown
    ax4 = axes[1, 1]
    times = [r['epoch_time_sec'] for r in results]
    ax4.bar(nodes, times, color='coral', edgecolor='black')
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Epoch Time (seconds)')
    ax4.set_title('Training Time per Epoch')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(nodes)
    
    plt.tight_layout()
    
    # Save plots
    for fmt in ['png', 'svg']:
        path = os.path.join(output_dir, f'scaling_analysis.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.close()


def plot_weak_scaling(analysis: Dict, output_dir: str):
    """Generate weak scaling plots."""
    if not HAS_MATPLOTLIB:
        return
    
    results = analysis.get('results', [])
    if not results:
        return
    
    nodes = [r['num_nodes'] for r in results]
    times = [r['epoch_time_sec'] for r in results]
    throughputs = [r['throughput_samples_per_sec'] for r in results]
    efficiencies = [r['efficiency'] * 100 for r in results]
    
    # Ideal throughput (linear scaling)
    baseline_throughput = results[0]['throughput_samples_per_sec']
    ideal_throughputs = [baseline_throughput * n for n in nodes]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Weak Scaling Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Time (should stay constant for perfect weak scaling)
    ax1 = axes[0, 0]
    ax1.plot(nodes, times, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.axhline(y=times[0], color='r', linestyle='--', label='Ideal (constant)')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Epoch Time (seconds)')
    ax1.set_title('Time vs Nodes (should be constant)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(nodes)
    
    # Plot 2: Throughput
    ax2 = axes[0, 1]
    ax2.plot(nodes, throughputs, 'go-', linewidth=2, markersize=8, label='Measured')
    ax2.plot(nodes, ideal_throughputs, 'r--', linewidth=2, label='Ideal (linear)')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Throughput Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(nodes)
    
    # Plot 3: Efficiency
    ax3 = axes[1, 0]
    ax3.bar(nodes, efficiencies, color='steelblue', edgecolor='black')
    ax3.axhline(y=80, color='orange', linestyle='--', label='80% target')
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Weak Scaling Efficiency (%)')
    ax3.set_title('Parallel Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(nodes)
    ax3.set_ylim(0, 110)
    
    # Plot 4: Throughput per node
    ax4 = axes[1, 1]
    per_node = [r['samples_per_node'] for r in results]
    ax4.bar(nodes, per_node, color='coral', edgecolor='black')
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Throughput per Node (samples/sec)')
    ax4.set_title('Per-Node Throughput')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(nodes)
    
    plt.tight_layout()
    
    for fmt in ['png', 'svg']:
        path = os.path.join(output_dir, f'scaling_analysis.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.close()


def print_summary(analysis: Dict, exp_type: str):
    """Print a summary of the analysis."""
    print("\n" + "=" * 60)
    print(f"{exp_type.upper()} SCALING SUMMARY")
    print("=" * 60)
    
    results = analysis.get('results', [])
    if not results:
        print("No results available")
        return
    
    if exp_type == 'strong':
        print(f"{'Nodes':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Throughput':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['num_nodes']:<8} {r['epoch_time_sec']:<12.2f} {r['speedup']:<10.2f} "
                  f"{r['efficiency']*100:<11.1f}% {r['throughput_samples_per_sec']:<15.1f}")
    else:
        print(f"{'Nodes':<8} {'Time (s)':<12} {'Throughput':<15} {'Efficiency':<12}")
        print("-" * 60)
        for r in results:
            print(f"{r['num_nodes']:<8} {r['epoch_time_sec']:<12.2f} "
                  f"{r['throughput_samples_per_sec']:<15.1f} {r['efficiency']*100:<11.1f}%")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze scaling experiment results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results directory (supports glob patterns)')
    parser.add_argument('--type', type=str, choices=['strong', 'weak'], required=True,
                        help='Type of scaling experiment')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis (default: results/scaling)')
    
    args = parser.parse_args()
    
    # Collect results
    print(f"Collecting results from: {args.results}")
    results = collect_scaling_results(args.results)
    
    if not results:
        print("No valid results found!")
        print("Make sure the results directory contains subdirectories like '1n', '2n', '4n' with metrics.csv files")
        sys.exit(1)
    
    print(f"Found {len(results)} configurations: {[r['num_nodes'] for r in results]} nodes")
    
    # Set output directory
    output_dir = args.output or 'results/scaling'
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze based on type
    if args.type == 'strong':
        analysis = analyze_strong_scaling(results, output_dir)
        plot_strong_scaling(analysis, output_dir)
    else:
        analysis = analyze_weak_scaling(results, output_dir)
        plot_weak_scaling(analysis, output_dir)
    
    # Print summary
    print_summary(analysis, args.type)
    
    # Save full analysis as JSON
    json_path = os.path.join(output_dir, f'{args.type}_scaling_full.json')
    with open(json_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(analysis, f, indent=2, default=convert)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
