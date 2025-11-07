#!/usr/bin/env python3
"""
Scaling Analysis Script
Analyzes strong/weak scaling and sensitivity sweep results
"""

import argparse
import os
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(results_dir):
    """Load metrics from a results directory"""
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        return None
    
    df = pd.read_csv(metrics_file)
    # Get last epoch metrics
    if len(df) > 0:
        return df.iloc[-1].to_dict()
    return None


def analyze_strong_scaling(results_base):
    """Analyze strong scaling results"""
    print("Analyzing strong scaling results...")
    
    # Find all node directories
    node_dirs = sorted(glob.glob(os.path.join(results_base, '*nodes')))
    if not node_dirs:
        # Try alternative pattern
        node_dirs = sorted([d for d in glob.glob(os.path.join(results_base, '*')) if os.path.isdir(d)])
    
    results = []
    for node_dir in node_dirs:
        # Extract number of nodes
        dir_name = os.path.basename(node_dir)
        try:
            num_nodes = int(dir_name.replace('nodes', '').replace('node', ''))
        except:
            continue
        
        metrics = load_metrics(node_dir)
        if metrics:
            metrics['num_nodes'] = num_nodes
            results.append(metrics)
    
    if not results:
        print("No results found!")
        return None
    
    df = pd.DataFrame(results)
    df = df.sort_values('num_nodes')
    
    # Calculate scaling metrics
    baseline = df[df['num_nodes'] == df['num_nodes'].min()].iloc[0]
    df['speedup'] = baseline['epoch_time_sec'] / df['epoch_time_sec']
    df['efficiency'] = df['speedup'] / df['num_nodes']
    df['ideal_speedup'] = df['num_nodes']
    
    # Save results
    output_file = os.path.join(results_base, 'strong_scaling_analysis.csv')
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Create plots
    create_scaling_plots(df, results_base, 'strong')
    
    return df


def analyze_weak_scaling(results_base):
    """Analyze weak scaling results"""
    print("Analyzing weak scaling results...")
    
    # Find all node directories
    node_dirs = sorted(glob.glob(os.path.join(results_base, '*nodes')))
    if not node_dirs:
        node_dirs = sorted([d for d in glob.glob(os.path.join(results_base, '*')) if os.path.isdir(d)])
    
    results = []
    for node_dir in node_dirs:
        dir_name = os.path.basename(node_dir)
        try:
            num_nodes = int(dir_name.replace('nodes', '').replace('node', ''))
        except:
            continue
        
        metrics = load_metrics(node_dir)
        if metrics:
            metrics['num_nodes'] = num_nodes
            results.append(metrics)
    
    if not results:
        print("No results found!")
        return None
    
    df = pd.DataFrame(results)
    df = df.sort_values('num_nodes')
    
    # For weak scaling, time should stay roughly constant
    baseline = df[df['num_nodes'] == df['num_nodes'].min()].iloc[0]
    df['time_ratio'] = df['epoch_time_sec'] / baseline['epoch_time_sec']
    df['efficiency'] = baseline['epoch_time_sec'] / df['epoch_time_sec']
    
    # Save results
    output_file = os.path.join(results_base, 'weak_scaling_analysis.csv')
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Create plots
    create_scaling_plots(df, results_base, 'weak')
    
    return df


def analyze_sensitivity(results_base):
    """Analyze sensitivity sweep results"""
    print("Analyzing sensitivity sweep results...")
    
    # Find all parameter directories
    param_dirs = sorted(glob.glob(os.path.join(results_base, 'batch*_workers*')))
    
    results = []
    for param_dir in param_dirs:
        dir_name = os.path.basename(param_dir)
        try:
            # Extract batch_size and num_workers
            parts = dir_name.replace('batch', '').replace('workers', '').split('_')
            batch_size = int(parts[0])
            num_workers = int(parts[1])
        except:
            continue
        
        metrics = load_metrics(param_dir)
        if metrics:
            metrics['batch_size'] = batch_size
            metrics['num_workers'] = num_workers
            results.append(metrics)
    
    if not results:
        print("No results found!")
        return None
    
    df = pd.DataFrame(results)
    
    # Save results
    output_file = os.path.join(results_base, 'sensitivity_analysis.csv')
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Create plots
    create_sensitivity_plots(df, results_base)
    
    return df


def create_scaling_plots(df, output_dir, scaling_type):
    """Create scaling plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    num_nodes = df['num_nodes'].values
    time = df['epoch_time_sec'].values
    throughput = df['throughput_samples_per_sec'].values
    
    # Plot 1: Time vs Nodes
    ax = axes[0, 0]
    ax.plot(num_nodes, time, 'o-', label='Actual', linewidth=2, markersize=8)
    if scaling_type == 'strong':
        # Ideal scaling: time should decrease linearly
        ideal_time = time[0] / num_nodes
        ax.plot(num_nodes, ideal_time, '--', label='Ideal', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title(f'{scaling_type.capitalize()} Scaling: Time vs Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Throughput vs Nodes
    ax = axes[0, 1]
    ax.plot(num_nodes, throughput, 'o-', label='Actual', linewidth=2, markersize=8)
    if scaling_type == 'strong':
        ideal_throughput = throughput[0] * num_nodes
        ax.plot(num_nodes, ideal_throughput, '--', label='Ideal', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Throughput (samples/s)')
    ax.set_title(f'{scaling_type.capitalize()} Scaling: Throughput vs Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speedup/Efficiency
    ax = axes[1, 0]
    if scaling_type == 'strong' and 'speedup' in df.columns:
        ax.plot(num_nodes, df['speedup'], 'o-', label='Actual Speedup', linewidth=2, markersize=8)
        ax.plot(num_nodes, df['ideal_speedup'], '--', label='Ideal Speedup', linewidth=2, alpha=0.7)
        ax.set_ylabel('Speedup')
    else:
        ax.plot(num_nodes, df['time_ratio'], 'o-', label='Time Ratio', linewidth=2, markersize=8)
        ax.axhline(y=1.0, linestyle='--', label='Ideal', linewidth=2, alpha=0.7)
        ax.set_ylabel('Time Ratio')
    ax.set_xlabel('Number of Nodes')
    ax.set_title(f'{scaling_type.capitalize()} Scaling: Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency
    ax = axes[1, 1]
    if 'efficiency' in df.columns:
        ax.plot(num_nodes, df['efficiency'], 'o-', label='Efficiency', linewidth=2, markersize=8)
        ax.axhline(y=1.0, linestyle='--', label='Ideal (100%)', linewidth=2, alpha=0.7)
        ax.set_ylabel('Parallel Efficiency')
        ax.set_xlabel('Number of Nodes')
        ax.set_title(f'{scaling_type.capitalize()} Scaling: Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.2])
    
    plt.tight_layout()
    
    # Save plots
    plot_file = os.path.join(output_dir, f'{scaling_type}_scaling_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    # Also save as SVG
    plot_file_svg = os.path.join(output_dir, f'{scaling_type}_scaling_plots.svg')
    plt.savefig(plot_file_svg, format='svg', bbox_inches='tight')
    
    plt.close()


def create_sensitivity_plots(df, output_dir):
    """Create sensitivity analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    batch_sizes = sorted(df['batch_size'].unique())
    num_workers_list = sorted(df['num_workers'].unique())
    
    # Plot 1: Time vs Batch Size
    ax = axes[0, 0]
    for nw in num_workers_list:
        subset = df[df['num_workers'] == nw].sort_values('batch_size')
        ax.plot(subset['batch_size'], subset['epoch_time_sec'], 'o-', 
                label=f'workers={nw}', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Sensitivity: Time vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Throughput vs Batch Size
    ax = axes[0, 1]
    for nw in num_workers_list:
        subset = df[df['num_workers'] == nw].sort_values('batch_size')
        ax.plot(subset['batch_size'], subset['throughput_samples_per_sec'], 'o-',
                label=f'workers={nw}', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (samples/s)')
    ax.set_title('Sensitivity: Throughput vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Time vs Num Workers
    ax = axes[1, 0]
    for bs in batch_sizes:
        subset = df[df['batch_size'] == bs].sort_values('num_workers')
        ax.plot(subset['num_workers'], subset['epoch_time_sec'], 'o-',
                label=f'batch={bs}', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Sensitivity: Time vs Num Workers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of throughput
    ax = axes[1, 1]
    pivot = df.pivot(index='num_workers', columns='batch_size', values='throughput_samples_per_sec')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of Workers')
    ax.set_title('Sensitivity: Throughput Heatmap')
    plt.colorbar(im, ax=ax, label='Throughput (samples/s)')
    
    plt.tight_layout()
    
    # Save plots
    plot_file = os.path.join(output_dir, 'sensitivity_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    # Also save as SVG
    plot_file_svg = os.path.join(output_dir, 'sensitivity_plots.svg')
    plt.savefig(plot_file_svg, format='svg', bbox_inches='tight')
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze scaling results')
    parser.add_argument('--results', type=str, required=True,
                        help='Base results directory')
    parser.add_argument('--type', type=str, choices=['strong', 'weak', 'sensitivity'],
                        required=True, help='Type of scaling analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results directory not found: {args.results}")
        return
    
    if args.type == 'strong':
        analyze_strong_scaling(args.results)
    elif args.type == 'weak':
        analyze_weak_scaling(args.results)
    elif args.type == 'sensitivity':
        analyze_sensitivity(args.results)


if __name__ == '__main__':
    main()

