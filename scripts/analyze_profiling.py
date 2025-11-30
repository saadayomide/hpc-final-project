#!/usr/bin/env python3
"""
Profiling Analysis Script

Analyzes profiling results from GPU/CPU monitoring and generates
bottleneck analysis reports and visualizations.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_gpu_metrics(results_dir):
    """Load GPU monitoring data"""
    gpu_file = Path(results_dir) / 'gpu_monitor.csv'
    if not gpu_file.exists():
        return None
    
    df = pd.read_csv(gpu_file)
    return df


def load_cpu_metrics(results_dir):
    """Load CPU monitoring data"""
    cpu_file = Path(results_dir) / 'cpu_monitor.csv'
    if not cpu_file.exists():
        return None
    
    df = pd.read_csv(cpu_file)
    return df


def load_training_metrics(results_dir):
    """Load training metrics"""
    metrics_file = Path(results_dir) / 'metrics.csv'
    if not metrics_file.exists():
        return None
    
    df = pd.read_csv(metrics_file)
    return df


def analyze_gpu_utilization(gpu_df):
    """Analyze GPU utilization patterns"""
    if gpu_df is None:
        return None
    
    analysis = {
        'avg_gpu_util': gpu_df['utilization_gpu'].mean(),
        'max_gpu_util': gpu_df['utilization_gpu'].max(),
        'min_gpu_util': gpu_df['utilization_gpu'].min(),
        'std_gpu_util': gpu_df['utilization_gpu'].std(),
        'avg_memory_util': gpu_df['utilization_memory'].mean(),
        'max_memory_used_mb': gpu_df['memory_used_mb'].max(),
        'avg_power_draw_w': gpu_df['power_draw_w'].mean() if 'power_draw_w' in gpu_df else 0,
        'avg_temperature': gpu_df['temperature'].mean() if 'temperature' in gpu_df else 0,
    }
    
    # Identify underutilization periods
    low_util_threshold = 50
    low_util_samples = (gpu_df['utilization_gpu'] < low_util_threshold).sum()
    analysis['low_util_fraction'] = low_util_samples / len(gpu_df)
    
    return analysis


def analyze_cpu_utilization(cpu_df):
    """Analyze CPU utilization patterns"""
    if cpu_df is None:
        return None
    
    analysis = {
        'avg_cpu_percent': cpu_df['cpu_percent'].mean(),
        'max_cpu_percent': cpu_df['cpu_percent'].max(),
        'avg_memory_percent': cpu_df['memory_percent'].mean(),
        'max_memory_used_gb': cpu_df['memory_used_gb'].max(),
    }
    
    return analysis


def analyze_training_performance(metrics_df):
    """Analyze training performance"""
    if metrics_df is None:
        return None
    
    analysis = {
        'total_epochs': len(metrics_df),
        'avg_epoch_time': metrics_df['epoch_time_sec'].mean(),
        'min_epoch_time': metrics_df['epoch_time_sec'].min(),
        'max_epoch_time': metrics_df['epoch_time_sec'].max(),
        'avg_throughput': metrics_df['throughput_samples_per_sec'].mean(),
        'final_train_loss': metrics_df['train_loss'].iloc[-1],
        'final_val_loss': metrics_df['val_loss'].iloc[-1],
        'final_val_mae': metrics_df['val_mae'].iloc[-1],
        'final_val_rmse': metrics_df['val_rmse'].iloc[-1],
    }
    
    # Calculate data loading overhead
    if 'data_load_time_sec' in metrics_df and 'compute_time_sec' in metrics_df:
        total_data_time = metrics_df['data_load_time_sec'].sum()
        total_compute_time = metrics_df['compute_time_sec'].sum()
        total_time = metrics_df['epoch_time_sec'].sum()
        
        analysis['data_load_fraction'] = total_data_time / total_time if total_time > 0 else 0
        analysis['compute_fraction'] = total_compute_time / total_time if total_time > 0 else 0
        analysis['overhead_fraction'] = 1 - (total_data_time + total_compute_time) / total_time if total_time > 0 else 0
    
    return analysis


def identify_bottlenecks(gpu_analysis, cpu_analysis, training_analysis):
    """Identify performance bottlenecks"""
    bottlenecks = []
    recommendations = []
    
    if gpu_analysis:
        # GPU underutilization
        if gpu_analysis['avg_gpu_util'] < 70:
            bottlenecks.append({
                'type': 'GPU Underutilization',
                'severity': 'HIGH' if gpu_analysis['avg_gpu_util'] < 50 else 'MEDIUM',
                'value': f"{gpu_analysis['avg_gpu_util']:.1f}%",
                'description': 'GPU is not being fully utilized'
            })
            recommendations.append('Increase batch size to improve GPU utilization')
            recommendations.append('Check for data loading bottlenecks')
        
        # Memory pressure
        if gpu_analysis['avg_memory_util'] > 90:
            bottlenecks.append({
                'type': 'GPU Memory Pressure',
                'severity': 'HIGH',
                'value': f"{gpu_analysis['avg_memory_util']:.1f}%",
                'description': 'GPU memory is nearly full'
            })
            recommendations.append('Consider gradient checkpointing')
            recommendations.append('Reduce batch size or use mixed precision')
    
    if training_analysis:
        # Data loading bottleneck
        if training_analysis.get('data_load_fraction', 0) > 0.3:
            bottlenecks.append({
                'type': 'Data Loading Bottleneck',
                'severity': 'HIGH' if training_analysis['data_load_fraction'] > 0.5 else 'MEDIUM',
                'value': f"{training_analysis['data_load_fraction']*100:.1f}%",
                'description': 'Significant time spent loading data'
            })
            recommendations.append('Increase num_workers for data loading')
            recommendations.append('Use pin_memory=True')
            recommendations.append('Consider prefetching and caching')
        
        # Communication overhead
        if training_analysis.get('overhead_fraction', 0) > 0.2:
            bottlenecks.append({
                'type': 'Communication/Sync Overhead',
                'severity': 'MEDIUM',
                'value': f"{training_analysis['overhead_fraction']*100:.1f}%",
                'description': 'Overhead from distributed communication'
            })
            recommendations.append('Consider gradient accumulation')
            recommendations.append('Optimize all-reduce operations')
    
    if cpu_analysis:
        # CPU bottleneck
        if cpu_analysis['avg_cpu_percent'] > 90:
            bottlenecks.append({
                'type': 'CPU Bottleneck',
                'severity': 'MEDIUM',
                'value': f"{cpu_analysis['avg_cpu_percent']:.1f}%",
                'description': 'CPU is heavily loaded'
            })
            recommendations.append('Reduce preprocessing complexity')
            recommendations.append('Move preprocessing to GPU')
    
    return bottlenecks, recommendations


def create_profiling_plots(gpu_df, cpu_df, metrics_df, output_dir):
    """Create profiling visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # GPU Utilization over time
    ax = axes[0, 0]
    if gpu_df is not None:
        time = (gpu_df['timestamp'] - gpu_df['timestamp'].iloc[0])
        ax.plot(time, gpu_df['utilization_gpu'], label='GPU Util', alpha=0.7)
        ax.axhline(y=gpu_df['utilization_gpu'].mean(), color='r', linestyle='--', 
                   label=f"Avg: {gpu_df['utilization_gpu'].mean():.1f}%")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No GPU data', ha='center', va='center')
    
    # GPU Memory usage
    ax = axes[0, 1]
    if gpu_df is not None:
        time = (gpu_df['timestamp'] - gpu_df['timestamp'].iloc[0])
        ax.plot(time, gpu_df['memory_used_mb'] / 1024, label='Memory Used (GB)', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No GPU data', ha='center', va='center')
    
    # CPU Utilization
    ax = axes[0, 2]
    if cpu_df is not None:
        time = (cpu_df['timestamp'] - cpu_df['timestamp'].iloc[0])
        ax.plot(time, cpu_df['cpu_percent'], label='CPU Util', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('CPU Utilization (%)')
        ax.set_title('CPU Utilization Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No CPU data', ha='center', va='center')
    
    # Training Loss
    ax = axes[1, 0]
    if metrics_df is not None:
        ax.plot(metrics_df['epoch'], metrics_df['train_loss'], 'o-', label='Train Loss')
        ax.plot(metrics_df['epoch'], metrics_df['val_loss'], 's-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training data', ha='center', va='center')
    
    # Throughput
    ax = axes[1, 1]
    if metrics_df is not None:
        ax.bar(metrics_df['epoch'], metrics_df['throughput_samples_per_sec'], alpha=0.7)
        ax.axhline(y=metrics_df['throughput_samples_per_sec'].mean(), color='r', 
                   linestyle='--', label=f"Avg: {metrics_df['throughput_samples_per_sec'].mean():.1f}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Throughput (samples/s)')
        ax.set_title('Training Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training data', ha='center', va='center')
    
    # Time breakdown
    ax = axes[1, 2]
    if metrics_df is not None and 'data_load_time_sec' in metrics_df and 'compute_time_sec' in metrics_df:
        avg_data_load = metrics_df['data_load_time_sec'].mean()
        avg_compute = metrics_df['compute_time_sec'].mean()
        avg_total = metrics_df['epoch_time_sec'].mean()
        avg_overhead = avg_total - avg_data_load - avg_compute
        
        labels = ['Data Loading', 'Compute', 'Overhead']
        sizes = [avg_data_load, avg_compute, max(0, avg_overhead)]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Time Breakdown per Epoch')
    else:
        ax.text(0.5, 0.5, 'No time breakdown data', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(output_dir / 'profiling_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'profiling_analysis.svg', format='svg', bbox_inches='tight')
    print(f"Saved plots to {output_dir}")
    
    plt.close()


def generate_report(gpu_analysis, cpu_analysis, training_analysis, bottlenecks, 
                    recommendations, output_dir):
    """Generate profiling report"""
    output_dir = Path(output_dir)
    
    report = []
    report.append("=" * 60)
    report.append("PROFILING ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    if gpu_analysis:
        report.append("GPU ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Average GPU Utilization: {gpu_analysis['avg_gpu_util']:.1f}%")
        report.append(f"  Max GPU Utilization: {gpu_analysis['max_gpu_util']:.1f}%")
        report.append(f"  Average Memory Utilization: {gpu_analysis['avg_memory_util']:.1f}%")
        report.append(f"  Max Memory Used: {gpu_analysis['max_memory_used_mb']:.0f} MB")
        if gpu_analysis['avg_power_draw_w'] > 0:
            report.append(f"  Average Power Draw: {gpu_analysis['avg_power_draw_w']:.1f} W")
        report.append("")
    
    if cpu_analysis:
        report.append("CPU ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Average CPU Utilization: {cpu_analysis['avg_cpu_percent']:.1f}%")
        report.append(f"  Max Memory Used: {cpu_analysis['max_memory_used_gb']:.1f} GB")
        report.append("")
    
    if training_analysis:
        report.append("TRAINING ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Total Epochs: {training_analysis['total_epochs']}")
        report.append(f"  Average Epoch Time: {training_analysis['avg_epoch_time']:.2f} s")
        report.append(f"  Average Throughput: {training_analysis['avg_throughput']:.1f} samples/s")
        report.append(f"  Final Train Loss: {training_analysis['final_train_loss']:.6f}")
        report.append(f"  Final Val Loss: {training_analysis['final_val_loss']:.6f}")
        report.append(f"  Final Val MAE: {training_analysis['final_val_mae']:.6f}")
        if 'data_load_fraction' in training_analysis:
            report.append(f"  Data Loading Time: {training_analysis['data_load_fraction']*100:.1f}%")
            report.append(f"  Compute Time: {training_analysis['compute_fraction']*100:.1f}%")
        report.append("")
    
    if bottlenecks:
        report.append("IDENTIFIED BOTTLENECKS")
        report.append("-" * 40)
        for b in bottlenecks:
            report.append(f"  [{b['severity']}] {b['type']}: {b['value']}")
            report.append(f"          {b['description']}")
        report.append("")
    
    if recommendations:
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for i, r in enumerate(recommendations, 1):
            report.append(f"  {i}. {r}")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    # Save report
    with open(output_dir / 'profiling_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {output_dir / 'profiling_report.txt'}")
    
    # Save as JSON for programmatic access
    analysis_data = {
        'gpu': gpu_analysis,
        'cpu': cpu_analysis,
        'training': training_analysis,
        'bottlenecks': bottlenecks,
        'recommendations': recommendations
    }
    with open(output_dir / 'profiling_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Analyze profiling results')
    parser.add_argument('--results', type=str, required=True,
                        help='Results directory containing profiling data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis (default: same as results)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output) if args.output else results_dir
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Analyzing profiling data from: {results_dir}")
    print()
    
    # Load data
    gpu_df = load_gpu_metrics(results_dir)
    cpu_df = load_cpu_metrics(results_dir)
    metrics_df = load_training_metrics(results_dir)
    
    # Analyze
    gpu_analysis = analyze_gpu_utilization(gpu_df)
    cpu_analysis = analyze_cpu_utilization(cpu_df)
    training_analysis = analyze_training_performance(metrics_df)
    
    # Identify bottlenecks
    bottlenecks, recommendations = identify_bottlenecks(
        gpu_analysis, cpu_analysis, training_analysis
    )
    
    # Create visualizations
    create_profiling_plots(gpu_df, cpu_df, metrics_df, output_dir)
    
    # Generate report
    generate_report(gpu_analysis, cpu_analysis, training_analysis,
                    bottlenecks, recommendations, output_dir)


if __name__ == '__main__':
    main()
