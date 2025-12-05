#!/usr/bin/env python3
"""
Profiling Analysis Script

Analyzes GPU/CPU monitoring data and generates bottleneck reports.

Usage:
    python scripts/analyze_profiling.py --results results/baseline_xxxx
"""

import argparse
import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will not be generated")


def parse_csv_timeseries(csv_path: str) -> Dict[str, List[float]]:
    """Parse a monitoring CSV file into time series data."""
    if not os.path.exists(csv_path):
        return {}
    
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(0.0)
    
    return data


def parse_metrics_csv(csv_path: str) -> List[Dict]:
    """Parse training metrics CSV."""
    if not os.path.exists(csv_path):
        return []
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (ValueError, TypeError):
                    parsed[key] = value
            rows.append(parsed)
    
    return rows


def analyze_gpu_utilization(gpu_data: Dict) -> Dict:
    """Analyze GPU utilization patterns."""
    if not gpu_data or 'utilization_gpu' not in gpu_data:
        return {'error': 'No GPU utilization data'}
    
    util = gpu_data['utilization_gpu']
    mem_util = gpu_data.get('utilization_memory', [])
    mem_used = gpu_data.get('memory_used_mb', [])
    power = gpu_data.get('power_draw_w', [])
    temp = gpu_data.get('temperature', [])
    
    if not HAS_NUMPY:
        avg_util = sum(util) / len(util) if util else 0
        analysis = {
            'avg_utilization': avg_util,
            'min_utilization': min(util) if util else 0,
            'max_utilization': max(util) if util else 0,
        }
    else:
        util = np.array(util)
        analysis = {
            'avg_utilization': float(np.mean(util)),
            'std_utilization': float(np.std(util)),
            'min_utilization': float(np.min(util)),
            'max_utilization': float(np.max(util)),
            'median_utilization': float(np.median(util)),
            'p25_utilization': float(np.percentile(util, 25)),
            'p75_utilization': float(np.percentile(util, 75)),
        }
        
        if mem_util:
            mem_util = np.array(mem_util)
            analysis['avg_memory_utilization'] = float(np.mean(mem_util))
        
        if mem_used:
            mem_used = np.array(mem_used)
            analysis['avg_memory_used_mb'] = float(np.mean(mem_used))
            analysis['max_memory_used_mb'] = float(np.max(mem_used))
        
        if power:
            power = np.array(power)
            analysis['avg_power_w'] = float(np.mean(power))
            analysis['max_power_w'] = float(np.max(power))
        
        if temp:
            temp = np.array(temp)
            analysis['avg_temperature_c'] = float(np.mean(temp))
            analysis['max_temperature_c'] = float(np.max(temp))
    
    # Categorize utilization
    avg = analysis['avg_utilization']
    if avg >= 85:
        analysis['utilization_category'] = 'High (compute-bound)'
    elif avg >= 60:
        analysis['utilization_category'] = 'Medium (potential optimization opportunity)'
    else:
        analysis['utilization_category'] = 'Low (memory/IO bound or underutilized)'
    
    return analysis


def analyze_cpu_utilization(cpu_data: Dict) -> Dict:
    """Analyze CPU utilization patterns."""
    if not cpu_data or 'cpu_percent' not in cpu_data:
        return {'error': 'No CPU utilization data'}
    
    cpu_util = cpu_data['cpu_percent']
    mem_percent = cpu_data.get('memory_percent', [])
    
    if not HAS_NUMPY:
        avg = sum(cpu_util) / len(cpu_util) if cpu_util else 0
        return {'avg_cpu_percent': avg}
    
    cpu_util = np.array(cpu_util)
    analysis = {
        'avg_cpu_percent': float(np.mean(cpu_util)),
        'std_cpu_percent': float(np.std(cpu_util)),
        'max_cpu_percent': float(np.max(cpu_util)),
        'min_cpu_percent': float(np.min(cpu_util)),
    }
    
    if mem_percent:
        mem = np.array(mem_percent)
        analysis['avg_memory_percent'] = float(np.mean(mem))
        analysis['max_memory_percent'] = float(np.max(mem))
    
    return analysis


def analyze_training_metrics(metrics: List[Dict]) -> Dict:
    """Analyze training progress and identify patterns."""
    if not metrics:
        return {'error': 'No training metrics'}
    
    epoch_times = [m.get('epoch_time_sec', 0) for m in metrics]
    throughputs = [m.get('throughput_samples_per_sec', 0) for m in metrics]
    data_load_times = [m.get('data_load_time_sec', 0) for m in metrics]
    compute_times = [m.get('compute_time_sec', 0) for m in metrics]
    train_losses = [m.get('train_loss', 0) for m in metrics]
    
    # Skip first epoch (warmup)
    if len(epoch_times) > 1:
        epoch_times = epoch_times[1:]
        throughputs = throughputs[1:]
        data_load_times = data_load_times[1:]
        compute_times = compute_times[1:]
    
    total_time = sum(epoch_times)
    total_data_load = sum(data_load_times)
    total_compute = sum(compute_times)
    
    # Calculate time breakdown
    data_load_pct = (total_data_load / total_time * 100) if total_time > 0 else 0
    compute_pct = (total_compute / total_time * 100) if total_time > 0 else 0
    other_pct = 100 - data_load_pct - compute_pct
    
    analysis = {
        'total_epochs': len(metrics),
        'total_time_sec': total_time,
        'avg_epoch_time_sec': sum(epoch_times) / len(epoch_times) if epoch_times else 0,
        'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
        'time_breakdown': {
            'data_loading_percent': data_load_pct,
            'compute_percent': compute_pct,
            'other_overhead_percent': max(0, other_pct),
        },
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'loss_reduction': (train_losses[0] - train_losses[-1]) if len(train_losses) > 1 else 0,
    }
    
    return analysis


def identify_bottlenecks(gpu_analysis: Dict, cpu_analysis: Dict, 
                         training_analysis: Dict) -> List[Dict]:
    """Identify performance bottlenecks based on analysis."""
    bottlenecks = []
    
    # Check GPU utilization
    gpu_util = gpu_analysis.get('avg_utilization', 0)
    if gpu_util < 60:
        bottlenecks.append({
            'type': 'LOW_GPU_UTILIZATION',
            'severity': 'HIGH',
            'description': f'GPU utilization is only {gpu_util:.1f}%',
            'recommendation': 'Increase batch size, reduce data loading overhead, or check for CPU bottlenecks',
        })
    elif gpu_util < 80:
        bottlenecks.append({
            'type': 'MEDIUM_GPU_UTILIZATION',
            'severity': 'MEDIUM',
            'description': f'GPU utilization at {gpu_util:.1f}% could be improved',
            'recommendation': 'Consider increasing batch size or using mixed precision training',
        })
    
    # Check data loading
    time_breakdown = training_analysis.get('time_breakdown', {})
    data_loading_pct = time_breakdown.get('data_loading_percent', 0)
    if data_loading_pct > 30:
        bottlenecks.append({
            'type': 'DATA_LOADING_BOTTLENECK',
            'severity': 'HIGH',
            'description': f'Data loading takes {data_loading_pct:.1f}% of training time',
            'recommendation': 'Increase num_workers, use pin_memory, or preload data to faster storage',
        })
    elif data_loading_pct > 15:
        bottlenecks.append({
            'type': 'DATA_LOADING_OVERHEAD',
            'severity': 'MEDIUM',
            'description': f'Data loading at {data_loading_pct:.1f}% is significant',
            'recommendation': 'Consider async data prefetching or data format optimization',
        })
    
    # Check CPU utilization (might indicate data loading issues)
    cpu_util = cpu_analysis.get('avg_cpu_percent', 0)
    if cpu_util > 80:
        bottlenecks.append({
            'type': 'HIGH_CPU_UTILIZATION',
            'severity': 'MEDIUM',
            'description': f'CPU utilization is {cpu_util:.1f}%',
            'recommendation': 'CPU may be bottleneck for data preprocessing; consider more workers or simpler transforms',
        })
    
    # Check memory
    mem_used = gpu_analysis.get('max_memory_used_mb', 0)
    mem_total = 16384  # Assume 16GB, adjust based on actual GPU
    if mem_used > 0 and mem_used / mem_total > 0.9:
        bottlenecks.append({
            'type': 'HIGH_GPU_MEMORY',
            'severity': 'MEDIUM',
            'description': f'GPU memory usage is high ({mem_used:.0f}MB)',
            'recommendation': 'May limit batch size scaling; consider gradient checkpointing',
        })
    
    # If no significant bottlenecks found
    if not bottlenecks:
        bottlenecks.append({
            'type': 'WELL_OPTIMIZED',
            'severity': 'INFO',
            'description': 'No major bottlenecks identified',
            'recommendation': 'System appears well-balanced; focus on algorithmic improvements',
        })
    
    return bottlenecks


def plot_profiling(gpu_data: Dict, cpu_data: Dict, metrics: List[Dict], 
                   bottlenecks: List[Dict], output_dir: str):
    """Generate profiling visualization plots."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Profiling Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: GPU Utilization over time
    ax1 = axes[0, 0]
    if gpu_data and 'utilization_gpu' in gpu_data:
        timestamps = gpu_data.get('timestamp', list(range(len(gpu_data['utilization_gpu']))))
        ax1.plot(timestamps, gpu_data['utilization_gpu'], 'b-', alpha=0.7, label='GPU Util')
        if 'utilization_memory' in gpu_data:
            ax1.plot(timestamps, gpu_data['utilization_memory'], 'r-', alpha=0.7, label='Memory Util')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('GPU Utilization Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
    else:
        ax1.text(0.5, 0.5, 'No GPU data', ha='center', va='center')
        ax1.set_title('GPU Utilization (no data)')
    
    # Plot 2: CPU/Memory utilization
    ax2 = axes[0, 1]
    if cpu_data and 'cpu_percent' in cpu_data:
        timestamps = cpu_data.get('timestamp', list(range(len(cpu_data['cpu_percent']))))
        ax2.plot(timestamps, cpu_data['cpu_percent'], 'g-', alpha=0.7, label='CPU Util')
        if 'memory_percent' in cpu_data:
            ax2.plot(timestamps, cpu_data['memory_percent'], 'm-', alpha=0.7, label='Memory')
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Utilization (%)')
        ax2.set_title('CPU/Memory Utilization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    else:
        ax2.text(0.5, 0.5, 'No CPU data', ha='center', va='center')
        ax2.set_title('CPU Utilization (no data)')
    
    # Plot 3: Training loss curve
    ax3 = axes[1, 0]
    if metrics:
        epochs = [m.get('epoch', i+1) for i, m in enumerate(metrics)]
        train_loss = [m.get('train_loss', 0) for m in metrics]
        val_loss = [m.get('val_loss', 0) for m in metrics]
        ax3.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
        ax3.plot(epochs, val_loss, 'r--', linewidth=2, label='Val Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No training data', ha='center', va='center')
    
    # Plot 4: Time breakdown pie chart
    ax4 = axes[1, 1]
    if metrics:
        data_load = sum(m.get('data_load_time_sec', 0) for m in metrics)
        compute = sum(m.get('compute_time_sec', 0) for m in metrics)
        total = sum(m.get('epoch_time_sec', 0) for m in metrics)
        other = max(0, total - data_load - compute)
        
        if total > 0:
            sizes = [compute, data_load, other]
            labels = [f'Compute\n({compute/total*100:.1f}%)', 
                     f'Data Loading\n({data_load/total*100:.1f}%)',
                     f'Other\n({other/total*100:.1f}%)']
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            ax4.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
            ax4.set_title('Time Breakdown')
        else:
            ax4.text(0.5, 0.5, 'No timing data', ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    plt.tight_layout()
    
    for fmt in ['png', 'svg']:
        path = os.path.join(output_dir, f'profiling_analysis.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.close()


def generate_report(gpu_analysis: Dict, cpu_analysis: Dict, 
                    training_analysis: Dict, bottlenecks: List[Dict],
                    output_dir: str):
    """Generate a text report summarizing the profiling analysis."""
    
    report_path = os.path.join(output_dir, 'profiling_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PROFILING ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # GPU Analysis
        f.write("GPU UTILIZATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if 'error' not in gpu_analysis:
            f.write(f"  Average Utilization: {gpu_analysis.get('avg_utilization', 0):.1f}%\n")
            f.write(f"  Std Dev: {gpu_analysis.get('std_utilization', 0):.1f}%\n")
            f.write(f"  Range: {gpu_analysis.get('min_utilization', 0):.1f}% - {gpu_analysis.get('max_utilization', 0):.1f}%\n")
            f.write(f"  Category: {gpu_analysis.get('utilization_category', 'Unknown')}\n")
            if 'avg_memory_used_mb' in gpu_analysis:
                f.write(f"  Memory Used (avg): {gpu_analysis['avg_memory_used_mb']:.0f} MB\n")
            if 'avg_power_w' in gpu_analysis:
                f.write(f"  Power Draw (avg): {gpu_analysis['avg_power_w']:.0f} W\n")
        else:
            f.write(f"  {gpu_analysis['error']}\n")
        f.write("\n")
        
        # CPU Analysis
        f.write("CPU UTILIZATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if 'error' not in cpu_analysis:
            f.write(f"  Average CPU: {cpu_analysis.get('avg_cpu_percent', 0):.1f}%\n")
            f.write(f"  Max CPU: {cpu_analysis.get('max_cpu_percent', 0):.1f}%\n")
            if 'avg_memory_percent' in cpu_analysis:
                f.write(f"  Memory Used: {cpu_analysis['avg_memory_percent']:.1f}%\n")
        else:
            f.write(f"  {cpu_analysis['error']}\n")
        f.write("\n")
        
        # Training Analysis
        f.write("TRAINING PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        if 'error' not in training_analysis:
            f.write(f"  Total Epochs: {training_analysis.get('total_epochs', 0)}\n")
            f.write(f"  Total Time: {training_analysis.get('total_time_sec', 0):.1f} seconds\n")
            f.write(f"  Avg Epoch Time: {training_analysis.get('avg_epoch_time_sec', 0):.1f} seconds\n")
            f.write(f"  Avg Throughput: {training_analysis.get('avg_throughput', 0):.1f} samples/sec\n")
            
            breakdown = training_analysis.get('time_breakdown', {})
            f.write("\n  Time Breakdown:\n")
            f.write(f"    Compute: {breakdown.get('compute_percent', 0):.1f}%\n")
            f.write(f"    Data Loading: {breakdown.get('data_loading_percent', 0):.1f}%\n")
            f.write(f"    Other Overhead: {breakdown.get('other_overhead_percent', 0):.1f}%\n")
        else:
            f.write(f"  {training_analysis['error']}\n")
        f.write("\n")
        
        # Bottlenecks
        f.write("IDENTIFIED BOTTLENECKS\n")
        f.write("-" * 40 + "\n")
        for b in bottlenecks:
            f.write(f"\n  [{b['severity']}] {b['type']}\n")
            f.write(f"    {b['description']}\n")
            f.write(f"    Recommendation: {b['recommendation']}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Saved: {report_path}")
    
    # Also print to console
    with open(report_path, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description='Analyze profiling data from training runs')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results directory containing metrics.csv and monitoring CSVs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis (default: same as results)')
    
    args = parser.parse_args()
    
    results_dir = args.results
    output_dir = args.output or results_dir
    
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse data files
    gpu_data = parse_csv_timeseries(os.path.join(results_dir, 'gpu_monitor.csv'))
    cpu_data = parse_csv_timeseries(os.path.join(results_dir, 'cpu_monitor.csv'))
    metrics = parse_metrics_csv(os.path.join(results_dir, 'metrics.csv'))
    
    # Analyze
    gpu_analysis = analyze_gpu_utilization(gpu_data)
    cpu_analysis = analyze_cpu_utilization(cpu_data)
    training_analysis = analyze_training_metrics(metrics)
    bottlenecks = identify_bottlenecks(gpu_analysis, cpu_analysis, training_analysis)
    
    # Generate outputs
    generate_report(gpu_analysis, cpu_analysis, training_analysis, bottlenecks, output_dir)
    plot_profiling(gpu_data, cpu_data, metrics, bottlenecks, output_dir)
    
    # Save JSON
    full_analysis = {
        'gpu_analysis': gpu_analysis,
        'cpu_analysis': cpu_analysis,
        'training_analysis': training_analysis,
        'bottlenecks': bottlenecks,
    }
    json_path = os.path.join(output_dir, 'profiling_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
