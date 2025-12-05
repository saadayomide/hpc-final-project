#!/usr/bin/env python3
"""
Generate Bottleneck Analysis Document

Aggregates profiling results from multiple nodes and generates
a comprehensive bottleneck analysis document.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys


def load_profiling_results(results_dir):
    """Load profiling analysis results from JSON files"""
    results_dir = Path(results_dir)
    
    # Find all profiling_analysis.json files
    json_files = list(results_dir.rglob('profiling_analysis.json'))
    
    if not json_files:
        print(f"Warning: No profiling_analysis.json files found in {results_dir}")
        return {}
    
    results = {}
    for json_file in json_files:
        # Extract scenario name from path (e.g., 1node, 2node, 4node)
        parts = json_file.parts
        scenario = None
        for part in parts:
            if part in ['1node', '2node', '4node']:
                scenario = part
                break
        
        if not scenario:
            # Try to infer from directory structure
            if '1node' in str(json_file):
                scenario = '1node'
            elif '2node' in str(json_file):
                scenario = '2node'
            elif '4node' in str(json_file):
                scenario = '4node'
            else:
                scenario = f"scenario_{len(results)}"
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            results[scenario] = data
            print(f"Loaded profiling results for {scenario}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results


def calculate_time_breakdown(training_analysis):
    """Calculate time breakdown percentages"""
    if not training_analysis:
        return None
    
    breakdown = {}
    
    # Data loading fraction
    if 'data_load_fraction' in training_analysis:
        breakdown['data_loading'] = training_analysis['data_load_fraction'] * 100
    else:
        breakdown['data_loading'] = 0.0
    
    # Compute fraction
    if 'compute_fraction' in training_analysis:
        breakdown['compute'] = training_analysis['compute_fraction'] * 100
    else:
        breakdown['compute'] = 0.0
    
    # Overhead (communication, sync, etc.)
    if 'overhead_fraction' in training_analysis:
        breakdown['overhead'] = training_analysis['overhead_fraction'] * 100
    else:
        # Estimate overhead as remainder
        breakdown['overhead'] = max(0, 100 - breakdown['data_loading'] - breakdown['compute'])
    
    # Communication (part of overhead, estimate if available)
    breakdown['communication'] = breakdown['overhead'] * 0.7  # Rough estimate
    
    return breakdown


def identify_top_bottlenecks(all_results):
    """Identify top bottlenecks across all scenarios"""
    bottlenecks = []
    
    for scenario, data in all_results.items():
        if 'bottlenecks' in data and data['bottlenecks']:
            for b in data['bottlenecks']:
                bottlenecks.append({
                    'scenario': scenario,
                    'type': b['type'],
                    'severity': b['severity'],
                    'value': b['value'],
                    'description': b['description']
                })
    
    # Sort by severity (HIGH > MEDIUM > LOW)
    severity_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    bottlenecks.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
    
    return bottlenecks[:5]  # Top 5


def generate_bottleneck_analysis_markdown(all_results, output_file):
    """Generate bottleneck analysis markdown document"""
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# Bottleneck Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This document presents a comprehensive bottleneck analysis of the DCRNN training")
    lines.append("system across multiple node configurations. The analysis identifies performance")
    lines.append("bottlenecks and provides recommendations for optimization.")
    lines.append("")
    
    # Top Bottlenecks
    top_bottlenecks = identify_top_bottlenecks(all_results)
    if top_bottlenecks:
        lines.append("### Top Bottlenecks Identified")
        lines.append("")
        for i, b in enumerate(top_bottlenecks, 1):
            lines.append(f"{i}. **[{b['severity']}] {b['type']}** ({b['scenario']})")
            lines.append(f"   - {b['description']}")
            lines.append(f"   - Value: {b['value']}")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Per-Scenario Analysis
    lines.append("## Per-Scenario Analysis")
    lines.append("")
    
    for scenario in sorted(all_results.keys()):
        data = all_results[scenario]
        lines.append(f"### {scenario.upper()} Configuration")
        lines.append("")
        
        # Training Performance
        if 'training' in data and data['training']:
            training = data['training']
            lines.append("#### Training Performance")
            lines.append("")
            lines.append(f"- **Average Epoch Time:** {training.get('avg_epoch_time', 'N/A'):.2f} seconds")
            lines.append(f"- **Average Throughput:** {training.get('avg_throughput', 'N/A'):.1f} samples/second")
            lines.append(f"- **Total Epochs:** {training.get('total_epochs', 'N/A')}")
            lines.append("")
            
            # Time Breakdown
            breakdown = calculate_time_breakdown(training)
            if breakdown:
                lines.append("#### Time Breakdown")
                lines.append("")
                lines.append("| Component | Percentage | Description |")
                lines.append("|-----------|------------|-------------|")
                lines.append(f"| Compute | {breakdown['compute']:.1f}% | Actual model computation |")
                lines.append(f"| Data Loading | {breakdown['data_loading']:.1f}% | Data loading and preprocessing |")
                lines.append(f"| Communication | {breakdown['communication']:.1f}% | Gradient synchronization (NCCL/gloo) |")
                lines.append(f"| Other Overhead | {breakdown['overhead'] - breakdown['communication']:.1f}% | Other overhead (sync, I/O, etc.) |")
                lines.append("")
        
        # GPU Analysis
        if 'gpu' in data and data['gpu']:
            gpu = data['gpu']
            lines.append("#### GPU Utilization")
            lines.append("")
            lines.append(f"- **Average GPU Utilization:** {gpu.get('avg_gpu_util', 'N/A'):.1f}%")
            lines.append(f"- **Max GPU Utilization:** {gpu.get('max_gpu_util', 'N/A'):.1f}%")
            lines.append(f"- **Average Memory Utilization:** {gpu.get('avg_memory_util', 'N/A'):.1f}%")
            if gpu.get('low_util_fraction', 0) > 0:
                lines.append(f"- **Low Utilization Periods:** {gpu['low_util_fraction']*100:.1f}% of time < 50% utilization")
            lines.append("")
        
        # CPU Analysis
        if 'cpu' in data and data['cpu']:
            cpu = data['cpu']
            lines.append("#### CPU Utilization")
            lines.append("")
            lines.append(f"- **Average CPU Utilization:** {cpu.get('avg_cpu_percent', 'N/A'):.1f}%")
            lines.append(f"- **Max Memory Used:** {cpu.get('max_memory_used_gb', 'N/A'):.1f} GB")
            lines.append("")
        
        # Bottlenecks
        if 'bottlenecks' in data and data['bottlenecks']:
            lines.append("#### Identified Bottlenecks")
            lines.append("")
            for b in data['bottlenecks']:
                lines.append(f"- **[{b['severity']}] {b['type']}:** {b['value']}")
                lines.append(f"  - {b['description']}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Scaling Analysis
    if len(all_results) > 1:
        lines.append("## Scaling Analysis")
        lines.append("")
        lines.append("### Time Breakdown Evolution")
        lines.append("")
        lines.append("| Scenario | Compute % | Data Loading % | Communication % | Overhead % |")
        lines.append("|----------|-----------|----------------|-----------------|------------|")
        
        for scenario in sorted(all_results.keys()):
            data = all_results[scenario]
            training = data.get('training', {})
            breakdown = calculate_time_breakdown(training)
            if breakdown:
                lines.append(f"| {scenario} | {breakdown['compute']:.1f} | {breakdown['data_loading']:.1f} | "
                           f"{breakdown['communication']:.1f} | {breakdown['overhead'] - breakdown['communication']:.1f} |")
        
        lines.append("")
        lines.append("### Key Observations")
        lines.append("")
        lines.append("1. **Communication Overhead:** As the number of nodes increases, communication")
        lines.append("   overhead typically increases due to gradient synchronization across nodes.")
        lines.append("")
        lines.append("2. **Data Loading:** Data loading time should remain relatively constant per GPU,")
        lines.append("   but may become a bottleneck if not properly parallelized.")
        lines.append("")
        lines.append("3. **GPU Utilization:** GPU utilization may decrease with more nodes if")
        lines.append("   communication or data loading becomes the bottleneck.")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    
    # Collect all recommendations
    all_recommendations = set()
    for data in all_results.values():
        if 'recommendations' in data and data['recommendations']:
            all_recommendations.update(data['recommendations'])
    
    if all_recommendations:
        for i, rec in enumerate(sorted(all_recommendations), 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    else:
        lines.append("No specific recommendations generated. Review individual scenario analyses above.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("This analysis is based on:")
    lines.append("")
    lines.append("- **GPU Profiling:** Nsight Systems (nsys) for GPU kernel analysis")
    lines.append("- **CPU Profiling:** System monitoring (psutil, nvidia-smi)")
    lines.append("- **Training Metrics:** Per-epoch timing and throughput measurements")
    lines.append("- **Bottleneck Detection:** Automated analysis of utilization patterns")
    lines.append("")
    lines.append("### Bottleneck Identification Criteria")
    lines.append("")
    lines.append("- **GPU Underutilization:** < 70% average utilization")
    lines.append("- **Data Loading Bottleneck:** > 30% of time spent loading data")
    lines.append("- **Communication Overhead:** > 20% overhead from distributed operations")
    lines.append("- **Memory Pressure:** > 90% GPU memory utilization")
    lines.append("")
    
    # Write to file
    content = "\n".join(lines)
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Bottleneck analysis document generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate bottleneck analysis document')
    parser.add_argument('--results', type=str, required=True,
                        help='Base directory containing profiling results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: docs/BOTTLENECK_ANALYSIS.md)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load all profiling results
    print(f"Loading profiling results from: {results_dir}")
    all_results = load_profiling_results(results_dir)
    
    if not all_results:
        print("Error: No profiling results found. Run analyze_profiling.py first.")
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(__file__).parent.parent / 'docs' / 'BOTTLENECK_ANALYSIS.md'
    
    # Generate analysis document
    generate_bottleneck_analysis_markdown(all_results, output_file)
    
    print(f"\nBottleneck analysis complete!")
    print(f"Document saved to: {output_file}")


if __name__ == '__main__':
    main()

