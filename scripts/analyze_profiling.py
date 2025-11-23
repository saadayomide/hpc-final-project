#!/usr/bin/env python3
"""
Phase 3: Analyze profiling results and identify bottlenecks

This script processes profiling outputs (Nsight Systems reports) and generates
a summary of bottlenecks for compute, memory, I/O, and communication.
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

def parse_nsys_summary(summary_file):
    """Parse Nsight Systems summary statistics CSV"""
    summary = {}
    
    if not os.path.exists(summary_file):
        return summary
    
    try:
        with open(summary_file, 'r') as f:
            # Try to parse CSV (may have multiple sections)
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) >= 2:
                    key = row[0].strip()
                    value = row[1].strip() if len(row) > 1 else ""
                    summary[key] = value
    except Exception as e:
        print(f"Warning: Could not parse summary file {summary_file}: {e}")
    
    return summary


def analyze_profile_directory(profile_dir):
    """Analyze a single profiling run directory"""
    results = {
        'directory': str(profile_dir),
        'metadata': {},
        'nsys_available': False,
        'summary_stats': {},
        'bottlenecks': {}
    }
    
    # Read job metadata
    metadata_file = profile_dir / 'job_metadata.txt'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    results['metadata'][key.strip()] = value.strip()
    
    # Check for Nsight Systems report
    nsys_dir = profile_dir / 'nsys'
    if nsys_dir.exists():
        nsys_reports = list(nsys_dir.glob('*.nsys-rep'))
        if nsys_reports:
            results['nsys_available'] = True
            results['nsys_report'] = str(nsys_reports[0])
            
            # Try to parse summary stats
            summary_file = nsys_dir / 'summary_stats.csv'
            if summary_file.exists():
                results['summary_stats'] = parse_nsys_summary(summary_file)
    
    # Try to identify bottlenecks from available data
    # This is a simplified analysis - full analysis would require opening .nsys-rep files
    results['bottlenecks'] = identify_bottlenecks(results)
    
    return results


def identify_bottlenecks(profile_data):
    """Identify potential bottlenecks from profiling data"""
    bottlenecks = {
        'compute_bound': False,
        'memory_bound': False,
        'io_bound': False,
        'communication_bound': False,
        'data_loader_bound': False,
        'notes': []
    }
    
    # Simplified bottleneck identification
    # Full analysis would require parsing .nsys-rep files with nsys API
    
    # Check metadata for node count
    nodes = profile_data['metadata'].get('Nodes', '1')
    try:
        node_count = int(nodes)
    except:
        node_count = 1
    
    # Communication bottleneck more likely at higher node counts
    if node_count > 1:
        bottlenecks['communication_bound'] = True
        bottlenecks['notes'].append(
            f"Multi-node run ({node_count} nodes): Communication overhead likely"
        )
    
    # Check if summary stats are available
    if profile_data['summary_stats']:
        # Look for NCCL-related metrics
        for key in profile_data['summary_stats'].keys():
            if 'nccl' in key.lower():
                bottlenecks['communication_bound'] = True
                bottlenecks['notes'].append(
                    f"NCCL activity detected: {key}"
                )
    
    return bottlenecks


def generate_bottleneck_summary(all_results):
    """Generate a summary of bottlenecks across all profiling runs"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'scenarios': {},
        'comparison': {},
        'key_findings': []
    }
    
    # Organize by scenario
    for result in all_results:
        scenario = result['metadata'].get('Scenario', 'unknown')
        summary['scenarios'][scenario] = result
        
        # Extract node count
        nodes = result['metadata'].get('Nodes', '1')
        summary['comparison'][nodes] = {
            'scenario': scenario,
            'bottlenecks': result['bottlenecks'],
            'metadata': result['metadata']
        }
    
    # Compare across scenarios
    node_counts = sorted([int(k) for k in summary['comparison'].keys() if k.isdigit()])
    
    if len(node_counts) >= 2:
        # Look for scaling patterns
        summary['key_findings'].append(
            f"Profiling runs completed for {len(node_counts)} scenarios: {node_counts} node(s)"
        )
        
        # Check if communication bottleneck increases with nodes
        comm_bottlenecks = [
            summary['comparison'][str(n)]['bottlenecks'].get('communication_bound', False)
            for n in node_counts
        ]
        
        if any(comm_bottlenecks):
            summary['key_findings'].append(
                "Communication bottlenecks detected in multi-node runs"
            )
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Phase 3 profiling results and identify bottlenecks'
    )
    parser.add_argument(
        '--results',
        type=str,
        default='results/profiling',
        help='Path to profiling results directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/profiling/bottleneck_analysis.json',
        help='Output file for analysis results'
    )
    parser.add_argument(
        '--summary',
        type=str,
        default='results/profiling/bottleneck_summary.txt',
        help='Output file for text summary'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        sys.exit(1)
    
    print("==========================================")
    print("Phase 3: Analyzing Profiling Results")
    print("==========================================")
    print(f"Results directory: {results_path}")
    print("")
    
    # Find all profiling run directories
    # Look for directories with job_metadata.txt
    profile_dirs = []
    
    # Check 1node, 2node, 4node subdirectories
    for node_dir in ['1node', '2node', '4node']:
        node_path = results_path / node_dir
        if node_path.exists():
            # Find run directories (both run_* and direct directories)
            for run_dir in node_path.iterdir():
                if run_dir.is_dir():
                    # Check for job_metadata.txt in this directory
                    if (run_dir / 'job_metadata.txt').exists():
                        profile_dirs.append(run_dir)
                    # Also check for subdirectories (in case there's a run_* wrapper)
                    for sub_dir in run_dir.iterdir():
                        if sub_dir.is_dir() and (sub_dir / 'job_metadata.txt').exists():
                            profile_dirs.append(sub_dir)
    
    # Also check for direct run directories (run_* pattern)
    for run_dir in results_path.iterdir():
        if run_dir.is_dir():
            # Check if this directory or subdirectories contain results
            if (run_dir / 'job_metadata.txt').exists():
                profile_dirs.append(run_dir)
            # Check for node subdirectories within run directories
            for node_dir in ['1node', '2node', '4node']:
                node_path = run_dir / node_dir
                if node_path.exists():
                    for sub_dir in node_path.iterdir():
                        if sub_dir.is_dir() and (sub_dir / 'job_metadata.txt').exists():
                            profile_dirs.append(sub_dir)
    
    # Remove duplicates
    profile_dirs = list(set(profile_dirs))
    
    if not profile_dirs:
        print("Warning: No profiling run directories found.")
        print("Expected structure: results/profiling/{1node,2node,4node}/run_*/")
        sys.exit(1)
    
    print(f"Found {len(profile_dirs)} profiling run(s):")
    for dir_path in profile_dirs:
        print(f"  - {dir_path}")
    print("")
    
    # Analyze each profiling run
    all_results = []
    for profile_dir in profile_dirs:
        print(f"Analyzing: {profile_dir.name}")
        result = analyze_profile_directory(profile_dir)
        all_results.append(result)
    
    print("")
    
    # Generate summary
    summary = generate_bottleneck_summary(all_results)
    
    # Save JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        'summary': summary,
        'detailed_results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"Analysis saved to: {output_path}")
    
    # Generate text summary
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Phase 3: Bottleneck Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {summary['timestamp']}\n\n")
        
        f.write("Profiling Scenarios:\n")
        f.write("-" * 60 + "\n")
        for scenario, result in summary['scenarios'].items():
            f.write(f"\n{scenario}:\n")
            f.write(f"  Directory: {result['directory']}\n")
            f.write(f"  Nodes: {result['metadata'].get('Nodes', 'N/A')}\n")
            f.write(f"  Nsight Systems: {'Yes' if result['nsys_available'] else 'No'}\n")
            
            bottlenecks = result['bottlenecks']
            f.write(f"\n  Potential Bottlenecks:\n")
            f.write(f"    - Compute-bound: {bottlenecks.get('compute_bound', False)}\n")
            f.write(f"    - Memory-bound: {bottlenecks.get('memory_bound', False)}\n")
            f.write(f"    - I/O-bound: {bottlenecks.get('io_bound', False)}\n")
            f.write(f"    - Communication-bound: {bottlenecks.get('communication_bound', False)}\n")
            f.write(f"    - Data-loader-bound: {bottlenecks.get('data_loader_bound', False)}\n")
            
            if bottlenecks.get('notes'):
                f.write(f"\n  Notes:\n")
                for note in bottlenecks['notes']:
                    f.write(f"    - {note}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Key Findings:\n")
        f.write("=" * 60 + "\n")
        for finding in summary['key_findings']:
            f.write(f"- {finding}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Next Steps:\n")
        f.write("=" * 60 + "\n")
        f.write("1. Open Nsight Systems reports (.nsys-rep files) in Nsight Systems GUI\n")
        f.write("2. Examine timeline for:\n")
        f.write("   - GPU utilization gaps (idle time)\n")
        f.write("   - NCCL all-reduce duration and frequency\n")
        f.write("   - Data loading overlaps with compute\n")
        f.write("   - Memory transfer overhead\n")
        f.write("3. Quantify time spent in:\n")
        f.write("   - Compute (kernels)\n")
        f.write("   - Communication (NCCL)\n")
        f.write("   - Data loading\n")
        f.write("   - I/O (checkpoints, logging)\n")
        f.write("4. Create detailed bottleneck analysis document\n")
    
    print(f"Summary saved to: {summary_path}")
    print("")
    print("==========================================")
    print("Analysis Complete!")
    print("==========================================")
    print("")
    print("Next steps:")
    print("1. Review the summary: cat {}".format(summary_path))
    print("2. Open .nsys-rep files in Nsight Systems GUI for detailed timeline analysis")
    print("3. Create detailed bottleneck analysis document")


if __name__ == '__main__':
    main()

