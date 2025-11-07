"""
GPU and CPU monitoring utilities
"""

import subprocess
import psutil
import torch
import time
import threading
from typing import Dict, List, Optional
import csv
import os


class GPUMonitor:
    """Monitor GPU utilization and memory"""
    
    def __init__(self, log_file: str, interval: float = 1.0):
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start GPU monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        # Clear/create log file
        with open(self.log_file, 'w') as f:
            f.write('timestamp,gpu_id,utilization_gpu,utilization_memory,memory_used_mb,memory_total_mb,temperature,power_draw_w\n')
        
        def monitor_loop():
            while self.monitoring:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', 
                         '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        timestamp = time.time()
                        lines = result.stdout.strip().split('\n')
                        
                        with open(self.log_file, 'a') as f:
                            for line in lines:
                                if line.strip():
                                    parts = line.split(', ')
                                    if len(parts) >= 7:
                                        f.write(f'{timestamp},{",".join(parts)}\n')
                    
                    time.sleep(self.interval)
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
                    time.sleep(self.interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


class CPUMonitor:
    """Monitor CPU utilization"""
    
    def __init__(self, log_file: str, interval: float = 1.0):
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start CPU monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        # Clear/create log file
        with open(self.log_file, 'w') as f:
            f.write('timestamp,cpu_percent,memory_percent,memory_used_gb,memory_total_gb\n')
        
        def monitor_loop():
            while self.monitoring:
                try:
                    timestamp = time.time()
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    
                    with open(self.log_file, 'a') as f:
                        f.write(f'{timestamp},{cpu_percent},{memory.percent},{memory.used / 1e9:.2f},{memory.total / 1e9:.2f}\n')
                    
                    time.sleep(self.interval)
                except Exception as e:
                    print(f"CPU monitoring error: {e}")
                    time.sleep(self.interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop CPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


def get_gpu_info() -> Dict:
    """Get GPU information"""
    info = {}
    if torch.cuda.is_available():
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    return info

