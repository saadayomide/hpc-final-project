#!/usr/bin/env python3
"""
DCRNN Training Script - Multi-node DDP support
"""

import argparse
import os
import sys
import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path

# Add project root to path for both script and module execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try relative imports first (for module execution), fall back to absolute
try:
    from src.model.dcrnn import DCRNN
    from src.data import get_dataloader
    from src.utils.metrics import mae, mse, rmse
    from src.utils.monitoring import GPUMonitor, CPUMonitor, get_gpu_info
except ImportError:
    # Fallback for direct script execution
    from model.dcrnn import DCRNN
    from data import get_dataloader
    from utils.metrics import mae, mse, rmse
    from utils.monitoring import GPUMonitor, CPUMonitor, get_gpu_info


def setup_ddp(rank, world_size, master_addr='localhost', master_port='29500'):
    """Initialize distributed process group with robust error handling"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Determine backend
    if torch.cuda.is_available():
        backend = os.environ.get("BACKEND", "nccl")
        # Verify CUDA device is accessible
        try:
            device_count = torch.cuda.device_count()
            if device_count == 0:
                print(f"Warning: CUDA available but no devices found, falling back to gloo")
                backend = "gloo"
        except Exception as e:
            print(f"Warning: CUDA error ({e}), falling back to gloo")
            backend = "gloo"
    else:
        backend = "gloo"
    
    print(f"Rank {rank}: Initializing DDP with backend={backend}, master={master_addr}:{master_port}")
    
    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        print(f"Rank {rank}: DDP initialized successfully")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize DDP: {e}")
        raise
    
    # Set device for this process (only for CUDA)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank}: Using GPU {local_rank} ({torch.cuda.get_device_name(local_rank)})")


def cleanup_ddp():
    """Cleanup distributed process group"""
    torch.distributed.destroy_process_group()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, output_dir, is_best=False, rank=0):
    """Save model checkpoint (only on rank 0)"""
    if rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)


def train_epoch(model, train_loader, optimizer, criterion, device, precision, 
                scaler=None, supports=None, rank=0, world_size=1, epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    # Use DistributedSampler if DDP
    if world_size > 1 and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)  # Set epoch for shuffling
    
    epoch_start = time.time()
    data_load_time = 0.0
    compute_time = 0.0
    
    for batch_idx, (x, y, _) in enumerate(train_loader):
        batch_start = time.time()
        
        # Move to device
        x = x.to(device, non_blocking=True)  # [batch, seq_len, num_nodes, input_dim]
        y = y.to(device, non_blocking=True)  # [batch, pred_len, num_nodes, output_dim]
        
        # Convert supports to device
        supports_device = [s.to(device) for s in supports]
        
        data_load_time += time.time() - batch_start
        compute_start = time.time()
        
        # Forward pass
        optimizer.zero_grad()
        
        if precision == 'bf16':
            with autocast(dtype=torch.bfloat16):
                pred = model(x, supports_device)
                pred = pred.unsqueeze(1)  # [batch, 1, num_nodes, output_dim]
                loss = criterion(pred, y)
        elif precision == 'fp16':
            with autocast():
                pred = model(x, supports_device)
                pred = pred.unsqueeze(1)
                loss = criterion(pred, y)
        else:  # fp32
            pred = model(x, supports_device)
            pred = pred.unsqueeze(1)
            loss = criterion(pred, y)
        
        # Backward pass
        if precision in ['bf16', 'fp16']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        compute_time += time.time() - compute_start
        
        # Average loss across all processes for DDP
        if world_size > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / world_size
        
        total_loss += loss.item() * x.size(0)
        num_samples += x.size(0)
    
    return total_loss / num_samples, data_load_time, compute_time


def validate(model, val_loader, criterion, device, precision, supports=None, world_size=1):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            supports_device = [s.to(device) for s in supports]
            
            if precision == 'bf16':
                with autocast(dtype=torch.bfloat16):
                    pred = model(x, supports_device)
                    pred = pred.unsqueeze(1)
            elif precision == 'fp16':
                with autocast():
                    pred = model(x, supports_device)
                    pred = pred.unsqueeze(1)
            else:
                pred = model(x, supports_device)
                pred = pred.unsqueeze(1)
            
            loss = criterion(pred, y)
            mae_val = mae(pred, y)
            rmse_val = rmse(pred, y)
            
            # Average across all processes for DDP
            if world_size > 1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(mae_val, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(rmse_val, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size
                mae_val = mae_val / world_size
                rmse_val = rmse_val / world_size
            
            total_loss += loss.item() * x.size(0)
            total_mae += mae_val.item() * x.size(0)
            total_rmse += rmse_val.item() * x.size(0)
            num_samples += x.size(0)
    
    return total_loss / num_samples, total_mae / num_samples, total_rmse / num_samples


def main():
    parser = argparse.ArgumentParser(description='DCRNN Training - Multi-node DDP')
    
    # DDP arguments
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--rank', type=int, default=-1,
                        help='Global rank for distributed training')
    parser.add_argument('--world-size', type=int, default=-1,
                        help='World size for distributed training')
    parser.add_argument('--master-addr', type=str, default='localhost',
                        help='Master address for distributed training')
    parser.add_argument('--master-port', type=str, default='29500',
                        help='Master port for distributed training')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='./data',
                        help='Directory containing dataset')
    parser.add_argument('--num-nodes', type=int, default=50,
                        help='Number of nodes in graph')
    parser.add_argument('--seq-len', type=int, default=12,
                        help='Input sequence length')
    parser.add_argument('--pred-len', type=int, default=1,
                        help='Prediction length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training settings
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
                        help='Training precision')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--results', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Monitoring
    parser.add_argument('--monitor-gpu', action='store_true',
                        help='Monitor GPU utilization')
    parser.add_argument('--monitor-cpu', action='store_true',
                        help='Monitor CPU utilization')
    
    args = parser.parse_args()
    
    # Determine if DDP is being used
    use_ddp = False
    if args.local_rank != -1 or 'RANK' in os.environ:
        use_ddp = True
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if args.rank == -1:
            args.rank = int(os.environ.get('RANK', 0))
        if args.world_size == -1:
            args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        setup_ddp(args.rank, args.world_size, args.master_addr, args.master_port)
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed (all processes use same seed for reproducibility)
    set_seed(args.seed)
    
    # Create output directory (only rank 0)
    if args.rank == 0:
        os.makedirs(args.results, exist_ok=True)
    
    # Wait for rank 0 to create directory
    if use_ddp:
        torch.distributed.barrier()
    
    # Device setup
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if args.rank == 0:
        print(f"Device: {device}")
        print(f"Using DDP: {use_ddp}")
        if use_ddp:
            print(f"Rank: {args.rank}, World Size: {args.world_size}, Local Rank: {args.local_rank}")
        if num_gpus > 0:
            print(f"Number of GPUs: {num_gpus}")
            gpu_info = get_gpu_info()
            print(f"GPU Info: {gpu_info}")
    
    # Start monitoring (only rank 0)
    gpu_monitor = None
    cpu_monitor = None
    if args.rank == 0:
        if args.monitor_gpu and num_gpus > 0:
            gpu_log = os.path.join(args.results, 'gpu_monitor.csv')
            gpu_monitor = GPUMonitor(gpu_log)
            gpu_monitor.start()
            print(f"GPU monitoring started: {gpu_log}")
        
        if args.monitor_cpu:
            cpu_log = os.path.join(args.results, 'cpu_monitor.csv')
            cpu_monitor = CPUMonitor(cpu_log)
            cpu_monitor.start()
            print(f"CPU monitoring started: {cpu_log}")
    
    # Data loaders
    if args.rank == 0:
        print("Loading data...")
    
    # Create datasets first to get their size
    from src.data import DCRNNDataset
    train_dataset = DCRNNDataset(
        args.data, split='train', num_nodes=args.num_nodes,
        seq_len=args.seq_len, pred_len=args.pred_len
    )
    val_dataset = DCRNNDataset(
        args.data, split='val', num_nodes=args.num_nodes,
        seq_len=args.seq_len, pred_len=args.pred_len, num_samples=200
    )
    
    # Get supports from first sample
    _, _, train_supports = train_dataset[0]
    _, _, val_supports = val_dataset[0]
    
    # Create DistributedSampler if using DDP
    train_sampler = None
    val_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    if args.rank == 0:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Effective batch size: {args.batch_size * args.world_size}")
    
    # Model
    model = DCRNN(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        num_layers=args.num_layers,
        output_dim=1,
        dropout=args.dropout
    )
    
    # Wrap with DDP if using distributed training
    if use_ddp:
        model = model.to(device)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        if args.rank == 0:
            print(f"Using DDP with {args.world_size} processes")
    else:
        # Single node multi-GPU
        if num_gpus > 1:
            model = DataParallel(model)
            if args.rank == 0:
                print(f"Using {num_gpus} GPUs with DataParallel")
        model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Mixed precision scaler
    scaler = None
    if args.precision in ['bf16', 'fp16']:
        scaler = GradScaler()
        if args.rank == 0:
            print(f"Using {args.precision} precision with GradScaler")
    
    # Metrics CSV (only rank 0 writes)
    metrics_file = os.path.join(args.results, 'metrics.csv')
    if args.rank == 0:
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_rmse',
                            'epoch_time_sec', 'throughput_samples_per_sec', 'data_load_time_sec',
                            'compute_time_sec', 'gpu_util_avg', 'cpu_util_avg'])
    
    # Training loop
    if args.rank == 0:
        print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_loss, data_load_time, compute_time = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.precision, scaler, train_supports, args.rank, args.world_size, epoch
        )
        
        # Validation
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion, device, args.precision, val_supports, args.world_size
        )
        
        epoch_time = time.time() - epoch_start
        
        # Calculate throughput
        num_train_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * args.batch_size
        throughput = num_train_samples / epoch_time if epoch_time > 0 else 0
        
        # Get GPU/CPU utilization (simplified - average from monitoring if available)
        gpu_util_avg = 0.0
        cpu_util_avg = 0.0
        if args.rank == 0:
            if gpu_monitor and os.path.exists(gpu_monitor.log_file):
                try:
                    with open(gpu_monitor.log_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            gpu_utils = []
                            for line in lines[1:]:
                                parts = line.strip().split(',')
                                if len(parts) > 2:
                                    try:
                                        gpu_utils.append(float(parts[2]))
                                    except:
                                        pass
                            gpu_util_avg = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
                except:
                    pass
            
            if cpu_monitor and os.path.exists(cpu_monitor.log_file):
                try:
                    with open(cpu_monitor.log_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            cpu_utils = []
                            for line in lines[1:]:
                                parts = line.strip().split(',')
                                if len(parts) > 1:
                                    try:
                                        cpu_utils.append(float(parts[1]))
                                    except:
                                        pass
                            cpu_util_avg = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
                except:
                    pass
        
        # Log metrics (only rank 0)
        if args.rank == 0:
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MAE: {val_mae:.6f} | "
                  f"Val RMSE: {val_rmse:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Throughput: {throughput:.2f} samples/s | "
                  f"Data Load: {data_load_time:.2f}s | "
                  f"Compute: {compute_time:.2f}s")
            
            # Write to CSV
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_mae, val_rmse,
                               epoch_time, throughput, data_load_time, compute_time,
                               gpu_util_avg, cpu_util_avg])
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, val_loss, args.results, is_best, args.rank)
            if args.rank == 0:
                print(f"Checkpoint saved at epoch {epoch}")
    
    # Stop monitoring
    if args.rank == 0:
        if gpu_monitor:
            gpu_monitor.stop()
        if cpu_monitor:
            cpu_monitor.stop()
        
        print(f"\nTraining complete! Results saved to {args.results}")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
