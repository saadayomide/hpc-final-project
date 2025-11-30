"""
Data loading and preprocessing utilities for DCRNN Traffic Prediction

Supports:
- Loading preprocessed METR-LA data
- Synthetic data generation for testing
- Distributed data loading with DistributedSampler
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


class TrafficDataset(Dataset):
    """
    Dataset class for traffic prediction
    
    Loads preprocessed data from .npz files or falls back to synthetic generation.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 num_nodes: int = 207, seq_len: int = 12, pred_len: int = 1,
                 num_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Try to load preprocessed data
        processed_dir = self.data_dir / 'processed'
        data_file = processed_dir / f'{split}.npz'
        adj_file = processed_dir / 'adj_norm.npy'
        
        if data_file.exists() and adj_file.exists():
            self._load_from_files(data_file, adj_file)
        else:
            # Fall back to synthetic data generation
            self._generate_synthetic(num_samples)
        
        # Limit samples if specified
        if num_samples is not None and num_samples < len(self.X):
            self.X = self.X[:num_samples]
            self.Y = self.Y[:num_samples]
    
    def _load_from_files(self, data_file: Path, adj_file: Path):
        """Load data from preprocessed files"""
        print(f"Loading {self.split} data from {data_file}")
        
        data = np.load(data_file)
        self.X = data['X']  # [num_samples, seq_len, num_nodes, input_dim]
        self.Y = data['Y']  # [num_samples, pred_len, num_nodes, output_dim]
        
        # Update num_nodes from data
        self.num_nodes = self.X.shape[2]
        
        # Load adjacency matrix
        adj = np.load(adj_file)
        self.supports = self._create_supports(adj)
        
        print(f"  Loaded {len(self.X)} samples, {self.num_nodes} nodes")
    
    def _generate_synthetic(self, num_samples: Optional[int] = None):
        """Generate synthetic data when preprocessed data is not available"""
        print(f"Generating synthetic {self.split} data...")
        
        if num_samples is None:
            num_samples = {
                'train': 5000,
                'val': 1000,
                'test': 1000
            }.get(self.split, 1000)
        
        # Generate synthetic adjacency matrix
        adj = self._generate_adjacency_matrix()
        self.supports = self._create_supports(adj)
        
        # Generate synthetic time series data
        self.X = np.zeros((num_samples, self.seq_len, self.num_nodes, 1), dtype=np.float32)
        self.Y = np.zeros((num_samples, self.pred_len, self.num_nodes, 1), dtype=np.float32)
        
        np.random.seed(42 if self.split == 'train' else 43 if self.split == 'val' else 44)
        
        for i in range(num_samples):
            # Generate correlated time series
            base_signal = np.sin(np.linspace(0, 4 * np.pi * i / num_samples, 
                                             self.seq_len + self.pred_len))
            
            for t in range(self.seq_len + self.pred_len):
                signal = base_signal[t] * np.ones(self.num_nodes)
                noise = np.random.randn(self.num_nodes) * 0.1
                
                if t < self.seq_len:
                    self.X[i, t, :, 0] = signal + noise
                else:
                    self.Y[i, t - self.seq_len, :, 0] = signal + noise
        
        print(f"  Generated {num_samples} synthetic samples")
    
    def _generate_adjacency_matrix(self) -> np.ndarray:
        """Generate a synthetic adjacency matrix"""
        np.random.seed(42)
        adj = np.random.rand(self.num_nodes, self.num_nodes)
        adj = (adj + adj.T) / 2
        adj[adj < 0.3] = 0
        np.fill_diagonal(adj, 0)
        # Row-normalize
        rowsum = adj.sum(axis=1) + 1e-8
        return adj / rowsum[:, np.newaxis]
    
    def _create_supports(self, adj_norm: np.ndarray) -> List[np.ndarray]:
        """Create support matrices for diffusion convolution"""
        supports = [
            np.eye(self.num_nodes, dtype=np.float32),  # Identity
            adj_norm.astype(np.float32)                 # Normalized adjacency
        ]
        return supports
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            x: Input sequence [seq_len, num_nodes, input_dim]
            y: Target [pred_len, num_nodes, output_dim]
            supports: List of support matrices [num_supports, num_nodes, num_nodes]
        """
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.Y[idx])
        supports = [torch.FloatTensor(s) for s in self.supports]
        
        return x, y, supports


# Alias for backwards compatibility
DCRNNDataset = TrafficDataset


def get_dataloader(data_dir: str, split: str = 'train', batch_size: int = 32, 
                   shuffle: bool = True, num_workers: int = 0, 
                   num_nodes: int = 207, seq_len: int = 12, pred_len: int = 1,
                   num_samples: Optional[int] = None, sampler=None) -> Tuple[DataLoader, List[torch.Tensor]]:
    """
    Create a DataLoader for the specified split
    
    Args:
        data_dir: Path to data directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler is provided)
        num_workers: Number of data loading workers
        num_nodes: Number of nodes in graph
        seq_len: Input sequence length
        pred_len: Prediction length
        num_samples: Limit number of samples (for testing)
        sampler: Optional DistributedSampler for DDP training
    
    Returns:
        dataloader: DataLoader instance
        supports: List of support matrices
    """
    dataset = TrafficDataset(
        data_dir, 
        split=split, 
        num_nodes=num_nodes,
        seq_len=seq_len, 
        pred_len=pred_len, 
        num_samples=num_samples
    )
    
    # Get supports from first sample
    _, _, supports = dataset[0]
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader, supports


def get_all_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4,
                        num_nodes: int = 207, seq_len: int = 12, pred_len: int = 1,
                        train_sampler=None, val_sampler=None, test_sampler=None):
    """
    Get all dataloaders (train, val, test)
    
    Args:
        train_sampler, val_sampler, test_sampler: Optional DistributedSamplers for DDP
    
    Returns:
        train_loader, val_loader, test_loader, supports
    """
    train_loader, supports = get_dataloader(
        data_dir, 'train', batch_size, shuffle=True, 
        num_workers=num_workers, num_nodes=num_nodes,
        seq_len=seq_len, pred_len=pred_len, sampler=train_sampler
    )
    
    val_loader, _ = get_dataloader(
        data_dir, 'val', batch_size, shuffle=False,
        num_workers=num_workers, num_nodes=num_nodes,
        seq_len=seq_len, pred_len=pred_len, sampler=val_sampler
    )
    
    test_loader, _ = get_dataloader(
        data_dir, 'test', batch_size, shuffle=False,
        num_workers=num_workers, num_nodes=num_nodes,
        seq_len=seq_len, pred_len=pred_len, sampler=test_sampler
    )
    
    return train_loader, val_loader, test_loader, supports
