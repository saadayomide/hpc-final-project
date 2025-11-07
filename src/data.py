"""
Data loading and preprocessing utilities for DCRNN
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional


class DCRNNDataset(Dataset):
    """Dataset class for DCRNN - generates synthetic spatio-temporal data"""
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 num_nodes: int = 50, seq_len: int = 12, pred_len: int = 1,
                 num_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.split = split
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Generate synthetic data for now
        # In production, load from actual data files
        if num_samples is None:
            num_samples = {
                'train': 1000,
                'val': 200,
                'test': 200
            }.get(split, 200)
        
        self.num_samples = num_samples
        
        # Generate synthetic adjacency matrix
        self.adj_matrix = self._generate_adjacency_matrix()
        
        # Generate support matrices (normalized adjacency)
        self.supports = self._generate_supports()
        
    def _generate_adjacency_matrix(self) -> np.ndarray:
        """Generate a synthetic adjacency matrix"""
        # Create a random graph
        adj = np.random.rand(self.num_nodes, self.num_nodes)
        # Make it symmetric
        adj = (adj + adj.T) / 2
        # Threshold to create sparse graph
        adj[adj < 0.3] = 0
        # Set diagonal to 0
        np.fill_diagonal(adj, 0)
        return adj
    
    def _generate_supports(self) -> List[np.ndarray]:
        """Generate normalized support matrices for diffusion convolution"""
        supports = []
        
        # Support 0: Identity
        supports.append(np.eye(self.num_nodes))
        
        # Support 1: Normalized adjacency
        adj = self.adj_matrix.copy()
        # Row normalization
        rowsum = adj.sum(axis=1) + 1e-8
        adj_norm = adj / rowsum[:, np.newaxis]
        supports.append(adj_norm)
        
        return supports
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            x: Input sequence [seq_len, num_nodes, input_dim]
            y: Target [pred_len, num_nodes, output_dim]
            supports: List of support matrices [num_supports, num_nodes, num_nodes]
        """
        # Generate synthetic spatio-temporal data
        # Simulate time series with spatial dependencies
        seq_len = self.seq_len
        num_nodes = self.num_nodes
        
        # Generate correlated time series
        base_signal = np.sin(np.linspace(0, 4 * np.pi * idx / self.num_samples, seq_len + self.pred_len))
        base_signal = base_signal[:, np.newaxis]  # [seq_len + pred_len, 1]
        
        # Add spatial correlation through adjacency
        x_data = np.zeros((seq_len + self.pred_len, num_nodes, 1))
        for t in range(seq_len + self.pred_len):
            # Propagate signal through graph
            signal = base_signal[t, 0] * np.ones(num_nodes)
            # Add noise
            noise = np.random.randn(num_nodes) * 0.1
            # Apply adjacency for spatial correlation
            x_data[t, :, 0] = signal + noise
        
        # Split into input and target
        x = torch.FloatTensor(x_data[:seq_len])  # [seq_len, num_nodes, input_dim]
        y = torch.FloatTensor(x_data[seq_len:seq_len + self.pred_len])  # [pred_len, num_nodes, output_dim]
        
        # Convert supports to tensors
        supports = [torch.FloatTensor(s) for s in self.supports]
        
        return x, y, supports


def get_dataloader(data_dir: str, split: str = 'train', batch_size: int = 32, 
                   shuffle: bool = True, num_workers: int = 0, 
                   num_nodes: int = 50, seq_len: int = 12, pred_len: int = 1,
                   num_samples: Optional[int] = None, sampler=None) -> Tuple[DataLoader, List[torch.Tensor]]:
    """
    Create a DataLoader for the specified split
    
    Args:
        sampler: Optional DistributedSampler for DDP training
    
    Returns:
        dataloader: DataLoader instance
        supports: List of support matrices (shared across all samples)
    """
    dataset = DCRNNDataset(data_dir, split=split, num_nodes=num_nodes, 
                          seq_len=seq_len, pred_len=pred_len, num_samples=num_samples)
    
    # Get supports from first sample (they're the same for all samples)
    _, _, supports = dataset[0]
    
    # Use sampler if provided (for DDP), otherwise use shuffle
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return dataloader, supports
