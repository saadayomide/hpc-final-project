"""
Evaluation metrics for DCRNN
"""

import torch
import numpy as np

def mae(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))

def mse(pred, target):
    """Mean Squared Error"""
    return torch.mean((pred - target) ** 2)

def rmse(pred, target):
    """Root Mean Squared Error"""
    return torch.sqrt(mse(pred, target))

def mape(pred, target, eps=1e-8):
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs((pred - target) / (target + eps))) * 100

