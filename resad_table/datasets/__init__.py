"""
ResAD Tabular Datasets Package

This package provides data loading and preprocessing utilities for tabular anomaly detection:
- ADBench dataset support
- Custom tabular dataset handling
- Cross-domain evaluation utilities
"""

from .tabular_loader import (
    TabularAnomalyDataset,
    ADBenchDataLoader,
    create_data_loader,
    get_dataset_info
)

__all__ = [
    'TabularAnomalyDataset',
    'ADBenchDataLoader', 
    'create_data_loader',
    'get_dataset_info',
]