import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path


class TabularAnomalyDataset(Dataset):
    """
    Dataset class for tabular anomaly detection.
    
    Handles loading and preprocessing of tabular datasets with anomaly labels.
    Compatible with ADBench format and custom datasets.
    """
    
    def __init__(self, data_path, dataset_name=None, train=True, test_size=0.3, 
                 random_state=42, normalize=True):
        """
        Initialize tabular anomaly dataset.
        
        Args:
            data_path: Path to dataset file or directory
            dataset_name: Name of the dataset
            train: Whether to load training or test split
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            normalize: Whether to normalize features
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.normalize = normalize
        
        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        self._create_splits()
    
    def _load_data(self):
        """Load data from file."""
        if self.data_path.is_file():
            # Single file
            if self.data_path.suffix == '.npz':
                self._load_npz()
            elif self.data_path.suffix in ['.csv', '.parquet']:
                self._load_csv_parquet()
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        else:
            # Directory with multiple files
            self._load_from_directory()
    
    def _load_npz(self):
        """Load from NPZ file (ADBench format)."""
        data = np.load(self.data_path, allow_pickle=True)
        self.X = data['X'].astype(np.float32)
        self.y = data['y'].astype(np.int32)
        self.feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
    
    def _load_csv_parquet(self):
        """Load from CSV or Parquet file."""
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            df = pd.read_parquet(self.data_path)
        
        # Assume last column is the label
        if 'is_anomaly' in df.columns:
            label_col = 'is_anomaly'
        elif 'y' in df.columns:
            label_col = 'y'
        else:
            label_col = df.columns[-1]
        
        self.y = df[label_col].values.astype(np.int32)
        self.X = df.drop(columns=[label_col]).values.astype(np.float32)
        self.feature_names = df.drop(columns=[label_col]).columns.tolist()
    
    def _load_from_directory(self):
        """Load from directory structure."""
        # Look for common dataset files
        possible_files = [
            self.data_path / f"{self.dataset_name}.npz",
            self.data_path / f"{self.dataset_name}.csv",
            self.data_path / "data.npz",
            self.data_path / "data.csv",
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                self.data_path = file_path
                self._load_data()
                return
        
        raise FileNotFoundError(f"No suitable data file found in {self.data_path}")
    
    def _preprocess_data(self):
        """Preprocess the loaded data."""
        # Handle categorical features if any
        self.X, self.categorical_indices = self._handle_categorical_features(self.X)
        
        # Handle missing values
        self.X = self._handle_missing_values(self.X)
        
        # Store original data for normalization
        self.X_original = self.X.copy()
        
    def _handle_categorical_features(self, X):
        """Handle categorical features with label encoding."""
        categorical_indices = []
        
        if isinstance(X, np.ndarray):
            # If numpy array, assume all numeric
            return X, categorical_indices
        
        # If DataFrame, handle categorical columns
        if isinstance(X, pd.DataFrame):
            processed_X = X.copy()
            
            for i, col in enumerate(X.columns):
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_indices.append(i)
                    le = LabelEncoder()
                    processed_X[col] = le.fit_transform(X[col].astype(str))
            
            return processed_X.values.astype(np.float32), categorical_indices
        
        return X, categorical_indices
    
    def _handle_missing_values(self, X):
        """Handle missing values by replacing with median."""
        if np.isnan(X).any():
            # Replace NaN with median for each column
            for col in range(X.shape[1]):
                col_data = X[:, col]
                if np.isnan(col_data).any():
                    median_val = np.nanmedian(col_data)
                    col_data[np.isnan(col_data)] = median_val
                    X[:, col] = col_data
        return X
    
    def _create_splits(self):
        """Create train/test splits."""
        # Stratified split to maintain anomaly ratio
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        # Normalize if requested
        if self.normalize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            self.scaler = None
        
        if self.train:
            self.X_split = X_train
            self.y_split = y_train
        else:
            self.X_split = X_test
            self.y_split = y_test
    
    def __len__(self):
        return len(self.y_split)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.X_split[idx], dtype=torch.float32),
            'label': torch.tensor(self.y_split[idx], dtype=torch.long),
            'index': idx
        }
    
    def get_normal_data(self):
        """Get only normal (non-anomalous) data for training."""
        normal_indices = self.y_split == 0
        return {
            'features': torch.tensor(self.X_split[normal_indices], dtype=torch.float32),
            'labels': torch.tensor(self.y_split[normal_indices], dtype=torch.long)
        }
    
    def get_anomaly_stats(self):
        """Get statistics about anomalies in the dataset."""
        total_samples = len(self.y_split)
        anomaly_samples = np.sum(self.y_split == 1)
        normal_samples = total_samples - anomaly_samples
        
        return {
            'total_samples': total_samples,
            'normal_samples': normal_samples,
            'anomaly_samples': anomaly_samples,
            'anomaly_ratio': anomaly_samples / total_samples,
            'feature_dim': self.X_split.shape[1]
        }


class ADBenchDataLoader:
    """
    Data loader for ADBench datasets.
    
    Provides easy access to multiple datasets from the ADBench benchmark
    with consistent preprocessing and splitting.
    """
    
    # Common ADBench datasets
    DATASETS = {
        'breastw': '4_breastw',
        'pima': '29_Pima', 
        'wdbc': '43_WDBC',
        'wine': '45_wine',
        'cardio': '6_cardio',
        'glass': '14_glass',
        'hepatitis': '15_Hepatitis',
        'cover': '10_cover'
    }
    
    def __init__(self, data_root, normalize=True):
        """
        Initialize ADBench data loader.
        
        Args:
            data_root: Root directory containing ADBench datasets
            normalize: Whether to normalize features
        """
        self.data_root = Path(data_root)
        self.normalize = normalize
    
    def load_dataset(self, dataset_name, train=True, test_size=0.3, random_state=42):
        """
        Load a specific ADBench dataset.
        
        Args:
            dataset_name: Name of the dataset (key in DATASETS)
            train: Whether to load training or test split
            test_size: Proportion for testing
            random_state: Random seed
        
        Returns:
            TabularAnomalyDataset instance
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.DATASETS.keys())}")
        
        dataset_file = self.DATASETS[dataset_name]
        dataset_path = self.data_root / f"{dataset_file}.npz"
        
        if not dataset_path.exists():
            # Try alternative paths
            alt_paths = [
                self.data_root / "Classical" / f"{dataset_file}.npz",
                self.data_root / f"{dataset_file}.csv"
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    dataset_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        return TabularAnomalyDataset(
            dataset_path,
            dataset_name=dataset_name,
            train=train,
            test_size=test_size,
            random_state=random_state,
            normalize=self.normalize
        )
    
    def load_multiple_datasets(self, dataset_names, **kwargs):
        """Load multiple datasets."""
        datasets = {}
        for name in dataset_names:
            datasets[name] = self.load_dataset(name, **kwargs)
        return datasets
    
    def combine_datasets(self, datasets_dict):
        """
        Combine multiple datasets into a single dataset.
        
        Args:
            datasets_dict: Dictionary of TabularAnomalyDataset instances
            
        Returns:
            Combined TabularAnomalyDataset
        """
        if not datasets_dict:
            raise ValueError("No datasets to combine")
        
        if len(datasets_dict) == 1:
            return list(datasets_dict.values())[0]
        
        # Collect all data
        all_X = []
        all_y = []
        all_features = []
        
        for name, dataset in datasets_dict.items():
            all_X.append(dataset.X)
            all_y.append(dataset.y)
            if hasattr(dataset, 'feature_names'):
                all_features.extend([f"{name}_{feat}" for feat in dataset.feature_names])
            else:
                all_features.extend([f"{name}_feat_{i}" for i in range(dataset.X.shape[1])])
        
        # Stack data
        combined_X = np.vstack(all_X)
        combined_y = np.hstack(all_y)
        
        # Create combined dataset
        combined_dataset = TabularAnomalyDataset(
            data_source="combined",
            X=combined_X,
            y=combined_y,
            feature_names=all_features[:combined_X.shape[1]],  # Take first n features
            normalize=self.normalize
        )
        
        return combined_dataset

    def get_cross_domain_splits(self, source_datasets, target_dataset, **kwargs):
        """
        Get cross-domain splits for domain adaptation experiments.
        
        Args:
            source_datasets: List of source dataset names
            target_dataset: Target dataset name
        
        Returns:
            Dictionary with source and target datasets
        """
        source_data_dict = self.load_multiple_datasets(source_datasets, train=True, **kwargs)
        source_combined = self.combine_datasets(source_data_dict)
        
        target_train = self.load_dataset(target_dataset, train=True, **kwargs)
        target_test = self.load_dataset(target_dataset, train=False, **kwargs)
        
        return {
            'source': source_combined,
            'source_individual': source_data_dict,
            'target_train': target_train,
            'target_test': target_test
        }


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a PyTorch DataLoader from TabularAnomalyDataset.
    
    Args:
        dataset: TabularAnomalyDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_dataset_info(dataset):
    """Get comprehensive information about a dataset."""
    stats = dataset.get_anomaly_stats()
    
    info = {
        'dataset_name': dataset.dataset_name,
        'split': 'train' if dataset.train else 'test',
        'shape': (stats['total_samples'], stats['feature_dim']),
        'anomaly_ratio': f"{stats['anomaly_ratio']:.3f}",
        'normal_samples': stats['normal_samples'],
        'anomaly_samples': stats['anomaly_samples'],
        'feature_names': dataset.feature_names[:5] + ['...'] if len(dataset.feature_names) > 5 else dataset.feature_names,
        'normalized': dataset.normalize,
        'categorical_features': len(dataset.categorical_indices) if hasattr(dataset, 'categorical_indices') else 0
    }
    
    return info