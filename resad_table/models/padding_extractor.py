import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class PaddingFeatureExtractor(nn.Module):
    """
    Padding-based feature extractor as an alternative to TabPFN.
    
    This module pads input features to match TabPFN's 192-dimensional output
    for fair comparison. Uses zero padding, mean padding, or learned padding.
    """
    
    def __init__(self, target_dim=192, padding_mode='zero', use_scaler=True):
        super().__init__()
        self.target_dim = target_dim
        self.padding_mode = padding_mode  # 'zero', 'mean', 'learned'
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.is_fitted = False
        
        # For learned padding
        self.learned_padding = None
        self.feature_dim = None
        
    def fit(self, X_train, y_train=None):
        """
        Fit the feature extractor on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels (optional, unused for padding)
        """
        X_train = np.array(X_train)
        self.feature_dim = X_train.shape[1]
        
        if self.use_scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        # Store training data for reference
        self.X_train_ref = X_train
        
        # Initialize learned padding if needed
        if self.padding_mode == 'learned' and self.feature_dim < self.target_dim:
            padding_dim = self.target_dim - self.feature_dim
            # Initialize learned padding with small random values
            self.learned_padding = nn.Parameter(
                torch.randn(padding_dim) * 0.1
            )
        
        self.is_fitted = True
    
    def _pad_features(self, X):
        """Pad features to target dimension."""
        X = torch.tensor(X, dtype=torch.float32)
        
        if X.shape[1] >= self.target_dim:
            # If input is larger than target, truncate
            return X[:, :self.target_dim]
        
        # Calculate padding size
        padding_size = self.target_dim - X.shape[1]
        
        if self.padding_mode == 'zero':
            # Zero padding
            padding = torch.zeros(X.shape[0], padding_size)
            
        elif self.padding_mode == 'mean':
            # Mean padding - use feature means from training data
            if hasattr(self, 'X_train_ref'):
                feature_means = torch.tensor(np.mean(self.X_train_ref, axis=0), dtype=torch.float32)
                # Repeat the means cyclically if needed
                padding_values = feature_means[:padding_size % len(feature_means)]
                if padding_size > len(feature_means):
                    # Repeat pattern if padding is larger than number of features
                    repeats = padding_size // len(feature_means) + 1
                    padding_values = feature_means.repeat(repeats)[:padding_size]
                else:
                    padding_values = feature_means[:padding_size]
                padding = padding_values.unsqueeze(0).repeat(X.shape[0], 1)
            else:
                padding = torch.zeros(X.shape[0], padding_size)
                
        elif self.padding_mode == 'learned':
            # Learned padding
            if self.learned_padding is not None:
                padding = self.learned_padding.unsqueeze(0).repeat(X.shape[0], 1)
            else:
                padding = torch.zeros(X.shape[0], padding_size)
        
        else:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")
        
        # Concatenate original features with padding
        padded_X = torch.cat([X, padding], dim=1)
        return padded_X
    
    def forward(self, X, data_source="test"):
        """
        Extract padded features for input data.
        
        Args:
            X: Input tabular data
            data_source: "train" or "test" (unused for padding)
        
        Returns:
            Padded features tensor
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before use")
        
        X = np.array(X)
        
        # Handle cross-domain case where input has different dimensions
        if X.shape[1] != self.feature_dim:
            # Create a new scaler for different input dimensions
            if self.use_scaler:
                from sklearn.preprocessing import StandardScaler
                cross_scaler = StandardScaler()
                X = cross_scaler.fit_transform(X)
            # Don't apply the original scaler for different dimensions
        else:
            # Same dimension - apply original scaling
            if self.use_scaler:
                X = self.scaler.transform(X)
        
        # Pad features to target dimension
        padded_features = self._pad_features(X)
        
        return padded_features
    
    def get_embedding_dim(self):
        """Get the dimensionality of the extracted embeddings."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted to determine embedding dimension")
        return self.target_dim


class MultiScalePaddingExtractor(nn.Module):
    """
    Multi-scale padding feature extractor for different levels of abstraction.
    
    Creates multiple padding views using different strategies.
    """
    
    def __init__(self, target_dim=192, n_levels=3):
        super().__init__()
        self.target_dim = target_dim
        self.n_levels = n_levels
        self.extractors = nn.ModuleList()
        
        # Different padding strategies for multi-scale
        padding_modes = ['zero', 'mean', 'learned']
        
        for i in range(n_levels):
            mode = padding_modes[i % len(padding_modes)]
            extractor = PaddingFeatureExtractor(
                target_dim=target_dim, 
                padding_mode=mode, 
                use_scaler=True
            )
            self.extractors.append(extractor)
    
    def fit(self, X_train, y_train=None):
        """Fit all extractors."""
        for extractor in self.extractors:
            extractor.fit(X_train, y_train)
    
    def forward(self, X, data_source="test"):
        """Extract multi-scale padded features."""
        embeddings = []
        for extractor in self.extractors:
            emb = extractor(X, data_source=data_source)
            embeddings.append(emb)
        return embeddings
    
    def get_embedding_dims(self):
        """Get embedding dimensions for each scale."""
        return [extractor.get_embedding_dim() for extractor in self.extractors]