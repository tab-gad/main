import warnings
import os
import sys
import logging

# Suppress all warnings including hyperopt deprecation warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress hyperopt specific warnings
logging.getLogger('hyperopt').setLevel(logging.ERROR)

# Redirect stderr to suppress pkg_resources warnings during imports
from io import StringIO
original_stderr = sys.stderr

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Temporarily suppress stderr during TabPFN imports
sys.stderr = StringIO()
try:
    from tabpfn_extensions import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding
finally:
    sys.stderr = original_stderr


class TabPFNFeatureExtractor(nn.Module):
    """
    Pre-trained feature extractor using TabPFN for tabular data.
    
    This module extracts embeddings from tabular data using TabPFN,
    which is a transformer-based model pre-trained on synthetic tabular data.
    """
    
    def __init__(self, n_estimators=1, n_fold=5, use_scaler=True):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_fold = n_fold
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.is_fitted = False
        
        # Suppress warnings during TabPFN initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Initialize TabPFN classifier and embedding extractor
            self.tabpfn_clf = TabPFNClassifier(n_estimators=n_estimators)
            self.embedder = TabPFNEmbedding(tabpfn_clf=self.tabpfn_clf, n_fold=n_fold)
        
        # For cross-domain: separate TabPFN for different feature dimensions
        self.cross_domain_extractors = {}  # {feature_dim: (tabpfn_clf, embedder, scaler)}
    
    def fit(self, X_train, y_train=None):
        """
        Fit the feature extractor on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels (optional, can use dummy labels)
        """
        X_train = np.array(X_train)
        
        if self.use_scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        # Use dummy labels if not provided (unsupervised setting)
        if y_train is None:
            y_train = np.zeros(len(X_train))
        else:
            y_train = np.array(y_train)
        
        # Store training data for embedding extraction
        self.X_train_ref = X_train
        self.y_train_ref = y_train
        
        # For n_fold=0, need to fit the embedder
        if self.n_fold == 0:
            self.embedder.fit(X_train, y_train)
        
        self.is_fitted = True
    
    def _get_or_create_extractor_for_dimension(self, feature_dim, X_sample):
        """Get or create a TabPFN extractor for the given feature dimension."""
        if feature_dim not in self.cross_domain_extractors:
            # Suppress warnings during TabPFN initialization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create new extractor for this dimension
                tabpfn_clf = TabPFNClassifier(n_estimators=self.n_estimators)
                embedder = TabPFNEmbedding(tabpfn_clf=tabpfn_clf, n_fold=self.n_fold)
                scaler = StandardScaler() if self.use_scaler else None
            
            # Fit the new extractor on a sample of the data
            if scaler is not None:
                X_normalized = scaler.fit_transform(X_sample)
            else:
                X_normalized = X_sample
                
            # Create dummy labels for unsupervised case
            y_dummy = np.zeros(len(X_sample))
            
            # Note: We don't actually fit TabPFN here as it's pre-trained
            # We just store the configuration
            
            self.cross_domain_extractors[feature_dim] = (tabpfn_clf, embedder, scaler)
            
        return self.cross_domain_extractors[feature_dim]

    def forward(self, X, data_source="test"):
        """
        Extract TabPFN embeddings for input data.
        
        Args:
            X: Input tabular data
            data_source: "train" or "test"
        
        Returns:
            Embeddings tensor
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before use")
        
        X = np.array(X)
        feature_dim = X.shape[1]
        
        # Check if this is the same dimension as training data
        if feature_dim == self.X_train_ref.shape[1]:
            # Same domain - use the original extractor
            if self.use_scaler:
                X_processed = self.scaler.transform(X)
            else:
                X_processed = X
                
            # Extract embeddings using the fitted TabPFN
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = self.embedder.get_embeddings(
                    self.X_train_ref, 
                    self.y_train_ref, 
                    X_processed, 
                    data_source=data_source
                )
        else:
            # Cross-domain - use separate extractor for this feature dimension
            tabpfn_clf, embedder, scaler = self._get_or_create_extractor_for_dimension(feature_dim, X)
            
            if scaler is not None:
                X_processed = scaler.transform(X)
            else:
                X_processed = X
            
            # For cross-domain, we create a simple dummy reference
            X_ref = X_processed[:min(10, len(X_processed))]  # Use first few samples as reference
            y_ref = np.zeros(len(X_ref))
            
            # Extract embeddings using TabPFN with the new dimension data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = embedder.get_embeddings(
                    X_ref,
                    y_ref,
                    X_processed, 
                    data_source="test"  # Always use "test" for cross-domain
                )
        
        # Convert to tensor and handle shape
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        # Handle 3D embeddings (ensemble case)
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=0)  # Average across ensemble
        
        return embeddings
    
    def get_embedding_dim(self):
        """Get the dimensionality of the extracted embeddings."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted to determine embedding dimension")
        
        # Get a sample embedding to determine dimension
        dummy_input = self.X_train_ref[:1]
        sample_embedding = self.forward(dummy_input, data_source="train")
        return sample_embedding.shape[-1]


class MultiScaleTabPFNExtractor(nn.Module):
    """
    Multi-scale TabPFN feature extractor for different levels of abstraction.
    
    This creates multiple embedding views of the same data using different
    configurations to capture various aspects of the tabular data.
    """
    
    def __init__(self, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        self.extractors = nn.ModuleList()
        
        # Different configurations for multi-scale extraction
        configs = [
            {"n_estimators": 1, "n_fold": 0},  # Fast, single model
            {"n_estimators": 1, "n_fold": 3},  # CV-based
            {"n_estimators": 1, "n_fold": 5},  # More robust CV
        ]
        
        for i in range(n_levels):
            config = configs[i % len(configs)]
            extractor = TabPFNFeatureExtractor(**config)
            self.extractors.append(extractor)
    
    def fit(self, X_train, y_train=None):
        """Fit all extractors."""
        for extractor in self.extractors:
            extractor.fit(X_train, y_train)
    
    def forward(self, X, data_source="test"):
        """Extract multi-scale embeddings."""
        embeddings = []
        for extractor in self.extractors:
            emb = extractor(X, data_source=data_source)
            embeddings.append(emb)
        return embeddings
    
    def get_embedding_dims(self):
        """Get embedding dimensions for each scale."""
        return [extractor.get_embedding_dim() for extractor in self.extractors]