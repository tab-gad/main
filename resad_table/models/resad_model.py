import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm

# Import other model components
from .tabpfn_extractor import TabPFNFeatureExtractor, MultiScaleTabPFNExtractor
from .constraintor import TabularFeatureConstraintor, ResidualConstraintor, AdaptiveConstraintor
from .estimator import NormalDistributionEstimator, FlowBasedEstimator, EnsembleEstimator


class ResADTabular(nn.Module):
    """
    ResAD for Tabular Data - Complete anomaly detection framework.
    
    This class integrates the three main components:
    1. TabPFN feature extractor (pre-trained)
    2. Feature constraintor (learnable)
    3. Normal distribution estimator (learnable)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize feature extractor
        if config.use_multiscale:
            self.feature_extractor = MultiScaleTabPFNExtractor(n_levels=config.n_scales)
        else:
            self.feature_extractor = TabPFNFeatureExtractor(
                n_estimators=config.n_estimators,
                n_fold=config.n_fold,
                use_scaler=config.use_scaler
            )
        
        # Will be initialized after feature extractor is fitted
        self.constraintor = None
        self.estimator = None
        self.embedding_dim = None
        
        self.fitted = False
    
    def _initialize_modules(self):
        """Initialize constraintor and estimator after feature extractor is fitted."""
        if self.config.use_multiscale:
            embedding_dims = self.feature_extractor.get_embedding_dims()
            self.embedding_dim = sum(embedding_dims)
            
            # For multi-scale, we'll concatenate embeddings
            self.feature_fusion = nn.Linear(self.embedding_dim, self.config.final_embedding_dim)
            self.embedding_dim = self.config.final_embedding_dim
        else:
            self.embedding_dim = self.feature_extractor.get_embedding_dim()
        
        # Initialize constraintor
        if self.config.constraintor_type == 'basic':
            self.constraintor = TabularFeatureConstraintor(
                self.embedding_dim,
                dropout_rate=self.config.dropout_rate
            )
        elif self.config.constraintor_type == 'residual':
            self.constraintor = ResidualConstraintor(
                self.embedding_dim,
                num_residual_blocks=self.config.num_residual_blocks
            )
        elif self.config.constraintor_type == 'adaptive':
            self.constraintor = AdaptiveConstraintor(self.embedding_dim)
        else:
            raise ValueError(f"Unknown constraintor type: {self.config.constraintor_type}")
        
        # Initialize estimator
        if self.config.estimator_type == 'normal':
            self.estimator = NormalDistributionEstimator(self.embedding_dim)
        elif self.config.estimator_type == 'flow':
            self.estimator = FlowBasedEstimator(
                self.embedding_dim,
                num_flows=self.config.num_flows
            )
        elif self.config.estimator_type == 'ensemble':
            self.estimator = EnsembleEstimator(
                self.embedding_dim,
                num_estimators=self.config.num_ensemble_estimators
            )
        else:
            raise ValueError(f"Unknown estimator type: {self.config.estimator_type}")
        
        # Move modules to device
        if hasattr(self.config, 'device'):
            self.constraintor = self.constraintor.to(self.config.device)
            self.estimator = self.estimator.to(self.config.device)
    
    def fit_feature_extractor(self, X_train, y_train=None):
        """Fit the TabPFN feature extractor."""
        logging.info(f"Fitting TabPFN feature extractor with {len(X_train)} training samples")
        print("Fitting TabPFN feature extractor...")
        self.feature_extractor.fit(X_train, y_train)
        
        # Initialize other modules
        logging.info("Initializing constraintor and estimator modules")
        self._initialize_modules()
        
        logging.info(f"Feature extractor fitted. Embedding dimension: {self.embedding_dim}")
        print(f"Feature extractor fitted. Embedding dimension: {self.embedding_dim}")
    
    def extract_embeddings(self, X, data_source="train"):
        """Extract embeddings from input data."""
        if self.config.use_multiscale:
            multi_embeddings = self.feature_extractor(X, data_source=data_source)
            # Concatenate multi-scale embeddings
            concatenated = torch.cat(multi_embeddings, dim=1)
            # Fuse to final dimension
            embeddings = self.feature_fusion(concatenated)
        else:
            embeddings = self.feature_extractor(X, data_source=data_source)
        
        return embeddings
    
    def forward_train(self, X_normal):
        """Forward pass for training (normal data only)."""
        # Extract embeddings
        embeddings = self.extract_embeddings(X_normal, data_source="train")
        
        # Apply constraints
        constrained_embeddings = self.constraintor(embeddings)
        
        return embeddings, constrained_embeddings
    
    def forward(self, X, data_source="test"):
        """Forward pass for inference."""
        # Extract embeddings
        embeddings = self.extract_embeddings(X, data_source=data_source)
        
        # Apply constraints
        constrained_embeddings = self.constraintor(embeddings)
        
        # Compute anomaly scores
        anomaly_scores = self.estimator(constrained_embeddings)
        
        return anomaly_scores
    
    def fit(self, train_loader, epochs=50, lr=1e-3):
        """
        Fit the ResAD model.
        
        Args:
            train_loader: DataLoader with training data
            epochs: Number of training epochs
            lr: Learning rate
        """
        # First, collect all normal training data to fit feature extractor
        all_normal_data = []
        all_normal_labels = []
        
        for batch in train_loader:
            features = batch['features'].numpy()
            labels = batch['label'].numpy()
            
            # Filter normal data only (label == 0)
            normal_mask = labels == 0
            if np.any(normal_mask):
                all_normal_data.append(features[normal_mask])
                all_normal_labels.append(labels[normal_mask])
        
        if not all_normal_data:
            raise ValueError("No normal samples found in training data")
        
        X_normal = np.vstack(all_normal_data)
        y_normal = np.hstack(all_normal_labels)
        
        # Fit feature extractor
        logging.info(f"Fitting feature extractor with {len(X_normal)} normal samples")
        self.fit_feature_extractor(X_normal, y_normal)
        logging.info("Feature extractor fitted successfully")
        
        # Pre-compute embeddings for all training data
        logging.info("Pre-computing embeddings for training data...")
        all_embeddings = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Computing embeddings"):
            features = batch['features'].numpy()
            labels = batch['label'].numpy()
            
            # Extract embeddings using the fitted feature extractor
            # Use "test" data_source to avoid dimension issues
            batch_embeddings = self.extract_embeddings(features, data_source="test")
            logging.info(f"Batch {len(all_embeddings)}: features shape={features.shape}, embeddings shape={batch_embeddings.shape}, labels shape={labels.shape}")
            all_embeddings.append(batch_embeddings)
            all_labels.append(torch.tensor(labels, dtype=torch.long))
        
        # Combine all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        logging.info(f"Total embeddings shape: {all_embeddings.shape}")
        logging.info(f"Total labels shape: {all_labels.shape}")
        logging.info(f"Total samples: embeddings={len(all_embeddings)}, labels={len(all_labels)}")
        
        # Filter normal samples for constraintor training
        normal_mask = all_labels == 0
        normal_embeddings = all_embeddings[normal_mask]
        
        logging.info(f"Normal embeddings shape for constraintor training: {normal_embeddings.shape}")
        
        # Training loop for constraintor
        logging.info("Starting feature constraintor training...")
        print("Training feature constraintor...")
        optimizer = torch.optim.Adam(self.constraintor.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Create batches from pre-computed embeddings
        batch_size = train_loader.batch_size
        num_batches = (len(normal_embeddings) + batch_size - 1) // batch_size
        
        self.constraintor.train()
        
        for epoch in range(epochs):
            total_loss = 0
            processed_batches = 0
            
            # Shuffle embeddings for each epoch
            indices = torch.randperm(len(normal_embeddings))
            shuffled_embeddings = normal_embeddings[indices]
            
            for i in range(0, len(shuffled_embeddings), batch_size):
                end_idx = min(i + batch_size, len(shuffled_embeddings))
                batch_embeddings = shuffled_embeddings[i:end_idx]
                
                if hasattr(self.config, 'device'):
                    batch_embeddings = batch_embeddings.to(self.config.device)
                
                optimizer.zero_grad()
                
                # Apply constraints to embeddings
                constrained_embeddings = self.constraintor(batch_embeddings)
                
                # Constraint loss (encourage meaningful transformation)
                constraint_loss = self._compute_constraint_loss(batch_embeddings, constrained_embeddings)
                
                # Regularization loss
                reg_loss = self._compute_regularization_loss()
                
                total_loss_batch = constraint_loss + self.config.reg_weight * reg_loss
                total_loss_batch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.constraintor.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                processed_batches += 1
            
            avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
                logging.info(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
        
        logging.info("Feature constraintor training completed")
        
        # Fit distribution estimator on constrained embeddings
        logging.info("Starting distribution estimator fitting...")
        print("Fitting distribution estimator...")
        
        # Use the normal embeddings and apply constraints to get final embeddings
        with torch.no_grad():
            final_normal_embeddings = self.constraintor(normal_embeddings)
        
        # Fit estimator with constrained embeddings
        self.estimator.fit(final_normal_embeddings)
        logging.info("Distribution estimator fitted successfully")
        
        self.fitted = True
        logging.info("ResAD training completed successfully!")
        print("ResAD training completed!")
    
    def _compute_constraint_loss(self, embeddings, constrained_embeddings):
        """Compute constraint loss to encourage meaningful transformations."""
        # Convert to tensors if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if isinstance(constrained_embeddings, np.ndarray):
            constrained_embeddings = torch.tensor(constrained_embeddings, dtype=torch.float32)
        
        if hasattr(self.config, 'device'):
            embeddings = embeddings.to(self.config.device)
            constrained_embeddings = constrained_embeddings.to(self.config.device)
        
        # Encourage the constraintor to create meaningful transformations
        # while preserving important information
        
        # 1. Information preservation loss (cosine similarity)
        cos_sim = F.cosine_similarity(embeddings, constrained_embeddings, dim=1)
        info_loss = 1 - cos_sim.mean()
        
        # 2. Transformation magnitude loss (encourage non-trivial transformations)
        diff = constrained_embeddings - embeddings
        mag_loss = 1 / (torch.norm(diff, dim=1).mean() + 1e-8)
        
        # 3. Compactness loss (encourage compact normal representations)
        compact_loss = torch.var(constrained_embeddings, dim=0).mean()
        
        # Combine losses
        total_loss = info_loss + 0.1 * mag_loss + 0.1 * compact_loss
        
        return total_loss
    
    def _compute_regularization_loss(self):
        """Compute regularization loss for constraintor parameters."""
        reg_loss = 0
        for param in self.constraintor.parameters():
            reg_loss += torch.norm(param, p=2)
        return reg_loss
    
    def _fit_estimator(self, X_normal):
        """Fit the distribution estimator."""
        # Extract constrained embeddings for normal data
        embeddings = self.extract_embeddings(X_normal, data_source="train")
        constrained_embeddings = self.constraintor(embeddings)
        
        # Fit estimator
        self.estimator.fit(constrained_embeddings)
    
    def predict(self, X):
        """Predict anomaly scores for input data."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                logging.info(f"Predicting anomaly scores for {len(X)} samples")
                scores = self.forward(X, data_source="test")
            else:
                # Handle DataLoader
                logging.info("Predicting anomaly scores for DataLoader batches")
                all_scores = []
                for batch in X:
                    features = batch['features'].numpy()
                    batch_scores = self.forward(features, data_source="test")
                    all_scores.append(batch_scores)
                scores = torch.cat(all_scores, dim=0)
                logging.info(f"Predicted scores for {len(scores)} total samples")
        
        return scores.cpu().numpy()