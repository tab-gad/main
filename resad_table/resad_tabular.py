import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import logging
from datetime import datetime
from pathlib import Path

# Import our modules
from models.tabpfn_extractor import TabPFNFeatureExtractor, MultiScaleTabPFNExtractor
from models.constraintor import TabularFeatureConstraintor, ResidualConstraintor, AdaptiveConstraintor
from models.estimator import NormalDistributionEstimator, FlowBasedEstimator, EnsembleEstimator
from datasets.tabular_loader import ADBenchDataLoader, TabularAnomalyDataset, create_data_loader, get_dataset_info

warnings.filterwarnings('ignore')

def setup_logging():
    """Set up logging with date-based filename and console output."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with current date
    current_time = datetime.now()
    log_filename = log_dir / f"resad_tabular_{current_time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up logging configuration
    root_logger.setLevel(logging.INFO)
    
    # Create formatter with filename and line number
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler  
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Test logging
    logging.info("=== ResAD Tabular Experiment Started ===")
    logging.info(f"Log file: {log_filename}")
    
    return root_logger


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


def train_resad(config):
    """Main training function."""
    logger = logging.getLogger(__name__)
    
    # Set device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {config.device}")
    logger.info(f"Experiment mode: {config.mode}")
    
    # Load datasets
    logger.info("Loading datasets...")
    data_loader = ADBenchDataLoader(config.data_root, normalize=True)
    
    # Load training and test datasets
    if config.mode == 'single_domain':
        logger.info(f"Single domain experiment on dataset: {config.dataset}")
        train_dataset = data_loader.load_dataset(config.dataset, train=True, 
                                                 test_size=config.test_size, 
                                                 random_state=config.random_state)
        test_dataset = data_loader.load_dataset(config.dataset, train=False,
                                                test_size=config.test_size,
                                                random_state=config.random_state)
        datasets = {'train': train_dataset, 'test': test_dataset}
        
        # Print dataset information
        logger.info("Dataset Information:")
        logger.info(f"Train: {get_dataset_info(datasets['train'])}")
        logger.info(f"Test: {get_dataset_info(datasets['test'])}")
        
        # Create data loaders
        train_loader = create_data_loader(datasets['train'], batch_size=config.batch_size, 
                                         shuffle=True, num_workers=config.num_workers)
        test_loader = create_data_loader(datasets['test'], batch_size=config.batch_size, 
                                        shuffle=False, num_workers=config.num_workers)
    
    elif config.mode == 'cross_domain':
        logger.info(f"Cross domain experiment:")
        logger.info(f"  Source datasets: {config.source_datasets}")
        logger.info(f"  Target dataset: {config.target_dataset}")
        
        datasets = data_loader.get_cross_domain_splits(
            config.source_datasets, config.target_dataset,
            test_size=config.test_size, random_state=config.random_state
        )
        
        # For cross-domain: Train on SOURCE datasets, Test on TARGET dataset
        train_dataset = datasets['source']  # Use combined source data for training
        test_dataset = datasets['target_test']       # Use target test data for testing
        
        logger.info("Dataset Information:")
        logger.info(f"Train (source combined): {get_dataset_info(train_dataset)}")
        logger.info(f"Test (target): {get_dataset_info(test_dataset)}")
        
        # Create data loaders
        train_loader = create_data_loader(train_dataset, batch_size=config.batch_size, 
                                         shuffle=True, num_workers=config.num_workers)
        test_loader = create_data_loader(test_dataset, batch_size=config.batch_size, 
                                        shuffle=False, num_workers=config.num_workers)
    
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
    
    # Initialize and train model
    logger.info("Initializing ResAD model...")
    logger.info(f"Model configuration:")
    logger.info(f"  - Use multiscale: {config.use_multiscale}")
    logger.info(f"  - Constraintor type: {config.constraintor_type}")
    logger.info(f"  - Estimator type: {config.estimator_type}")
    logger.info(f"  - Training epochs: {config.epochs}")
    logger.info(f"  - Learning rate: {config.lr}")
    
    model = ResADTabular(config)
    
    logger.info("Starting model training...")
    model.fit(train_loader, epochs=config.epochs, lr=config.lr)
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    test_scores = model.predict(test_loader)
    
    # Get true labels
    all_labels = []
    for batch in test_loader:
        all_labels.append(batch['label'].numpy())
    y_true = np.hstack(all_labels)
    
    # Compute metrics
    auc = roc_auc_score(y_true, test_scores)
    ap = average_precision_score(y_true, test_scores)
    
    logger.info("=== EXPERIMENT RESULTS ===")
    logger.info(f"AUC (Area Under ROC Curve): {auc:.4f}")
    logger.info(f"AP (Average Precision): {ap:.4f}")
    logger.info(f"Test samples: {len(y_true)}")
    logger.info(f"Anomaly ratio: {y_true.mean():.3f}")
    
    # Save results
    config_dict = {}
    for k, v in vars(config).items():
        if isinstance(v, torch.device):
            config_dict[k] = str(v)
        else:
            config_dict[k] = v

    results = {
        'config': config_dict,
        'auc': float(auc),
        'ap': float(ap),
        'test_scores': test_scores.tolist(),
        'test_labels': y_true.tolist(),
        'experiment_time': datetime.now().isoformat()
    }
    
    # Create results directory
    results_dir = Path(config.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config.mode == 'cross_domain':
        results_file = results_dir / f"results_cross_{'-'.join(config.source_datasets)}_to_{config.target_dataset}_{timestamp}.json"
    else:
        results_file = results_dir / f"results_{config.dataset}_{config.mode}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info("=== EXPERIMENT COMPLETED ===")
    
    return model, results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ResAD for Tabular Data')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='../dataset/Classical',
                        help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, default='breastw',
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default='single_domain',
                        choices=['single_domain', 'cross_domain'],
                        help='Evaluation mode')
    parser.add_argument('--source_datasets', nargs='+', default=['breastw', 'pima'],
                        help='Source datasets for cross-domain evaluation')
    parser.add_argument('--target_dataset', type=str, default='wdbc',
                        help='Target dataset for cross-domain evaluation')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    
    # Model arguments
    parser.add_argument('--use_multiscale', action='store_true',
                        help='Use multi-scale feature extractor')
    parser.add_argument('--n_scales', type=int, default=3,
                        help='Number of scales for multi-scale extractor')
    parser.add_argument('--n_estimators', type=int, default=1,
                        help='Number of TabPFN estimators')
    parser.add_argument('--n_fold', type=int, default=5,
                        help='Number of folds for TabPFN cross-validation')
    parser.add_argument('--use_scaler', action='store_true', default=True,
                        help='Use feature scaling in TabPFN extractor')
    parser.add_argument('--final_embedding_dim', type=int, default=128,
                        help='Final embedding dimension for multi-scale fusion')
    
    # Constraintor arguments
    parser.add_argument('--constraintor_type', type=str, default='residual',
                        choices=['basic', 'residual', 'adaptive'],
                        help='Type of feature constraintor')
    parser.add_argument('--num_residual_blocks', type=int, default=3,
                        help='Number of residual blocks')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Estimator arguments
    parser.add_argument('--estimator_type', type=str, default='normal',
                        choices=['normal', 'flow', 'ensemble'],
                        help='Type of distribution estimator')
    parser.add_argument('--num_flows', type=int, default=4,
                        help='Number of flow layers')
    parser.add_argument('--num_ensemble_estimators', type=int, default=3,
                        help='Number of estimators in ensemble')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                        help='Regularization weight')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Save arguments
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    
    # Train model
    model, results = train_resad(args)