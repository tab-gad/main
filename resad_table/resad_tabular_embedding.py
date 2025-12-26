#!/usr/bin/env python3
"""
ResAD for Tabular Data using Pre-computed TabPFN Embeddings.

This script uses pre-computed TabPFN embeddings to speed up experiments
by skipping the embedding extraction step.
"""

import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

# Import model components
from models.constraintor import TabularFeatureConstraintor, ResidualConstraintor, AdaptiveConstraintor
from models.estimator import NormalDistributionEstimator, FlowBasedEstimator, EnsembleEstimator

# Comprehensive warning suppression
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.getLogger('hyperopt').setLevel(logging.ERROR)
logging.getLogger('pkg_resources').setLevel(logging.ERROR)

def setup_logging():
    """Set up logging with date-based filename and console output."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    current_time = datetime.now()
    log_filename = log_dir / f"resad_embedding_{current_time.strftime('%Y%m%d_%H%M%S')}.log"
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info("=== ResAD Tabular Embedding Experiment Started ===")
    logging.info(f"Log file: {log_filename}")
    
    return root_logger

class PrecomputedEmbeddingDataset(Dataset):
    """Dataset class for pre-computed embeddings."""
    
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Ensure embeddings and labels have the same length
        assert len(self.embeddings) == len(self.labels), \
            f"Embeddings and labels must have same length: {len(self.embeddings)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if idx >= len(self.embeddings):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.embeddings)}")
        return {
            'embeddings': self.embeddings[idx],
            'label': self.labels[idx],
            'index': idx
        }

class ResADEmbedding(nn.Module):
    """ResAD model using pre-computed embeddings."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = 192  # TabPFN embedding dimension
        
        # Initialize constraintor
        if config.constraintor_type == 'basic':
            self.constraintor = TabularFeatureConstraintor(
                self.embedding_dim, dropout_rate=config.dropout_rate
            )
        elif config.constraintor_type == 'residual':
            self.constraintor = ResidualConstraintor(
                self.embedding_dim, num_residual_blocks=config.num_residual_blocks
            )
        elif config.constraintor_type == 'adaptive':
            self.constraintor = AdaptiveConstraintor(
                self.embedding_dim, dropout_rate=config.dropout_rate
            )
        else:
            raise ValueError(f"Unknown constraintor type: {config.constraintor_type}")
        
        # Initialize estimator
        if config.estimator_type == 'normal':
            self.estimator = NormalDistributionEstimator(self.embedding_dim)
        elif config.estimator_type == 'flow':
            self.estimator = FlowBasedEstimator(self.embedding_dim, num_flows=config.num_flows)
        elif config.estimator_type == 'ensemble':
            self.estimator = EnsembleEstimator(self.embedding_dim, num_estimators=config.num_ensemble_estimators)
        else:
            raise ValueError(f"Unknown estimator type: {config.estimator_type}")
        
        self.fitted = False
    
    def fit(self, train_loader, epochs=50, lr=1e-3):
        """Fit the ResAD model on pre-computed embeddings."""
        logger = logging.getLogger(__name__)
        
        # Extract normal embeddings for training
        normal_embeddings = []
        
        for batch in train_loader:
            embeddings = batch['embeddings']
            labels = batch['label']
            
            # Select normal samples (label == 0)
            normal_mask = labels == 0
            if normal_mask.any():
                normal_embeddings.append(embeddings[normal_mask])
        
        if not normal_embeddings:
            raise ValueError("No normal samples found in training data")
        
        normal_embeddings = torch.cat(normal_embeddings, dim=0)
        logger.info(f"Training on {len(normal_embeddings)} normal samples")
        
        # Move to device
        device = next(self.parameters()).device
        normal_embeddings = normal_embeddings.to(device)
        
        # Train constraintor
        logger.info("Training feature constraintor...")
        optimizer = torch.optim.Adam(self.constraintor.parameters(), lr=lr)
        
        self.constraintor.train()
        for epoch in range(epochs):
            # Random sampling for training
            if len(normal_embeddings) > 256:
                indices = torch.randperm(len(normal_embeddings))[:256]
                batch_embeddings = normal_embeddings[indices]
            else:
                batch_embeddings = normal_embeddings
            
            optimizer.zero_grad()
            constrained = self.constraintor(batch_embeddings)
            
            # Simple reconstruction loss
            loss = nn.MSELoss()(constrained, batch_embeddings)
            
            # Compactness loss (encourage smaller variance)
            compactness_loss = torch.var(constrained, dim=0).mean()
            
            total_loss = loss + 0.1 * compactness_loss
            total_loss.backward()
            optimizer.step()
        
        self.constraintor.eval()
        logger.info("Constraintor training completed")
        
        # Fit estimator on constrained embeddings
        logger.info("Fitting distribution estimator...")
        with torch.no_grad():
            constrained_embeddings = self.constraintor(normal_embeddings)
        
        self.estimator.fit(constrained_embeddings)
        logger.info("Distribution estimator fitted")
        
        self.fitted = True
        logger.info("ResAD embedding training completed!")
    
    def predict(self, test_loader):
        """Predict anomaly scores."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.eval()
        scores = []
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch['embeddings'].to(device)
                constrained = self.constraintor(embeddings)
                batch_scores = self.estimator(constrained)
                scores.append(batch_scores.cpu().numpy())
        
        return np.concatenate(scores)

def load_dataset_embeddings(dataset_name, embeddings_dir="embeddings"):
    """Load pre-computed embeddings for a dataset."""
    embeddings_path = Path(embeddings_dir) / f"{dataset_name}_tabpfn_embeddings.pkl"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def train_resad_embedding(config):
    """Main training function using pre-computed embeddings."""
    logger = logging.getLogger(__name__)
    
    # Set device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {config.device}")
    logger.info(f"Experiment mode: {config.mode}")
    
    # Load pre-computed embeddings
    logger.info("Loading pre-computed embeddings...")
    
    if config.mode == 'single_domain':
        logger.info(f"Single domain experiment on dataset: {config.dataset}")
        
        # Load single dataset embeddings
        data = load_dataset_embeddings(config.dataset)
        
        train_dataset = PrecomputedEmbeddingDataset(
            data['train_embeddings'], data['train_labels']
        )
        test_dataset = PrecomputedEmbeddingDataset(
            data['test_embeddings'], data['test_labels']
        )
        
        logger.info(f"Train embeddings: {data['train_embeddings'].shape}")
        logger.info(f"Test embeddings: {data['test_embeddings'].shape}")
        
    elif config.mode == 'cross_domain':
        logger.info(f"Cross domain experiment:")
        logger.info(f"  Source datasets: {config.source_datasets}")
        logger.info(f"  Target dataset: {config.target_dataset}")
        
        # Load source datasets and combine
        source_embeddings = []
        source_labels = []
        
        for source_dataset in config.source_datasets:
            source_data = load_dataset_embeddings(source_dataset)
            source_embeddings.append(source_data['train_embeddings'])
            source_labels.append(source_data['train_labels'])
            logger.info(f"  {source_dataset}: {source_data['train_embeddings'].shape}")
        
        # Combine source data
        combined_embeddings = np.vstack(source_embeddings)
        combined_labels = np.hstack(source_labels)
        
        # Ensure they have the same length (trim to minimum)
        min_len = min(len(combined_embeddings), len(combined_labels))
        combined_embeddings = combined_embeddings[:min_len]
        combined_labels = combined_labels[:min_len]
        
        train_dataset = PrecomputedEmbeddingDataset(combined_embeddings, combined_labels)
        
        # Load target dataset for testing
        target_data = load_dataset_embeddings(config.target_dataset)
        test_dataset = PrecomputedEmbeddingDataset(
            target_data['test_embeddings'], target_data['test_labels']
        )
        
        logger.info(f"Combined train embeddings: {combined_embeddings.shape}")
        logger.info(f"Target test embeddings: {target_data['test_embeddings'].shape}")
    
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Initialize and train model
    logger.info("Initializing ResAD embedding model...")
    logger.info(f"Model configuration:")
    logger.info(f"  - Constraintor type: {config.constraintor_type}")
    logger.info(f"  - Estimator type: {config.estimator_type}")
    logger.info(f"  - Training epochs: {config.epochs}")
    logger.info(f"  - Learning rate: {config.lr}")
    
    model = ResADEmbedding(config)
    model = model.to(config.device)
    
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
    
    # Calculate accuracy
    threshold = np.median(test_scores)
    y_pred = (test_scores > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    logger.info("=== EXPERIMENT RESULTS ===")
    logger.info(f"AUC (Area Under ROC Curve): {auc:.4f}")
    logger.info(f"AP (Average Precision): {ap:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Test samples: {len(y_true)}")
    logger.info(f"Anomaly ratio: {y_true.mean():.3f}")
    
    # Save results to CSV (same format as original)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_file = results_dir / "results.csv"
    
    # Prepare experiment summary row
    if config.mode == 'cross_domain':
        experiment_name = f"cross_{'-'.join(config.source_datasets)}_to_{config.target_dataset}"
        datasets_info = f"{'+'.join(config.source_datasets)} → {config.target_dataset}"
    else:
        experiment_name = f"{config.dataset}_{config.mode}"
        datasets_info = config.dataset
    
    summary_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_name': experiment_name,
        'mode': config.mode,
        'datasets': datasets_info,
        'feature_extractor': 'tabpfn_precomputed',
        'padding_mode': 'N/A',
        'constraintor_type': config.constraintor_type,
        'estimator_type': config.estimator_type,
        'use_multiscale': False,
        'epochs': config.epochs,
        'lr': config.lr,
        'auc': float(auc),
        'ap': float(ap),
        'accuracy': float(accuracy),
        'test_samples': len(y_true),
        'anomaly_ratio': float(y_true.mean())
    }
    
    # Save to CSV
    if csv_file.exists():
        try:
            csv_df = pd.read_csv(csv_file)
            csv_df = pd.concat([csv_df, pd.DataFrame([summary_row])], ignore_index=True)
        except Exception as e:
            logger.warning(f"Error loading existing CSV, creating new: {e}")
            csv_df = pd.DataFrame([summary_row])
    else:
        csv_df = pd.DataFrame([summary_row])
    
    csv_df.to_csv(csv_file, index=False)
    
    logger.info(f"Summary saved to: {csv_file}")
    logger.info("=== EXPERIMENT COMPLETED ===")
    
    return model, auc, ap, accuracy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ResAD for Tabular Data with Pre-computed Embeddings')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='breastw',
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default='single_domain',
                        choices=['single_domain', 'cross_domain'],
                        help='Evaluation mode')
    parser.add_argument('--source_datasets', nargs='+', default=['breastw'],
                        help='Source datasets for cross-domain evaluation')
    parser.add_argument('--target_dataset', type=str, default='wdbc',
                        help='Target dataset for cross-domain evaluation')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings',
                        help='Directory containing pre-computed embeddings')
    
    # Model arguments
    parser.add_argument('--constraintor_type', type=str, default='residual',
                        choices=['basic', 'residual', 'adaptive'],
                        help='Type of feature constraintor')
    parser.add_argument('--num_residual_blocks', type=int, default=3,
                        help='Number of residual blocks')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    
    parser.add_argument('--estimator_type', type=str, default='normal',
                        choices=['normal', 'flow', 'ensemble'],
                        help='Type of distribution estimator')
    parser.add_argument('--num_flows', type=int, default=4,
                        help='Number of flow layers')
    parser.add_argument('--num_ensemble_estimators', type=int, default=3,
                        help='Number of estimators in ensemble')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    model, auc, ap, accuracy = train_resad_embedding(args)