import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import our modules
from models.resad_model import ResADTabular
from datasets.tabular_loader import ADBenchDataLoader, TabularAnomalyDataset, create_data_loader, get_dataset_info

# Comprehensive warning suppression
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress specific library warnings
logging.getLogger('hyperopt').setLevel(logging.ERROR)
logging.getLogger('pkg_resources').setLevel(logging.ERROR)

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
    
    # Calculate accuracy using threshold (median of scores)
    threshold = np.median(test_scores)
    y_pred = (test_scores > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    logger.info("=== EXPERIMENT RESULTS ===")
    logger.info(f"AUC (Area Under ROC Curve): {auc:.4f}")
    logger.info(f"AP (Average Precision): {ap:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
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
        'accuracy': float(accuracy),
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
    
    # Save summary to CSV file
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
        'constraintor_type': config.constraintor_type,
        'estimator_type': config.estimator_type,
        'use_multiscale': config.use_multiscale,
        'epochs': config.epochs,
        'lr': config.lr,
        'auc': float(auc),
        'ap': float(ap),
        'accuracy': float(accuracy),
        'test_samples': len(y_true),
        'anomaly_ratio': float(y_true.mean())
    }
    
    # Check if CSV exists and load it, otherwise create new DataFrame
    if csv_file.exists():
        try:
            csv_df = pd.read_csv(csv_file)
            csv_df = pd.concat([csv_df, pd.DataFrame([summary_row])], ignore_index=True)
        except Exception as e:
            logger.warning(f"Error loading existing CSV, creating new: {e}")
            csv_df = pd.DataFrame([summary_row])
    else:
        csv_df = pd.DataFrame([summary_row])
    
    # Save updated CSV
    csv_df.to_csv(csv_file, index=False)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Summary saved to: {csv_file}")
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