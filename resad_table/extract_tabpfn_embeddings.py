#!/usr/bin/env python3
"""
Extract and save TabPFN embeddings for all ADBench datasets.

This script pre-computes TabPFN embeddings for all datasets and saves them
to disk for faster reuse in experiments.
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import torch
import pickle
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

# Import data loading and TabPFN extractor
from datasets.tabular_loader import ADBenchDataLoader
from models.tabpfn_extractor import TabPFNFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/embedding_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Available datasets
DATASETS = ['breastw', 'pima', 'wdbc', 'wine', 'cardio', 'glass', 'hepatitis', 'cover']

def extract_dataset_embeddings(dataset_name, data_loader, save_dir):
    """Extract TabPFN embeddings for a single dataset."""
    logging.info(f"Processing dataset: {dataset_name}")
    
    try:
        # Load train and test data
        train_dataset = data_loader.load_dataset(dataset_name, train=True, test_size=0.3, random_state=42)
        test_dataset = data_loader.load_dataset(dataset_name, train=False, test_size=0.3, random_state=42)
        
        logging.info(f"{dataset_name} - Train: {train_dataset.X_split.shape}, Labels: {train_dataset.y_split.shape}")
        logging.info(f"{dataset_name} - Test: {test_dataset.X_split.shape}, Labels: {test_dataset.y_split.shape}")
        
        # Verify data shapes before proceeding
        assert train_dataset.X_split.shape[0] == train_dataset.y_split.shape[0], \
            f"Train data shape mismatch: {train_dataset.X_split.shape[0]} != {train_dataset.y_split.shape[0]}"
        assert test_dataset.X_split.shape[0] == test_dataset.y_split.shape[0], \
            f"Test data shape mismatch: {test_dataset.X_split.shape[0]} != {test_dataset.y_split.shape[0]}"
        
        # Initialize TabPFN extractor
        extractor = TabPFNFeatureExtractor(n_estimators=1, n_fold=5, use_scaler=True)
        
        # Fit on training data (normal samples only)
        normal_mask = train_dataset.y_split == 0
        normal_train_data = train_dataset.X_split[normal_mask]
        normal_train_labels = train_dataset.y_split[normal_mask]
        
        logging.info(f"{dataset_name} - Fitting on {len(normal_train_data)} normal samples")
        extractor.fit(normal_train_data, normal_train_labels)
        
        # Extract embeddings for train set
        logging.info(f"{dataset_name} - Extracting train embeddings...")
        train_embeddings = extractor(train_dataset.X_split, data_source="train")
        
        # Extract embeddings for test set  
        logging.info(f"{dataset_name} - Extracting test embeddings...")
        test_embeddings = extractor(test_dataset.X_split, data_source="test")
        
        # CRITICAL: Verify embedding and label shapes match
        logging.info(f"{dataset_name} - Verifying shapes after extraction...")
        logging.info(f"  Train embeddings: {train_embeddings.shape}")
        logging.info(f"  Train labels: {train_dataset.y_split.shape}")
        logging.info(f"  Test embeddings: {test_embeddings.shape}")
        logging.info(f"  Test labels: {test_dataset.y_split.shape}")
        
        # Ensure embeddings and labels have matching lengths
        if train_embeddings.shape[0] != train_dataset.y_split.shape[0]:
            logging.warning(f"⚠️  {dataset_name} - Train length mismatch! Trimming to minimum...")
            min_len = min(train_embeddings.shape[0], train_dataset.y_split.shape[0])
            train_embeddings = train_embeddings[:min_len]
            train_features = train_dataset.X_split[:min_len]
            train_labels = train_dataset.y_split[:min_len]
        else:
            train_features = train_dataset.X_split
            train_labels = train_dataset.y_split
            
        if test_embeddings.shape[0] != test_dataset.y_split.shape[0]:
            logging.warning(f"⚠️  {dataset_name} - Test length mismatch! Trimming to minimum...")
            min_len = min(test_embeddings.shape[0], test_dataset.y_split.shape[0])
            test_embeddings = test_embeddings[:min_len]
            test_features = test_dataset.X_split[:min_len]
            test_labels = test_dataset.y_split[:min_len]
        else:
            test_features = test_dataset.X_split
            test_labels = test_dataset.y_split
        
        # Convert to numpy for saving
        train_embeddings_np = train_embeddings.detach().cpu().numpy()
        test_embeddings_np = test_embeddings.detach().cpu().numpy()
        
        # Prepare data to save (using the length-matched arrays)
        embedding_data = {
            'dataset_name': dataset_name,
            'train_embeddings': train_embeddings_np,
            'train_labels': train_labels,
            'train_features': train_features,
            'test_embeddings': test_embeddings_np,
            'test_labels': test_labels,
            'test_features': test_features,
            'feature_names': train_dataset.feature_names,
            'embedding_dim': train_embeddings_np.shape[1],
            'extractor_config': {
                'n_estimators': extractor.n_estimators,
                'n_fold': extractor.n_fold,
                'use_scaler': extractor.use_scaler
            },
            'extraction_time': datetime.now().isoformat()
        }
        
        # Final verification before saving
        logging.info(f"{dataset_name} - Final verification:")
        logging.info(f"  Train: embeddings {train_embeddings_np.shape} == labels {train_labels.shape}")
        logging.info(f"  Test: embeddings {test_embeddings_np.shape} == labels {test_labels.shape}")
        
        assert train_embeddings_np.shape[0] == train_labels.shape[0], "Train embeddings-labels mismatch!"
        assert test_embeddings_np.shape[0] == test_labels.shape[0], "Test embeddings-labels mismatch!"
        
        # Save embeddings
        save_path = save_dir / f"{dataset_name}_tabpfn_embeddings.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        logging.info(f"{dataset_name} - Saved embeddings to: {save_path}")
        logging.info(f"{dataset_name} - Train embeddings shape: {train_embeddings_np.shape}")
        logging.info(f"{dataset_name} - Test embeddings shape: {test_embeddings_np.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"{dataset_name} - Error: {str(e)}")
        return False

def main():
    """Extract embeddings for all datasets."""
    logging.info("Starting TabPFN embedding extraction for all datasets...")
    
    # Create save directory
    save_dir = Path("embeddings")
    save_dir.mkdir(exist_ok=True)
    
    # Initialize data loader
    data_loader = ADBenchDataLoader('../dataset/Classical', normalize=True)
    
    success_count = 0
    failed_datasets = []
    
    for dataset_name in tqdm(DATASETS, desc="Processing datasets"):
        success = extract_dataset_embeddings(dataset_name, data_loader, save_dir)
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
    
    # Summary
    logging.info("="*60)
    logging.info("EMBEDDING EXTRACTION SUMMARY")
    logging.info("="*60)
    logging.info(f"Total datasets: {len(DATASETS)}")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Failed: {len(failed_datasets)}")
    
    if failed_datasets:
        logging.info(f"Failed datasets: {failed_datasets}")
    
    logging.info(f"Embeddings saved in: {save_dir.absolute()}")
    
    # List saved files
    embedding_files = list(save_dir.glob("*.pkl"))
    logging.info(f"Saved files ({len(embedding_files)}):")
    for file in embedding_files:
        size_mb = file.stat().st_size / (1024*1024)
        logging.info(f"  {file.name} ({size_mb:.1f} MB)")

def load_embeddings(dataset_name, embeddings_dir="embeddings"):
    """Utility function to load saved embeddings."""
    embeddings_path = Path(embeddings_dir) / f"{dataset_name}_tabpfn_embeddings.pkl"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

if __name__ == '__main__':
    main()