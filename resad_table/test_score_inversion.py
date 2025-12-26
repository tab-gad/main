#!/usr/bin/env python3
"""
Quick test to see if inverting scores fixes the performance issue.
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import torch
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# Import model components
from models.constraintor import ResidualConstraintor
from models.estimator import NormalDistributionEstimator

def load_embeddings(dataset_name, embeddings_dir="embeddings"):
    """Load pre-computed embeddings."""
    embeddings_path = Path(embeddings_dir) / f"{dataset_name}_tabpfn_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    return data

def test_score_inversion():
    """Test if inverting scores fixes the issue."""
    print("Testing score inversion for cross-domain experiment...")
    
    # Load data
    wine_data = load_embeddings('wine')
    glass_data = load_embeddings('glass')
    target_data = load_embeddings('breastw')
    
    # Prepare training data
    source_embeddings = np.vstack([wine_data['train_embeddings'], glass_data['train_embeddings']])
    source_labels = np.hstack([wine_data['train_labels'], glass_data['train_labels']])
    
    # Get normal embeddings for training
    normal_mask = source_labels == 0
    normal_embeddings = torch.tensor(source_embeddings[normal_mask], dtype=torch.float32)
    
    print(f"Training on {len(normal_embeddings)} normal samples")
    
    # Initialize model components
    constraintor = ResidualConstraintor(192, num_residual_blocks=3)
    estimator = NormalDistributionEstimator(192)
    
    # Train constraintor
    optimizer = torch.optim.Adam(constraintor.parameters(), lr=1e-3)
    constraintor.train()
    
    for epoch in range(10):  # Quick training
        if len(normal_embeddings) > 128:
            indices = torch.randperm(len(normal_embeddings))[:128]
            batch = normal_embeddings[indices]
        else:
            batch = normal_embeddings
            
        optimizer.zero_grad()
        constrained = constraintor(batch)
        loss = torch.nn.MSELoss()(constrained, batch) + 0.1 * torch.var(constrained, dim=0).mean()
        loss.backward()
        optimizer.step()
    
    constraintor.eval()
    
    # Fit estimator
    with torch.no_grad():
        constrained_normal = constraintor(normal_embeddings)
    estimator.fit(constrained_normal)
    
    # Test predictions
    test_embeddings = torch.tensor(target_data['test_embeddings'], dtype=torch.float32)
    test_labels = target_data['test_labels']
    
    with torch.no_grad():
        constrained_test = constraintor(test_embeddings)
        original_scores = estimator(constrained_test).numpy()
        inverted_scores = -original_scores  # Invert scores
    
    print(f"\nOriginal scores stats: mean={original_scores.mean():.2f}, std={original_scores.std():.2f}")
    print(f"Inverted scores stats: mean={inverted_scores.mean():.2f}, std={inverted_scores.std():.2f}")
    
    # Test performance
    original_auc = roc_auc_score(test_labels, original_scores)
    inverted_auc = roc_auc_score(test_labels, inverted_scores)
    
    original_ap = average_precision_score(test_labels, original_scores)
    inverted_ap = average_precision_score(test_labels, inverted_scores)
    
    print(f"\nPerformance Comparison:")
    print(f"Original scores: AUC={original_auc:.4f}, AP={original_ap:.4f}")
    print(f"Inverted scores: AUC={inverted_auc:.4f}, AP={inverted_ap:.4f}")
    
    if inverted_auc > original_auc:
        print("✓ Inverting scores improves performance!")
        print("The model is learning inverted patterns - normal data appears more anomalous.")
        return True
    else:
        print("✗ Score inversion doesn't help.")
        return False

if __name__ == '__main__':
    test_score_inversion()