#!/usr/bin/env python3
"""
Debug version of ResAD with extensive logging to understand poor performance.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.manifold import TSNE
import pickle
from pathlib import Path

# Import model components
from models.constraintor import ResidualConstraintor
from models.estimator import NormalDistributionEstimator

warnings.filterwarnings('ignore')

def load_embeddings(dataset_name, embeddings_dir="embeddings"):
    """Load pre-computed embeddings."""
    embeddings_path = Path(embeddings_dir) / f"{dataset_name}_tabpfn_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_embeddings_quality():
    """Analyze the quality of pre-computed embeddings."""
    print("="*60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*60)
    
    datasets = ['wine', 'glass', 'breastw']
    
    for dataset in datasets:
        data = load_embeddings(dataset)
        train_emb = data['train_embeddings']
        train_labels = data['train_labels']
        test_emb = data['test_embeddings'] 
        test_labels = data['test_labels']
        
        print(f"\n{dataset.upper()} Dataset:")
        print(f"  Train embeddings: {train_emb.shape}")
        print(f"  Train labels: {train_labels.shape}, Normal: {np.sum(train_labels == 0)}, Anomaly: {np.sum(train_labels == 1)}")
        print(f"  Test embeddings: {test_emb.shape}")
        print(f"  Test labels: {test_labels.shape}, Normal: {np.sum(test_labels == 0)}, Anomaly: {np.sum(test_labels == 1)}")
        
        # Check embedding statistics
        print(f"  Train embedding stats: mean={train_emb.mean():.4f}, std={train_emb.std():.4f}")
        print(f"  Test embedding stats: mean={test_emb.mean():.4f}, std={test_emb.std():.4f}")
        
        # Check for NaN or inf
        print(f"  Train NaN: {np.isnan(train_emb).any()}, Inf: {np.isinf(train_emb).any()}")
        print(f"  Test NaN: {np.isnan(test_emb).any()}, Inf: {np.isinf(test_emb).any()}")
        
        # Analyze normal vs anomaly embedding separation
        normal_emb = train_emb[train_labels == 0]
        anomaly_emb = train_emb[train_labels == 1]
        
        if len(anomaly_emb) > 0:
            normal_mean = normal_emb.mean(axis=0)
            anomaly_mean = anomaly_emb.mean(axis=0)
            separation = np.linalg.norm(normal_mean - anomaly_mean)
            print(f"  Normal vs Anomaly separation: {separation:.4f}")
            
            # Check variance
            normal_var = normal_emb.var(axis=0).mean()
            anomaly_var = anomaly_emb.var(axis=0).mean()
            print(f"  Normal variance: {normal_var:.4f}, Anomaly variance: {anomaly_var:.4f}")

def debug_cross_domain_experiment():
    """Debug the cross-domain experiment step by step."""
    print("="*60)
    print("CROSS-DOMAIN EXPERIMENT DEBUG")
    print("="*60)
    
    # Load source datasets (wine + glass)
    wine_data = load_embeddings('wine')
    glass_data = load_embeddings('glass')
    target_data = load_embeddings('breastw')
    
    print("\nSOURCE DATA ANALYSIS:")
    print(f"Wine train: {wine_data['train_embeddings'].shape}, labels: {wine_data['train_labels'].shape}")
    print(f"Glass train: {glass_data['train_embeddings'].shape}, labels: {glass_data['train_labels'].shape}")
    
    # Combine source data
    source_embeddings = np.vstack([wine_data['train_embeddings'], glass_data['train_embeddings']])
    source_labels = np.hstack([wine_data['train_labels'], glass_data['train_labels']])
    
    print(f"Combined source: {source_embeddings.shape}, labels: {source_labels.shape}")
    print(f"Source normal samples: {np.sum(source_labels == 0)}")
    print(f"Source anomaly samples: {np.sum(source_labels == 1)}")
    
    # Target data
    target_embeddings = target_data['test_embeddings']
    target_labels = target_data['test_labels']
    
    print(f"\nTARGET DATA:")
    print(f"Target test: {target_embeddings.shape}, labels: {target_labels.shape}")
    print(f"Target normal samples: {np.sum(target_labels == 0)}")
    print(f"Target anomaly samples: {np.sum(target_labels == 1)}")
    
    # Analyze domain differences
    source_mean = source_embeddings.mean(axis=0)
    target_mean = target_embeddings.mean(axis=0)
    domain_gap = np.linalg.norm(source_mean - target_mean)
    print(f"\nDomain gap (L2 distance): {domain_gap:.4f}")
    
    # Check embedding distributions
    source_norm = np.linalg.norm(source_embeddings, axis=1)
    target_norm = np.linalg.norm(target_embeddings, axis=1)
    print(f"Source embedding norms: mean={source_norm.mean():.4f}, std={source_norm.std():.4f}")
    print(f"Target embedding norms: mean={target_norm.mean():.4f}, std={target_norm.std():.4f}")

def debug_model_training():
    """Debug the model training process."""
    print("="*60)
    print("MODEL TRAINING DEBUG")
    print("="*60)
    
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
    print(f"Normal embeddings stats: mean={normal_embeddings.mean():.4f}, std={normal_embeddings.std():.4f}")
    
    # Initialize model components
    constraintor = ResidualConstraintor(192, num_residual_blocks=3)
    estimator = NormalDistributionEstimator(192)
    
    print("\nTRAINING CONSTRAINTOR...")
    
    # Train constraintor
    optimizer = torch.optim.Adam(constraintor.parameters(), lr=1e-3)
    constraintor.train()
    
    losses = []
    for epoch in range(5):  # Few epochs for debugging
        # Sample subset
        if len(normal_embeddings) > 128:
            indices = torch.randperm(len(normal_embeddings))[:128]
            batch_embeddings = normal_embeddings[indices]
        else:
            batch_embeddings = normal_embeddings
            
        optimizer.zero_grad()
        constrained = constraintor(batch_embeddings)
        
        # Loss components
        recon_loss = nn.MSELoss()(constrained, batch_embeddings)
        compactness_loss = torch.var(constrained, dim=0).mean()
        total_loss = recon_loss + 0.1 * compactness_loss
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        print(f"  Epoch {epoch}: Total loss={total_loss:.4f}, Recon={recon_loss:.4f}, Compact={compactness_loss:.4f}")
    
    constraintor.eval()
    
    # Check transformation effects
    with torch.no_grad():
        original = normal_embeddings[:10]
        constrained = constraintor(original)
        
        print(f"\nTRANSFORMATION ANALYSIS:")
        print(f"Original stats: mean={original.mean():.4f}, std={original.std():.4f}")
        print(f"Constrained stats: mean={constrained.mean():.4f}, std={constrained.std():.4f}")
        
        # Check if transformation is meaningful
        change_norm = torch.norm(constrained - original, dim=1).mean()
        print(f"Average change magnitude: {change_norm:.4f}")
    
    # Fit estimator
    print("\nFITTING ESTIMATOR...")
    with torch.no_grad():
        all_constrained = constraintor(normal_embeddings)
    
    estimator.fit(all_constrained)
    
    print(f"Estimator fitted on {len(all_constrained)} samples")
    print(f"Learned mean: {estimator.mean[:5]}...")  # First 5 dimensions
    print(f"Learned covariance trace: {torch.trace(estimator.covariance):.4f}")

def debug_prediction_process():
    """Debug the prediction process."""
    print("="*60)
    print("PREDICTION PROCESS DEBUG")
    print("="*60)
    
    # Recreate the full experiment
    wine_data = load_embeddings('wine')
    glass_data = load_embeddings('glass')
    target_data = load_embeddings('breastw')
    
    # Training phase
    source_embeddings = np.vstack([wine_data['train_embeddings'], glass_data['train_embeddings']])
    source_labels = np.hstack([wine_data['train_labels'], glass_data['train_labels']])
    
    normal_mask = source_labels == 0
    normal_embeddings = torch.tensor(source_embeddings[normal_mask], dtype=torch.float32)
    
    # Quick training
    constraintor = ResidualConstraintor(192)
    estimator = NormalDistributionEstimator(192)
    
    optimizer = torch.optim.Adam(constraintor.parameters(), lr=1e-3)
    constraintor.train()
    
    for epoch in range(10):
        if len(normal_embeddings) > 128:
            indices = torch.randperm(len(normal_embeddings))[:128]
            batch = normal_embeddings[indices]
        else:
            batch = normal_embeddings
            
        optimizer.zero_grad()
        constrained = constraintor(batch)
        loss = nn.MSELoss()(constrained, batch) + 0.1 * torch.var(constrained, dim=0).mean()
        loss.backward()
        optimizer.step()
    
    constraintor.eval()
    
    with torch.no_grad():
        constrained_normal = constraintor(normal_embeddings)
    
    estimator.fit(constrained_normal)
    
    # Testing phase
    test_embeddings = torch.tensor(target_data['test_embeddings'], dtype=torch.float32)
    test_labels = target_data['test_labels']
    
    print(f"Predicting on {len(test_embeddings)} test samples")
    print(f"Test labels: Normal={np.sum(test_labels == 0)}, Anomaly={np.sum(test_labels == 1)}")
    
    # Get predictions
    with torch.no_grad():
        constrained_test = constraintor(test_embeddings)
        anomaly_scores = estimator(constrained_test).numpy()
    
    print(f"\nPREDICTION ANALYSIS:")
    print(f"Anomaly scores stats: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
    print(f"Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    
    # Analyze scores by true labels
    normal_scores = anomaly_scores[test_labels == 0]
    anomaly_scores_true = anomaly_scores[test_labels == 1]
    
    if len(anomaly_scores_true) > 0:
        print(f"Normal samples score: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
        print(f"Anomaly samples score: mean={anomaly_scores_true.mean():.4f}, std={anomaly_scores_true.std():.4f}")
        
        # Check if anomalies have higher scores
        score_separation = anomaly_scores_true.mean() - normal_scores.mean()
        print(f"Score separation (anomaly - normal): {score_separation:.4f}")
        
        if score_separation < 0:
            print("WARNING: Normal samples have higher anomaly scores than anomalies!")
        
        # Distribution overlap
        overlap_ratio = len(anomaly_scores_true[anomaly_scores_true < normal_scores.mean()]) / len(anomaly_scores_true)
        print(f"Overlap ratio (anomalies below normal mean): {overlap_ratio:.4f}")
    
    # Final metrics
    auc = roc_auc_score(test_labels, anomaly_scores)
    ap = average_precision_score(test_labels, anomaly_scores)
    
    threshold = np.median(anomaly_scores)
    y_pred = (anomaly_scores > threshold).astype(int)
    accuracy = accuracy_score(test_labels, y_pred)
    
    print(f"\nFINAL METRICS:")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return auc, ap, accuracy

def main():
    print("Starting comprehensive debugging analysis...")
    
    # Step 1: Check embedding quality
    analyze_embeddings_quality()
    
    # Step 2: Analyze cross-domain setup
    debug_cross_domain_experiment()
    
    # Step 3: Debug model training
    debug_model_training()
    
    # Step 4: Debug prediction
    auc, ap, acc = debug_prediction_process()
    
    print("\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    print(f"Final performance: AUC={auc:.4f}, AP={ap:.4f}, Accuracy={acc:.4f}")
    
    if auc < 0.5:
        print("\n🔍 POTENTIAL ISSUES:")
        print("1. Domain gap too large between source (wine+glass) and target (breastw)")
        print("2. Normal/anomaly labels might be inverted")
        print("3. Constraintor might be learning wrong patterns")
        print("4. Estimator might not be fitting correctly")
        print("5. TabPFN embeddings might not be domain-transferable")

if __name__ == '__main__':
    main()