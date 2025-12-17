import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np


class NormalDistributionEstimator(nn.Module):
    """
    Normal distribution estimator for anomaly detection.
    
    This module estimates the normal distribution parameters (mean and covariance)
    of the constrained embeddings and computes anomaly scores based on 
    likelihood under the learned normal distribution.
    """
    
    def __init__(self, embedding_dim, reg_covar=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.reg_covar = reg_covar
        
        # Learnable parameters for normal distribution
        self.register_buffer('mean', torch.zeros(embedding_dim))
        self.register_buffer('covariance', torch.eye(embedding_dim))
        self.register_buffer('inv_covariance', torch.eye(embedding_dim))
        self.register_buffer('log_det_covariance', torch.tensor(0.0))
        
        self.fitted = False
    
    def fit(self, embeddings):
        """
        Fit normal distribution to training embeddings.
        
        Args:
            embeddings: Training embeddings [batch_size, embedding_dim]
        """
        embeddings = embeddings.detach()
        
        # Compute empirical mean and covariance
        mean = torch.mean(embeddings, dim=0)
        centered = embeddings - mean.unsqueeze(0)
        covariance = torch.matmul(centered.T, centered) / (embeddings.shape[0] - 1)
        
        # Regularize covariance matrix
        covariance = covariance + self.reg_covar * torch.eye(
            self.embedding_dim, device=embeddings.device
        )
        
        # Compute inverse and log determinant
        try:
            inv_covariance = torch.inverse(covariance)
            log_det_covariance = torch.logdet(covariance)
        except RuntimeError:
            # Fallback to regularized version
            covariance = covariance + 1e-3 * torch.eye(
                self.embedding_dim, device=embeddings.device
            )
            inv_covariance = torch.inverse(covariance)
            log_det_covariance = torch.logdet(covariance)
        
        # Update buffers
        self.mean.copy_(mean)
        self.covariance.copy_(covariance)
        self.inv_covariance.copy_(inv_covariance)
        self.log_det_covariance.copy_(log_det_covariance)
        
        self.fitted = True
    
    def forward(self, embeddings):
        """
        Compute anomaly scores based on normal distribution.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
        
        Returns:
            Anomaly scores [batch_size] (higher = more anomalous)
        """
        if not self.fitted:
            raise ValueError("Estimator must be fitted before computing scores")
        
        # Center the embeddings
        centered = embeddings - self.mean.unsqueeze(0)
        
        # Compute Mahalanobis distance squared
        mahalanobis_sq = torch.sum(
            centered * torch.matmul(centered, self.inv_covariance), 
            dim=1
        )
        
        # Compute negative log-likelihood (anomaly score)
        log_likelihood = -0.5 * (
            mahalanobis_sq + 
            self.log_det_covariance + 
            self.embedding_dim * np.log(2 * np.pi)
        )
        
        # Return negative log-likelihood as anomaly score
        anomaly_scores = -log_likelihood
        
        return anomaly_scores


class FlowBasedEstimator(nn.Module):
    """
    Flow-based distribution estimator using normalizing flows.
    
    This provides a more flexible alternative to simple Gaussian assumption,
    allowing for more complex normal distributions.
    """
    
    def __init__(self, embedding_dim, num_flows=4, hidden_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_flows = num_flows
        
        if hidden_dim is None:
            hidden_dim = embedding_dim * 2
        
        # Normalizing flow layers
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            flow = CouplingLayer(embedding_dim, hidden_dim)
            self.flows.append(flow)
        
        # Base distribution (standard normal)
        self.register_buffer('base_mean', torch.zeros(embedding_dim))
        self.register_buffer('base_std', torch.ones(embedding_dim))
        
        self.fitted = False
    
    def fit(self, embeddings, epochs=100, lr=1e-3):
        """
        Fit normalizing flow to training embeddings.
        
        Args:
            embeddings: Training embeddings [batch_size, embedding_dim]
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass through flows
            z, log_det_jacobian = self.forward_flow(embeddings)
            
            # Compute log probability under base distribution
            log_prob_base = -0.5 * torch.sum(z**2, dim=1) - \
                           0.5 * self.embedding_dim * np.log(2 * np.pi)
            
            # Total log probability
            log_prob = log_prob_base + log_det_jacobian
            
            # Negative log-likelihood loss
            loss = -torch.mean(log_prob)
            
            loss.backward()
            optimizer.step()
        
        self.eval()
        self.fitted = True
    
    def forward_flow(self, x):
        """Forward pass through normalizing flows."""
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)
        
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_jacobian += log_det
        
        return x, log_det_jacobian
    
    def forward(self, embeddings):
        """Compute anomaly scores using flow-based likelihood."""
        if not self.fitted:
            raise ValueError("Estimator must be fitted before computing scores")
        
        with torch.no_grad():
            # Forward pass through flows
            z, log_det_jacobian = self.forward_flow(embeddings)
            
            # Compute log probability under base distribution
            log_prob_base = -0.5 * torch.sum(z**2, dim=1) - \
                           0.5 * self.embedding_dim * np.log(2 * np.pi)
            
            # Total log probability
            log_prob = log_prob_base + log_det_jacobian
            
            # Return negative log probability as anomaly score
            anomaly_scores = -log_prob
        
        return anomaly_scores


class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flows."""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Split dimension
        self.split_dim = input_dim // 2
        
        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim)
        )
        
        self.translation_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim)
        )
    
    def forward(self, x):
        """Forward pass through coupling layer."""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        scale = self.scale_net(x1)
        translation = self.translation_net(x1)
        
        # Apply affine transformation
        y2 = x2 * torch.exp(scale) + translation
        y = torch.cat([x1, y2], dim=1)
        
        # Log determinant of Jacobian
        log_det_jacobian = torch.sum(scale, dim=1)
        
        return y, log_det_jacobian


class EnsembleEstimator(nn.Module):
    """
    Ensemble of multiple distribution estimators.
    
    Combines multiple estimators to provide more robust anomaly detection.
    """
    
    def __init__(self, embedding_dim, num_estimators=3, estimator_types=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_estimators = num_estimators
        
        if estimator_types is None:
            estimator_types = ['normal', 'normal', 'flow']
        
        # Create ensemble of estimators
        self.estimators = nn.ModuleList()
        for i in range(num_estimators):
            est_type = estimator_types[i % len(estimator_types)]
            
            if est_type == 'normal':
                estimator = NormalDistributionEstimator(embedding_dim)
            elif est_type == 'flow':
                estimator = FlowBasedEstimator(embedding_dim, num_flows=2)
            else:
                raise ValueError(f"Unknown estimator type: {est_type}")
            
            self.estimators.append(estimator)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_estimators))
        
        self.fitted = False
    
    def fit(self, embeddings):
        """Fit all estimators in the ensemble."""
        for estimator in self.estimators:
            if isinstance(estimator, FlowBasedEstimator):
                estimator.fit(embeddings, epochs=50)  # Fewer epochs for ensemble
            else:
                estimator.fit(embeddings)
        
        self.fitted = True
    
    def forward(self, embeddings):
        """Compute ensemble anomaly scores."""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before computing scores")
        
        # Compute scores from each estimator
        scores = []
        for estimator in self.estimators:
            score = estimator(embeddings)
            scores.append(score)
        
        # Stack scores
        scores = torch.stack(scores, dim=1)  # [batch_size, num_estimators]
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_scores = torch.sum(scores * weights.unsqueeze(0), dim=1)
        
        return ensemble_scores