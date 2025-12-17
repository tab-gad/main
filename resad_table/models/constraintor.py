import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularFeatureConstraintor(nn.Module):
    """
    Feature constraintor for tabular data embeddings.
    
    This module learns to constrain/transform the TabPFN embeddings
    to better separate normal and anomalous patterns in the embedding space.
    """
    
    def __init__(self, embedding_dim, hidden_dims=None, dropout_rate=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        if hidden_dims is None:
            hidden_dims = [embedding_dim // 2, embedding_dim // 4, embedding_dim // 2]
        
        # Build constraint network
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer to map back to embedding dimension
        layers.append(nn.Linear(input_dim, embedding_dim))
        
        self.constraintor = nn.Sequential(*layers)
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, embeddings):
        """
        Apply feature constraints to embeddings.
        
        Args:
            embeddings: Input TabPFN embeddings [batch_size, embedding_dim]
        
        Returns:
            Constrained embeddings [batch_size, embedding_dim]
        """
        constrained = self.constraintor(embeddings)
        
        # Residual connection with learnable weight
        output = self.residual_weight * constrained + (1 - self.residual_weight) * embeddings
        
        return output


class MultiScaleConstraintor(nn.Module):
    """
    Multi-scale feature constraintor for handling different embedding scales.
    
    This handles the multi-scale embeddings from MultiScaleTabPFNExtractor
    and applies appropriate constraints to each scale.
    """
    
    def __init__(self, embedding_dims, shared_hidden_ratio=0.5):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.n_scales = len(embedding_dims)
        
        # Individual constraintors for each scale
        self.constraintors = nn.ModuleList()
        for embedding_dim in embedding_dims:
            hidden_dims = [
                int(embedding_dim * shared_hidden_ratio),
                int(embedding_dim * shared_hidden_ratio // 2),
                int(embedding_dim * shared_hidden_ratio)
            ]
            constraintor = TabularFeatureConstraintor(
                embedding_dim, 
                hidden_dims=hidden_dims
            )
            self.constraintors.append(constraintor)
        
        # Cross-scale attention mechanism
        total_dim = sum(embedding_dims)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final projection layers
        self.final_projections = nn.ModuleList()
        for embedding_dim in embedding_dims:
            self.final_projections.append(
                nn.Linear(total_dim, embedding_dim)
            )
    
    def forward(self, multi_scale_embeddings):
        """
        Apply multi-scale feature constraints.
        
        Args:
            multi_scale_embeddings: List of embeddings for different scales
        
        Returns:
            List of constrained embeddings for each scale
        """
        # Apply individual constraints
        constrained_embeddings = []
        for i, embeddings in enumerate(multi_scale_embeddings):
            constrained = self.constraintors[i](embeddings)
            constrained_embeddings.append(constrained)
        
        # Cross-scale attention
        # Concatenate all scales
        concatenated = torch.cat(constrained_embeddings, dim=1)  # [batch_size, total_dim]
        concatenated = concatenated.unsqueeze(1)  # Add sequence dimension for attention
        
        # Self-attention across concatenated features
        attended, _ = self.cross_attention(
            concatenated, concatenated, concatenated
        )
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Project back to individual scales
        final_embeddings = []
        for i, projection in enumerate(self.final_projections):
            final_emb = projection(attended)
            final_embeddings.append(final_emb)
        
        return final_embeddings


class AdaptiveConstraintor(nn.Module):
    """
    Adaptive feature constraintor that adjusts based on input characteristics.
    
    This module adapts its constraint strength based on the input embeddings,
    providing stronger constraints for more uncertain samples.
    """
    
    def __init__(self, embedding_dim, num_adaptation_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Adaptation network to compute constraint weights
        adaptation_layers = []
        input_dim = embedding_dim
        
        for _ in range(num_adaptation_layers):
            hidden_dim = input_dim // 2
            adaptation_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            input_dim = hidden_dim
        
        # Output adaptation weights
        adaptation_layers.append(nn.Linear(input_dim, 1))
        adaptation_layers.append(nn.Sigmoid())
        
        self.adaptation_net = nn.Sequential(*adaptation_layers)
        
        # Base constraintor
        self.base_constraintor = TabularFeatureConstraintor(embedding_dim)
    
    def forward(self, embeddings):
        """
        Apply adaptive feature constraints.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
        
        Returns:
            Adaptively constrained embeddings [batch_size, embedding_dim]
        """
        # Compute adaptation weights for each sample
        adaptation_weights = self.adaptation_net(embeddings)  # [batch_size, 1]
        
        # Apply base constraints
        constrained = self.base_constraintor(embeddings)
        
        # Adaptive mixing between original and constrained
        output = adaptation_weights * constrained + (1 - adaptation_weights) * embeddings
        
        return output


class ResidualConstraintor(nn.Module):
    """
    Residual-based constraintor inspired by ResAD architecture.
    
    This module learns residual transformations that are added to
    the original embeddings to highlight anomalous patterns.
    """
    
    def __init__(self, embedding_dim, num_residual_blocks=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            block = ResidualBlock(embedding_dim)
            self.residual_blocks.append(block)
        
        # Final residual projection
        self.final_residual = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, embeddings):
        """
        Apply residual constraints.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
        
        Returns:
            Residual-constrained embeddings [batch_size, embedding_dim]
        """
        x = embeddings
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Compute final residual
        residual = self.final_residual(x)
        
        # Add residual to original embeddings
        output = embeddings + residual
        
        return output


class ResidualBlock(nn.Module):
    """Individual residual block for the ResidualConstraintor."""
    
    def __init__(self, embedding_dim, hidden_ratio=0.5):
        super().__init__()
        hidden_dim = int(embedding_dim * hidden_ratio)
        
        self.block = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = self.block(x)
        output = self.activation(x + residual)
        return output