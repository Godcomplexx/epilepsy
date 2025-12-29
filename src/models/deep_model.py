"""
CNN-LSTM model for seizure prediction with Transfer Learning support.

Improvements:
- Focal Loss for class imbalance
- Data Augmentation for EEG
- Artifact Rejection
- Ensemble Learning
- Domain Adaptation (Gradient Reversal Layer)
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from loguru import logger


# =============================================================================
# FOCAL LOSS - Better handling of class imbalance
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses on hard examples by down-weighting easy ones.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (0 = BCE, 2 = recommended)
            pos_weight: Additional positive class weight (for severe imbalance)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (before sigmoid)
            targets: Binary labels (0 or 1)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply pos_weight if provided
        if self.pos_weight is not None:
            weight = self.pos_weight * targets + (1 - targets)
            bce = bce * weight.squeeze()
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


# =============================================================================
# DATA AUGMENTATION for EEG
# =============================================================================

class EEGAugmentation:
    """
    Data augmentation transforms for EEG signals.
    
    Augmentations:
    - Gaussian noise addition
    - Amplitude scaling
    - Time shift
    - Channel dropout
    """
    
    def __init__(
        self,
        noise_std: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        time_shift_max: int = 50,
        channel_dropout_prob: float = 0.1,
        p: float = 0.5  # Probability of applying each augmentation
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to EEG signal.
        
        Args:
            x: EEG tensor of shape (channels, timepoints)
            
        Returns:
            Augmented tensor
        """
        # Gaussian noise
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Amplitude scaling
        if torch.rand(1).item() < self.p:
            scale = torch.empty(1).uniform_(*self.scale_range).item()
            x = x * scale
        
        # Time shift (circular)
        if torch.rand(1).item() < self.p:
            shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)
        
        # Channel dropout (zero out random channels)
        if torch.rand(1).item() < self.p:
            n_channels = x.shape[0]
            dropout_mask = torch.rand(n_channels) > self.channel_dropout_prob
            dropout_mask = dropout_mask.float().unsqueeze(-1)
            x = x * dropout_mask
        
        return x


# =============================================================================
# ARTIFACT REJECTION
# =============================================================================

def reject_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    max_amplitude: float = 200.0,
    flat_threshold: float = 1e-6,
    max_reject_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reject windows with artifacts.
    
    Criteria:
    - Amplitude > max_amplitude (likely electrode pop or movement)
    - Flat-line signal (std < threshold)
    
    Args:
        X: EEG data (n_samples, n_channels, n_timepoints)
        y: Labels
        max_amplitude: Maximum allowed amplitude (μV)
        flat_threshold: Minimum std for non-flat signal
        max_reject_ratio: Maximum ratio of samples to reject
        
    Returns:
        X_clean, y_clean, valid_indices
    """
    n_samples = len(X)
    valid_mask = np.ones(n_samples, dtype=bool)
    
    for i in range(n_samples):
        window = X[i]
        
        # Check max amplitude
        if np.abs(window).max() > max_amplitude:
            valid_mask[i] = False
            continue
        
        # Check for flat-line (per channel)
        channel_stds = np.std(window, axis=1)
        if np.any(channel_stds < flat_threshold):
            valid_mask[i] = False
            continue
    
    n_rejected = n_samples - valid_mask.sum()
    reject_ratio = n_rejected / n_samples
    
    # If too many rejected, relax criteria
    if reject_ratio > max_reject_ratio:
        logger.warning(f"Artifact rejection would remove {reject_ratio:.1%} of data, limiting to {max_reject_ratio:.1%}")
        # Keep the best samples based on amplitude
        amplitudes = np.abs(X).max(axis=(1, 2))
        threshold_idx = int(n_samples * (1 - max_reject_ratio))
        amplitude_threshold = np.sort(amplitudes)[threshold_idx]
        valid_mask = amplitudes <= amplitude_threshold
    
    valid_indices = np.where(valid_mask)[0]
    
    logger.info(f"Artifact rejection: {n_rejected}/{n_samples} windows rejected ({n_rejected/n_samples:.1%})")
    
    return X[valid_mask], y[valid_mask], valid_indices


# =============================================================================
# GRADIENT REVERSAL LAYER for Domain Adaptation
# =============================================================================

class GradientReversalFunction(Function):
    """Gradient Reversal Layer for domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) for domain-adversarial training.
    
    During forward pass: identity function
    During backward pass: reverses gradients and scales by lambda
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class EEGDataset(Dataset):
    """Dataset for EEG windows."""
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        entropy_features: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            labels: Binary labels (0=interictal, 1=preictal)
            entropy_features: Optional entropy features (n_samples, n_entropy_features)
            transform: Optional transform to apply
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.entropy_features = torch.FloatTensor(entropy_features) if entropy_features is not None else None
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.entropy_features is not None:
            return x, self.entropy_features[idx], y
        return x, y


class CNN1DBlock(nn.Module):
    """1D Convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SpatialAttention(nn.Module):
    """Spatial attention module for channel weighting."""
    
    def __init__(self, n_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            nn.ReLU(),
            nn.Linear(n_channels // 2, n_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        # Global average pooling over time
        weights = x.mean(dim=2)  # (batch, channels)
        weights = self.attention(weights)  # (batch, channels)
        weights = weights.unsqueeze(2)  # (batch, channels, 1)
        return x * weights


class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for EEG seizure prediction with optional entropy features
    and domain adaptation.
    
    Architecture:
    - Spatial attention for channel weighting
    - 3 CNN blocks for temporal feature extraction
    - Bidirectional LSTM for sequence modeling
    - Optional entropy features concatenated before FC
    - Fully connected layers for classification
    - Optional domain classifier with Gradient Reversal Layer
    """
    
    def __init__(
        self,
        n_channels: int = 18,
        n_timepoints: int = 1024,
        cnn_filters: List[int] = [32, 64, 128],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
        n_entropy_features: int = 0,  # Number of entropy features (0 = disabled)
        n_domains: int = 0  # Number of domains/patients for domain adaptation (0 = disabled)
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.use_attention = use_attention
        self.n_entropy_features = n_entropy_features
        self.n_domains = n_domains
        
        # Spatial attention
        if use_attention:
            self.spatial_attention = SpatialAttention(n_channels)
        
        # CNN blocks
        self.cnn_blocks = nn.ModuleList()
        in_ch = n_channels
        for out_ch in cnn_filters:
            self.cnn_blocks.append(CNN1DBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate LSTM input size
        lstm_input_size = cnn_filters[-1]
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Feature size after LSTM
        self.feature_size = lstm_hidden * 2  # *2 for bidirectional
        
        # Entropy feature processing (if enabled)
        if n_entropy_features > 0:
            self.entropy_fc = nn.Sequential(
                nn.Linear(n_entropy_features, 32),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fc_input_size = self.feature_size + 32  # LSTM output + entropy features
        else:
            self.entropy_fc = None
            fc_input_size = self.feature_size
        
        # Fully connected layers (task classifier)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )
        
        # Domain classifier with Gradient Reversal Layer (for domain adaptation)
        if n_domains > 0:
            self.grl = GradientReversalLayer(lambda_=1.0)
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.feature_size, fc_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden, n_domains)
            )
        else:
            self.grl = None
            self.domain_classifier = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        entropy_features: Optional[torch.Tensor] = None,
        return_domain: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, timepoints)
            entropy_features: Optional entropy features (batch, n_entropy_features)
            return_domain: If True, also return domain classifier output
            
        Returns:
            Output tensor of shape (batch, 1)
            If return_domain=True: (task_output, domain_output)
        """
        # Spatial attention
        if self.use_attention:
            x = self.spatial_attention(x)
        
        # CNN blocks with pooling
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
            x = self.pool(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last hidden state (shared features)
        features = lstm_out[:, -1, :]
        
        # Domain classification (if enabled and requested)
        domain_output = None
        if return_domain and self.domain_classifier is not None:
            reversed_features = self.grl(features)
            domain_output = self.domain_classifier(reversed_features)
        
        # Concatenate entropy features if provided
        if self.entropy_fc is not None and entropy_features is not None:
            entropy_out = self.entropy_fc(entropy_features)
            task_features = torch.cat([features, entropy_out], dim=1)
        else:
            task_features = features
        
        # Task classifier (seizure prediction)
        task_output = self.fc(task_features)
        
        if return_domain and domain_output is not None:
            return task_output, domain_output
        return task_output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def freeze_cnn(self):
        """Freeze CNN layers for fine-tuning."""
        for block in self.cnn_blocks:
            for param in block.parameters():
                param.requires_grad = False
        if self.use_attention:
            for param in self.spatial_attention.parameters():
                param.requires_grad = False
        logger.info("CNN layers frozen for fine-tuning")
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")


class SeizurePredictorDL:
    """
    Deep Learning seizure predictor with Transfer Learning support.
    """
    
    def __init__(
        self,
        n_channels: int = 18,
        n_timepoints: int = 1024,
        device: str = 'auto',
        n_entropy_features: int = 0,
        **model_kwargs
    ):
        """
        Args:
            n_channels: Number of EEG channels
            n_timepoints: Number of time points per window
            device: 'cuda', 'cpu', or 'auto'
            n_entropy_features: Number of entropy features (0 = disabled)
            **model_kwargs: Additional arguments for CNNLSTM
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.use_entropy = False  # Will be set in fit()
        
        self.model = CNNLSTM(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_entropy_features=n_entropy_features,
            **model_kwargs
        ).to(self.device)
        
        logger.info(f"Initialized CNNLSTM on {self.device}" + (f" with {n_entropy_features} entropy features" if n_entropy_features > 0 else ""))
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weight: Optional[float] = None,
        val_split: float = 0.1,
        early_stopping: int = 10,
        verbose: bool = True,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        entropy_features: Optional[np.ndarray] = None,
        use_focal_loss: bool = True,
        use_augmentation: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X: Training data (n_samples, n_channels, n_timepoints)
            y: Labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            class_weight: Weight for positive class (for imbalanced data)
            val_split: Validation split ratio (used if train/val_indices not provided)
            early_stopping: Patience for early stopping
            verbose: Print progress
            train_indices: Pre-computed train indices (for file-based split)
            val_indices: Pre-computed val indices (for file-based split)
            entropy_features: Optional entropy features (n_samples, n_entropy_features)
            use_focal_loss: Use Focal Loss instead of BCE (better for imbalanced data)
            use_augmentation: Apply data augmentation during training
            focal_gamma: Focal loss gamma parameter (focusing strength)
            focal_alpha: Focal loss alpha parameter (class balance)
            
        Returns:
            Training history
        """
        self.use_entropy = entropy_features is not None
        
        # Data augmentation transform (only for training)
        augmentation = EEGAugmentation(
            noise_std=0.1,
            scale_range=(0.9, 1.1),
            time_shift_max=25,
            channel_dropout_prob=0.1,
            p=0.5
        ) if use_augmentation else None
        
        # Split data - use provided indices or random split
        if train_indices is None or val_indices is None:
            n_val = int(len(y) * val_split)
            indices = np.random.permutation(len(y))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
        
        # Create datasets with or without entropy
        if self.use_entropy:
            train_dataset = EEGDataset(X[train_indices], y[train_indices], entropy_features[train_indices], transform=augmentation)
            val_dataset = EEGDataset(X[val_indices], y[val_indices], entropy_features[val_indices])
        else:
            train_dataset = EEGDataset(X[train_indices], y[train_indices], transform=augmentation)
            val_dataset = EEGDataset(X[val_indices], y[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss function with class weighting
        if class_weight is None:
            # Auto-compute based on class imbalance
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            class_weight = min(n_neg / n_pos if n_pos > 0 else 1.0, 10.0)  # Cap at 10
        
        pos_weight = torch.tensor([class_weight]).to(self.device)
        
        # Use Focal Loss or BCE
        if use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                if self.use_entropy:
                    batch_x, batch_entropy, batch_y = batch
                    batch_entropy = batch_entropy.to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_entropy = None
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_entropy)
                loss = criterion(outputs.squeeze(-1), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if self.use_entropy:
                        batch_x, batch_entropy, batch_y = batch
                        batch_entropy = batch_entropy.to(self.device)
                    else:
                        batch_x, batch_y = batch
                        batch_entropy = None
                    
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x, batch_entropy)
                    loss = criterion(outputs.squeeze(-1), batch_y)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs.squeeze(-1))
                    val_preds.extend(probs.cpu().numpy().flatten())
                    val_labels.extend(batch_y.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            
            # Compute AUC
            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except:
                val_auc = 0.5
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val AUC: {val_auc:.4f}"
                )
            
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        return history
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 64, entropy_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Data (n_samples, n_channels, n_timepoints)
            batch_size: Batch size for inference
            entropy_features: Optional entropy features (n_samples, n_entropy_features)
            
        Returns:
            Probabilities (n_samples,)
        """
        self.model.eval()
        use_entropy = entropy_features is not None and self.model.n_entropy_features > 0
        
        if use_entropy:
            dataset = EEGDataset(X, np.zeros(len(X)), entropy_features)
        else:
            dataset = EEGDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        probs = []
        with torch.no_grad():
            for batch in loader:
                if use_entropy:
                    batch_x, batch_entropy, _ = batch
                    batch_entropy = batch_entropy.to(self.device)
                else:
                    batch_x, _ = batch
                    batch_entropy = None
                
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x, batch_entropy)
                batch_probs = torch.sigmoid(outputs.squeeze())
                probs.extend(batch_probs.cpu().numpy())
        
        return np.array(probs)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def save(self, path: Union[str, Path]):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_channels': self.n_channels,
            'n_timepoints': self.n_timepoints,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load model from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def freeze_pretrained(self):
        """Freeze CNN layers for fine-tuning."""
        self.model.freeze_cnn()
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        self.model.unfreeze_all()


def pretrain_global_model(
    patient_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_channels: int = 18,
    n_timepoints: int = 1024,
    epochs: int = 30,
    batch_size: int = 64,
    save_path: Optional[Path] = None,
    device: str = 'auto',
    n_entropy_features: int = 0,
    patient_entropy: Optional[Dict[str, np.ndarray]] = None
) -> SeizurePredictorDL:
    """
    Pretrain a global model on all patients.
    
    Args:
        patient_data: Dict of patient_id -> (X, y) tuples
        n_channels: Number of EEG channels
        n_timepoints: Number of time points
        epochs: Training epochs
        batch_size: Batch size
        save_path: Path to save pretrained model
        device: Device to use
        n_entropy_features: Number of entropy features (0 = disabled)
        patient_entropy: Dict of patient_id -> entropy_features arrays
        
    Returns:
        Pretrained model
    """
    logger.info(f"Pretraining global model on {len(patient_data)} patients")
    
    # Combine all patient data, normalizing channel count
    all_X = []
    all_y = []
    
    for patient_id, (X, y) in patient_data.items():
        patient_channels = X.shape[1]
        
        # Pad or truncate channels to match expected n_channels
        if patient_channels < n_channels:
            # Pad with zeros
            padding = np.zeros((X.shape[0], n_channels - patient_channels, X.shape[2]), dtype=X.dtype)
            X = np.concatenate([X, padding], axis=1)
            logger.info(f"  {patient_id}: {len(y)} samples, {y.sum():.0f} preictal (padded {patient_channels}→{n_channels} channels)")
        elif patient_channels > n_channels:
            # Truncate
            X = X[:, :n_channels, :]
            logger.info(f"  {patient_id}: {len(y)} samples, {y.sum():.0f} preictal (truncated {patient_channels}→{n_channels} channels)")
        else:
            logger.info(f"  {patient_id}: {len(y)} samples, {y.sum():.0f} preictal")
        
        all_X.append(X)
        all_y.append(y)
    
    # Subsample BEFORE concatenation to save memory
    max_samples = 50000  # ~3 GB - balanced for memory and performance
    
    # First, compute total and decide on sampling ratio
    total_samples = sum(len(y) for _, (X, y) in enumerate(patient_data.values()))
    total_preictal = sum(y.sum() for _, (X, y) in enumerate(patient_data.values()))
    
    logger.info(f"Total before subsampling: {total_samples} samples, {total_preictal:.0f} preictal")
    
    if total_samples > max_samples:
        sample_ratio = max_samples / total_samples
        logger.info(f"Subsampling to {max_samples} samples (ratio: {sample_ratio:.3f})")
        
        # Subsample each patient's data proportionally
        sampled_X = []
        sampled_y = []
        
        for patient_id, (X, y) in patient_data.items():
            n_patient = len(y)
            n_sample = max(1, int(n_patient * sample_ratio))
            
            # Stratified sampling within patient
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)
            
            if n_pos > 0:
                n_pos_sample = max(1, int(n_sample * (n_pos / n_patient)))
                n_neg_sample = n_sample - n_pos_sample
            else:
                n_pos_sample = 0
                n_neg_sample = n_sample
            
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            
            selected = np.concatenate([
                pos_idx[:n_pos_sample],
                neg_idx[:n_neg_sample]
            ])
            
            # Normalize channels
            patient_channels = X.shape[1]
            if patient_channels < n_channels:
                padding = np.zeros((X.shape[0], n_channels - patient_channels, X.shape[2]), dtype=X.dtype)
                X = np.concatenate([X, padding], axis=1)
            elif patient_channels > n_channels:
                X = X[:, :n_channels, :]
            
            sampled_X.append(X[selected])
            sampled_y.append(y[selected])
        
        X_combined = np.concatenate(sampled_X, axis=0)
        y_combined = np.concatenate(sampled_y, axis=0)
        
        # Free memory
        del sampled_X, sampled_y
    else:
        # Normalize and concatenate all
        normalized_X = []
        for patient_id, (X, y) in patient_data.items():
            patient_channels = X.shape[1]
            if patient_channels < n_channels:
                padding = np.zeros((X.shape[0], n_channels - patient_channels, X.shape[2]), dtype=X.dtype)
                X = np.concatenate([X, padding], axis=1)
            elif patient_channels > n_channels:
                X = X[:, :n_channels, :]
            normalized_X.append(X)
        
        X_combined = np.concatenate(normalized_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        del normalized_X
    
    logger.info(f"After processing: {len(y_combined)} samples, {y_combined.sum():.0f} preictal")
    
    # Create and train model
    model = SeizurePredictorDL(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        device=device
    )
    
    history = model.fit(
        X_combined, y_combined,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Save if path provided
    if save_path:
        model.save(save_path)
    
    return model


def finetune_patient_model(
    pretrained_model: SeizurePredictorDL,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    freeze_cnn: bool = True,
    windows_df: Optional[pd.DataFrame] = None,
    entropy_features: Optional[np.ndarray] = None,
    adaptive_lr: bool = True,
    adaptive_epochs: bool = True,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0
) -> Dict[str, List[float]]:
    """
    Fine-tune pretrained model on patient-specific data with adaptive parameters.
    
    Args:
        pretrained_model: Pretrained global model
        X: Patient data (n_samples, n_channels, n_timepoints)
        y: Labels
        epochs: Base fine-tuning epochs (may be adjusted if adaptive_epochs=True)
        batch_size: Batch size
        lr: Base learning rate (may be adjusted if adaptive_lr=True)
        freeze_cnn: Whether to freeze CNN layers
        windows_df: Window metadata with 'edf_file' column for file-based split
        entropy_features: Optional entropy features (n_samples, n_entropy_features)
        adaptive_lr: Adjust learning rate based on data characteristics
        adaptive_epochs: Adjust epochs based on data size and complexity
        
    Returns:
        Training history
    """
    if freeze_cnn:
        pretrained_model.freeze_pretrained()
    
    # Patient-Adaptive Learning Rate
    if adaptive_lr:
        n_samples = len(y)
        n_preictal = y.sum()
        class_ratio = n_preictal / n_samples if n_samples > 0 else 0
        
        # Adjust LR based on data characteristics:
        # - More data -> lower LR (more stable gradients)
        # - Higher class imbalance -> higher LR (need stronger signal)
        if n_samples < 10000:
            lr_multiplier = 2.0  # Small dataset - higher LR
        elif n_samples < 30000:
            lr_multiplier = 1.0  # Medium dataset
        else:
            lr_multiplier = 0.5  # Large dataset - lower LR
        
        # Adjust for class imbalance
        if class_ratio < 0.05:
            lr_multiplier *= 1.5  # Very imbalanced - boost LR
        
        lr = lr * lr_multiplier
        logger.info(f"  Adaptive LR: {lr:.6f} (multiplier: {lr_multiplier:.1f})")
    
    # Adaptive Epochs based on data size and complexity
    if adaptive_epochs:
        n_samples = len(y)
        n_preictal = y.sum()
        
        # Base epochs adjusted by data size
        if n_samples < 10000:
            epochs = max(epochs, 30)  # Small dataset - more epochs
        elif n_samples < 30000:
            epochs = epochs  # Medium - keep default
        else:
            epochs = min(epochs, 15)  # Large dataset - fewer epochs (faster convergence)
        
        # If very few preictal samples, increase epochs
        if n_preictal < 500:
            epochs = int(epochs * 1.5)
        
        logger.info(f"  Adaptive epochs: {epochs}")
    
    # If windows_df provided, do file-based split ensuring seizure files in val
    train_indices = None
    val_indices = None
    
    if windows_df is not None and 'edf_file' in windows_df.columns:
        train_indices, val_indices = split_by_files_with_seizures(
            windows_df, y, val_ratio=0.2
        )
        logger.info(f"  File-based split: {len(train_indices)} train, {len(val_indices)} val")
        
        # If val_indices is empty, fall back to random split
        if len(val_indices) == 0:
            logger.warning("  No validation data from file split, using random 20% split")
            train_indices = None
            val_indices = None
    
    history = pretrained_model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
        train_indices=train_indices,
        val_indices=val_indices,
        entropy_features=entropy_features,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    return history


def split_by_files_with_seizures(
    windows_df: pd.DataFrame,
    y: np.ndarray,
    val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data by files, ensuring at least one seizure file goes to validation.
    
    Args:
        windows_df: DataFrame with 'edf_file' and window info
        y: Labels (1=preictal, 0=interictal)
        val_ratio: Ratio of files for validation
        
    Returns:
        train_indices, val_indices
    """
    # Get unique files
    files = windows_df['edf_file'].unique()
    
    # Find files that have preictal windows (seizure files)
    seizure_files = []
    non_seizure_files = []
    
    for f in files:
        file_mask = windows_df['edf_file'] == f
        file_indices = np.where(file_mask)[0]
        if y[file_indices].sum() > 0:  # Has preictal windows
            seizure_files.append(f)
        else:
            non_seizure_files.append(f)
    
    # Ensure at least 1 seizure file in validation
    n_val_seizure = max(1, int(len(seizure_files) * val_ratio))
    n_val_non_seizure = max(0, int(len(non_seizure_files) * val_ratio))
    
    # Shuffle and split
    np.random.shuffle(seizure_files)
    np.random.shuffle(non_seizure_files)
    
    val_seizure_files = seizure_files[:n_val_seizure]
    train_seizure_files = seizure_files[n_val_seizure:]
    
    val_non_seizure_files = non_seizure_files[:n_val_non_seizure]
    train_non_seizure_files = non_seizure_files[n_val_non_seizure:]
    
    val_files = set(val_seizure_files + val_non_seizure_files)
    train_files = set(train_seizure_files + train_non_seizure_files)
    
    # Get indices (use positional indices, not DataFrame index)
    train_indices = []
    val_indices = []
    
    for i, (idx, row) in enumerate(windows_df.iterrows()):
        if row['edf_file'] in val_files:
            val_indices.append(i)
        else:
            train_indices.append(i)
    
    logger.info(f"  Split: {len(train_files)} train files ({len(train_seizure_files)} with seizures), "
                f"{len(val_files)} val files ({len(val_seizure_files)} with seizures)")
    
    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold for a patient.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        metric: Metric to optimize ('f1', 'sensitivity', 'balanced')
        
    Returns:
        optimal_threshold, metrics_dict
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    best_threshold = 0.5
    best_score = 0
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'sensitivity':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            sens = recall_score(y_true, y_pred, zero_division=0)
            spec = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
            score = (sens + spec) / 2
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    # Compute final metrics at optimal threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    metrics = {
        'threshold': best_threshold,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0)
    }
    
    return best_threshold, metrics


def ensemble_predict(
    models: List[SeizurePredictorDL],
    X: np.ndarray,
    entropy_features: Optional[np.ndarray] = None,
    method: str = 'mean'
) -> np.ndarray:
    """
    Ensemble prediction from multiple models.
    
    Args:
        models: List of trained models
        X: Input data
        entropy_features: Optional entropy features
        method: Aggregation method ('mean', 'max', 'vote')
        
    Returns:
        Ensemble probabilities
    """
    all_probs = []
    
    for model in models:
        probs = model.predict_proba(X, entropy_features=entropy_features)
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)  # (n_models, n_samples)
    
    if method == 'mean':
        return np.mean(all_probs, axis=0)
    elif method == 'max':
        return np.max(all_probs, axis=0)
    elif method == 'vote':
        # Majority voting at 0.5 threshold
        votes = (all_probs >= 0.5).astype(int)
        return np.mean(votes, axis=0)
    else:
        return np.mean(all_probs, axis=0)


def train_ensemble_models(
    X: np.ndarray,
    y: np.ndarray,
    n_models: int = 3,
    n_channels: int = 18,
    n_timepoints: int = 1024,
    n_entropy_features: int = 0,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    entropy_features: Optional[np.ndarray] = None
) -> List[SeizurePredictorDL]:
    """
    Train ensemble of models with different random seeds and data subsets.
    
    Args:
        X: Training data
        y: Labels
        n_models: Number of models in ensemble
        n_channels: Number of EEG channels
        n_timepoints: Number of time points
        n_entropy_features: Number of entropy features
        epochs: Training epochs per model
        batch_size: Batch size
        lr: Learning rate
        entropy_features: Optional entropy features
        
    Returns:
        List of trained models
    """
    models = []
    
    for i in range(n_models):
        logger.info(f"  Training ensemble model {i+1}/{n_models}")
        
        # Different random seed for each model
        np.random.seed(42 + i)
        torch.manual_seed(42 + i)
        
        # Bootstrap sampling (sample with replacement)
        n_samples = len(y)
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_boot = X[bootstrap_idx]
        y_boot = y[bootstrap_idx]
        entropy_boot = entropy_features[bootstrap_idx] if entropy_features is not None else None
        
        # Create and train model
        model = SeizurePredictorDL(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_entropy_features=n_entropy_features
        )
        
        model.fit(
            X_boot, y_boot,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=False,
            entropy_features=entropy_boot
        )
        
        models.append(model)
    
    return models


# =============================================================================
# META-LEARNING (MAML) for fast adaptation
# =============================================================================

class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) trainer for seizure prediction.
    
    MAML learns an initialization that can quickly adapt to new patients
    with just a few gradient steps.
    
    Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
    of Deep Networks", ICML 2017
    """
    
    def __init__(
        self,
        model: SeizurePredictorDL,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        device: str = 'auto'
    ):
        """
        Args:
            model: Base model to meta-train
            inner_lr: Learning rate for inner loop (task-specific adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            inner_steps: Number of gradient steps in inner loop
            device: Device to use
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Meta-optimizer (updates the initialization)
        self.meta_optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=outer_lr
        )
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform inner loop adaptation on support set and evaluate on query set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            
        Returns:
            Query loss after adaptation
        """
        # Clone model parameters for this task
        fast_weights = {
            name: param.clone() 
            for name, param in self.model.model.named_parameters()
        }
        
        # Inner loop: adapt to support set
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        for _ in range(self.inner_steps):
            # Forward pass with fast weights
            outputs = self._forward_with_weights(support_x, fast_weights)
            loss = criterion(outputs.squeeze(-1), support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            # Update fast weights
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        
        # Evaluate on query set with adapted weights
        query_outputs = self._forward_with_weights(query_x, fast_weights)
        query_loss = criterion(query_outputs.squeeze(-1), query_y)
        
        return query_loss
    
    def _forward_with_weights(
        self,
        x: torch.Tensor,
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using specified weights (for MAML inner loop)."""
        # This is a simplified version - for full implementation,
        # we would need to use functional forward pass
        # For now, we temporarily set the weights
        original_weights = {}
        for name, param in self.model.model.named_parameters():
            original_weights[name] = param.data.clone()
            param.data = weights[name]
        
        output = self.model.model(x)
        
        # Restore original weights
        for name, param in self.model.model.named_parameters():
            param.data = original_weights[name]
        
        return output
    
    def meta_train(
        self,
        patient_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        n_epochs: int = 100,
        tasks_per_batch: int = 4,
        support_size: int = 32,
        query_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Meta-train the model on multiple patients.
        
        Args:
            patient_data: Dict of patient_id -> (X, y) tuples
            n_epochs: Number of meta-training epochs
            tasks_per_batch: Number of tasks (patients) per meta-batch
            support_size: Size of support set per task
            query_size: Size of query set per task
            
        Returns:
            Training history
        """
        logger.info(f"Starting MAML meta-training on {len(patient_data)} patients")
        
        patient_ids = list(patient_data.keys())
        history = {'meta_loss': []}
        
        for epoch in range(n_epochs):
            # Sample tasks (patients) for this batch
            batch_patients = np.random.choice(
                patient_ids, 
                size=min(tasks_per_batch, len(patient_ids)),
                replace=False
            )
            
            meta_loss = 0.0
            
            for patient_id in batch_patients:
                X, y = patient_data[patient_id]
                
                # Sample support and query sets
                n_samples = len(y)
                indices = np.random.permutation(n_samples)
                
                support_idx = indices[:support_size]
                query_idx = indices[support_size:support_size + query_size]
                
                support_x = torch.FloatTensor(X[support_idx]).to(self.device)
                support_y = torch.FloatTensor(y[support_idx]).to(self.device)
                query_x = torch.FloatTensor(X[query_idx]).to(self.device)
                query_y = torch.FloatTensor(y[query_idx]).to(self.device)
                
                # Inner loop adaptation and query evaluation
                task_loss = self.inner_loop(support_x, support_y, query_x, query_y)
                meta_loss += task_loss
            
            # Meta-update (outer loop)
            meta_loss = meta_loss / len(batch_patients)
            
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            self.meta_optimizer.step()
            
            history['meta_loss'].append(meta_loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"MAML Epoch {epoch+1}/{n_epochs} - Meta Loss: {meta_loss.item():.4f}")
        
        return history
    
    def adapt_to_patient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_steps: int = 10
    ) -> SeizurePredictorDL:
        """
        Quickly adapt the meta-learned model to a new patient.
        
        Args:
            X: Patient data
            y: Patient labels
            n_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        # Create a copy of the model
        import copy
        adapted_model = copy.deepcopy(self.model)
        
        # Fine-tune with few steps
        adapted_model.fit(
            X, y,
            epochs=n_steps,
            lr=self.inner_lr,
            early_stopping=n_steps,  # No early stopping
            verbose=False
        )
        
        return adapted_model


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_channels = 18
    n_timepoints = 1024
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    # Create model
    model = SeizurePredictorDL(
        n_channels=n_channels,
        n_timepoints=n_timepoints
    )
    
    # Train
    history = model.fit(X, y, epochs=5, batch_size=32, verbose=True)
    
    # Predict
    probs = model.predict_proba(X[:10])
    print(f"Predictions: {probs}")
    
    # Save and load
    model.save("test_model.pt")
    model.load("test_model.pt")
    
    print("Test completed successfully!")
