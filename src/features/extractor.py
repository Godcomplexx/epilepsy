"""
Module for feature extraction from EEG windows.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats
from loguru import logger


def compute_psd_features(
    data: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Compute Power Spectral Density features for frequency bands.
    
    Args:
        data: Window data (n_channels, n_samples)
        sfreq: Sampling frequency
        bands: Dictionary of band_name -> (low_freq, high_freq)
        
    Returns:
        Feature array of shape (n_channels * n_bands,)
    """
    n_channels = data.shape[0]
    n_bands = len(bands)
    features = np.zeros(n_channels * n_bands)
    
    # Compute PSD using Welch's method
    nperseg = min(256, data.shape[1])
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=-1)
    
    # Extract band powers
    for ch_idx in range(n_channels):
        for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
            # Find frequency indices
            freq_mask = (freqs >= low) & (freqs <= high)
            
            if freq_mask.sum() > 0:
                band_power = np.mean(psd[ch_idx, freq_mask])
            else:
                band_power = 0.0
            
            features[ch_idx * n_bands + band_idx] = band_power
    
    return features


def compute_relative_band_power(
    data: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Compute relative band power (normalized by total power).
    
    Args:
        data: Window data (n_channels, n_samples)
        sfreq: Sampling frequency
        bands: Dictionary of band_name -> (low_freq, high_freq)
        
    Returns:
        Feature array of shape (n_channels * n_bands,)
    """
    n_channels = data.shape[0]
    n_bands = len(bands)
    features = np.zeros(n_channels * n_bands)
    
    nperseg = min(256, data.shape[1])
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=-1)
    
    for ch_idx in range(n_channels):
        total_power = np.sum(psd[ch_idx])
        
        if total_power < 1e-10:
            continue
        
        for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
            freq_mask = (freqs >= low) & (freqs <= high)
            
            if freq_mask.sum() > 0:
                band_power = np.sum(psd[ch_idx, freq_mask])
                relative_power = band_power / total_power
            else:
                relative_power = 0.0
            
            features[ch_idx * n_bands + band_idx] = relative_power
    
    return features


def compute_statistical_features(
    data: np.ndarray,
    feature_list: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute statistical features for each channel.
    
    Args:
        data: Window data (n_channels, n_samples)
        feature_list: List of features to compute. Options:
            - mean, std, var, skewness, kurtosis
            - min, max, ptp (peak-to-peak)
            - line_length, energy, zero_crossings
            
    Returns:
        Feature array of shape (n_channels * n_features,)
    """
    if feature_list is None:
        feature_list = ['mean', 'std', 'skewness', 'kurtosis', 'line_length', 'energy']
    
    n_channels = data.shape[0]
    n_features = len(feature_list)
    features = np.zeros(n_channels * n_features)
    
    for ch_idx in range(n_channels):
        ch_data = data[ch_idx]
        
        for feat_idx, feat_name in enumerate(feature_list):
            value = _compute_single_stat(ch_data, feat_name)
            features[ch_idx * n_features + feat_idx] = value
    
    return features


def _compute_single_stat(data: np.ndarray, feat_name: str) -> float:
    """Compute a single statistical feature."""
    if feat_name == 'mean':
        return np.mean(data)
    elif feat_name == 'std':
        return np.std(data)
    elif feat_name == 'var':
        return np.var(data)
    elif feat_name == 'skewness':
        return stats.skew(data)
    elif feat_name == 'kurtosis':
        return stats.kurtosis(data)
    elif feat_name == 'min':
        return np.min(data)
    elif feat_name == 'max':
        return np.max(data)
    elif feat_name == 'ptp':
        return np.ptp(data)
    elif feat_name == 'line_length':
        return np.sum(np.abs(np.diff(data)))
    elif feat_name == 'energy':
        return np.sum(data ** 2)
    elif feat_name == 'zero_crossings':
        return np.sum(np.diff(np.sign(data)) != 0)
    elif feat_name == 'rms':
        return np.sqrt(np.mean(data ** 2))
    else:
        logger.warning(f"Unknown feature: {feat_name}")
        return 0.0


def compute_sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Sample Entropy (SampEn) for a 1D signal.
    
    Args:
        data: 1D signal array
        m: Embedding dimension
        r: Tolerance (fraction of std)
        
    Returns:
        Sample entropy value
    """
    N = len(data)
    if N < m + 2:
        return 0.0
    
    r_val = r * np.std(data)
    if r_val < 1e-10:
        return 0.0
    
    def _count_matches(template_len):
        count = 0
        templates = np.array([data[i:i + template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r_val:
                    count += 1
        return count
    
    A = _count_matches(m + 1)
    B = _count_matches(m)
    
    if B == 0 or A == 0:
        return 0.0
    
    return -np.log(A / B)


def compute_permutation_entropy(data: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Compute Permutation Entropy (PermEn) for a 1D signal.
    
    Args:
        data: 1D signal array
        order: Embedding dimension (order of permutation)
        delay: Time delay
        
    Returns:
        Normalized permutation entropy (0 to 1)
    """
    from math import factorial
    
    N = len(data)
    if N < order * delay:
        return 0.0
    
    # Create embedded vectors
    n_patterns = N - (order - 1) * delay
    patterns = np.zeros((n_patterns, order))
    
    for i in range(n_patterns):
        patterns[i] = data[i:i + order * delay:delay]
    
    # Get permutation patterns (argsort gives the ranking)
    perms = np.array([tuple(np.argsort(p)) for p in patterns])
    
    # Count unique permutations
    unique, counts = np.unique([str(p) for p in perms], return_counts=True)
    
    # Compute probabilities
    probs = counts / n_patterns
    
    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalize by maximum entropy
    max_entropy = np.log2(factorial(order))
    
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def compute_entropy_features(data: np.ndarray, sfreq: float = 256.0) -> np.ndarray:
    """
    Compute entropy features for each channel.
    
    Args:
        data: Window data (n_channels, n_samples)
        sfreq: Sampling frequency
        
    Returns:
        Feature array of shape (n_channels * 2,) - SampEn and PermEn per channel
    """
    n_channels = data.shape[0]
    features = np.zeros(n_channels * 2)
    
    for ch_idx in range(n_channels):
        ch_data = data[ch_idx]
        
        # Downsample for faster computation (every 4th sample)
        ch_downsampled = ch_data[::4] if len(ch_data) > 256 else ch_data
        
        # Sample Entropy
        sampen = compute_sample_entropy(ch_downsampled, m=2, r=0.2)
        features[ch_idx * 2] = sampen
        
        # Permutation Entropy
        permen = compute_permutation_entropy(ch_downsampled, order=3, delay=1)
        features[ch_idx * 2 + 1] = permen
    
    return features


def compute_hjorth_parameters(data: np.ndarray) -> np.ndarray:
    """
    Compute Hjorth parameters (Activity, Mobility, Complexity).
    
    Args:
        data: Window data (n_channels, n_samples)
        
    Returns:
        Feature array of shape (n_channels * 3,)
    """
    n_channels = data.shape[0]
    features = np.zeros(n_channels * 3)
    
    for ch_idx in range(n_channels):
        ch_data = data[ch_idx]
        
        # First derivative
        d1 = np.diff(ch_data)
        # Second derivative
        d2 = np.diff(d1)
        
        # Activity (variance)
        activity = np.var(ch_data)
        
        # Mobility
        if activity > 1e-10:
            mobility = np.sqrt(np.var(d1) / activity)
        else:
            mobility = 0.0
        
        # Complexity
        if np.var(d1) > 1e-10:
            complexity = np.sqrt(np.var(d2) / np.var(d1)) / mobility if mobility > 1e-10 else 0.0
        else:
            complexity = 0.0
        
        features[ch_idx * 3] = activity
        features[ch_idx * 3 + 1] = mobility
        features[ch_idx * 3 + 2] = complexity
    
    return features


def extract_features(
    data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    statistical_features: Optional[List[str]] = None,
    include_hjorth: bool = True,
    include_relative_power: bool = True,
    include_entropy: bool = False
) -> np.ndarray:
    """
    Extract all features from a window.
    
    Args:
        data: Window data (n_channels, n_samples)
        sfreq: Sampling frequency
        bands: Frequency bands for PSD features
        statistical_features: List of statistical features
        include_hjorth: Whether to include Hjorth parameters
        include_relative_power: Whether to include relative band power
        include_entropy: Whether to include entropy features (SampEn, PermEn)
        
    Returns:
        Feature vector
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    if statistical_features is None:
        statistical_features = ['mean', 'std', 'skewness', 'kurtosis', 'line_length', 'energy']
    
    feature_parts = []
    
    # PSD features (absolute band power)
    psd_features = compute_psd_features(data, sfreq, bands)
    feature_parts.append(psd_features)
    
    # Relative band power
    if include_relative_power:
        rel_power = compute_relative_band_power(data, sfreq, bands)
        feature_parts.append(rel_power)
    
    # Statistical features
    stat_features = compute_statistical_features(data, statistical_features)
    feature_parts.append(stat_features)
    
    # Hjorth parameters
    if include_hjorth:
        hjorth = compute_hjorth_parameters(data)
        feature_parts.append(hjorth)
    
    # Entropy features
    if include_entropy:
        entropy_feats = compute_entropy_features(data, sfreq)
        feature_parts.append(entropy_feats)
    
    # Concatenate all features
    features = np.concatenate(feature_parts)
    
    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def get_feature_names(
    n_channels: int,
    channel_names: Optional[List[str]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    statistical_features: Optional[List[str]] = None,
    include_hjorth: bool = True,
    include_relative_power: bool = True,
    include_entropy: bool = False
) -> List[str]:
    """
    Get feature names for interpretation.
    
    Args:
        n_channels: Number of channels
        channel_names: Optional list of channel names
        bands: Frequency bands
        statistical_features: Statistical features
        include_hjorth: Whether Hjorth parameters are included
        include_relative_power: Whether relative power is included
        include_entropy: Whether entropy features are included
        
    Returns:
        List of feature names
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    if statistical_features is None:
        statistical_features = ['mean', 'std', 'skewness', 'kurtosis', 'line_length', 'energy']
    
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(n_channels)]
    
    names = []
    
    # PSD features
    for ch in channel_names:
        for band in bands.keys():
            names.append(f'{ch}_psd_{band}')
    
    # Relative power
    if include_relative_power:
        for ch in channel_names:
            for band in bands.keys():
                names.append(f'{ch}_rel_{band}')
    
    # Statistical features
    for ch in channel_names:
        for feat in statistical_features:
            names.append(f'{ch}_{feat}')
    
    # Hjorth parameters
    if include_hjorth:
        for ch in channel_names:
            names.append(f'{ch}_hjorth_activity')
            names.append(f'{ch}_hjorth_mobility')
            names.append(f'{ch}_hjorth_complexity')
    
    # Entropy features
    if include_entropy:
        for ch in channel_names:
            names.append(f'{ch}_sample_entropy')
            names.append(f'{ch}_permutation_entropy')
    
    return names


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Simulate 4 seconds of 18-channel EEG at 256 Hz
    sfreq = 256
    duration = 4
    n_channels = 18
    n_samples = duration * sfreq
    
    data = np.random.randn(n_channels, n_samples)
    
    # Extract features
    features = extract_features(data, sfreq)
    
    print(f"Input shape: {data.shape}")
    print(f"Feature vector length: {len(features)}")
    
    # Get feature names
    names = get_feature_names(n_channels)
    print(f"Number of feature names: {len(names)}")
    print(f"First 10 features: {names[:10]}")
