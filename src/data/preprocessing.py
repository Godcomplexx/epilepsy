"""
Module for EEG signal preprocessing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy import signal
from loguru import logger

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logger.warning("MNE not available, using pyedflib for EDF loading")

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False


def load_edf_channels(
    edf_path: Path,
    channels: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load EDF file and return selected channels.
    
    Args:
        edf_path: Path to EDF file
        channels: List of channel names to load. If None, loads all.
        verbose: Whether to print loading info
        
    Returns:
        Tuple of (data, sampling_rate, channel_names)
        - data: np.ndarray of shape (n_channels, n_samples)
        - sampling_rate: float
        - channel_names: List[str]
    """
    edf_path = Path(edf_path)
    
    if MNE_AVAILABLE:
        return _load_with_mne(edf_path, channels, verbose)
    elif PYEDFLIB_AVAILABLE:
        return _load_with_pyedflib(edf_path, channels)
    else:
        raise ImportError("Neither MNE nor pyedflib is available")


def _load_with_mne(
    edf_path: Path,
    channels: Optional[List[str]],
    verbose: bool
) -> Tuple[np.ndarray, float, List[str]]:
    """Load EDF using MNE."""
    # CHB-MIT EDF files may contain duplicate labels; suppress this known warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="Channel names are not unique.*"
        )
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=verbose)
    
    # Get available channels
    available_channels = raw.ch_names
    
    if channels is not None:
        # Filter to requested channels that exist
        channels_to_pick = [ch for ch in channels if ch in available_channels]
        if len(channels_to_pick) < len(channels):
            missing = set(channels) - set(channels_to_pick)
            logger.warning(f"Missing channels: {missing}")
        raw.pick(channels_to_pick)
    
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    
    return data, sfreq, list(ch_names)


def _load_with_pyedflib(
    edf_path: Path,
    channels: Optional[List[str]]
) -> Tuple[np.ndarray, float, List[str]]:
    """Load EDF using pyedflib."""
    f = pyedflib.EdfReader(str(edf_path))
    
    try:
        n_channels = f.signals_in_file
        ch_names = f.getSignalLabels()
        sfreq = f.getSampleFrequency(0)
        
        if channels is not None:
            indices = [i for i, ch in enumerate(ch_names) if ch in channels]
            ch_names = [ch_names[i] for i in indices]
        else:
            indices = list(range(n_channels))
        
        n_samples = f.getNSamples()[indices[0]]
        data = np.zeros((len(indices), n_samples))
        
        for i, idx in enumerate(indices):
            data[i, :] = f.readSignal(idx)
        
    finally:
        f.close()
    
    return data, sfreq, ch_names


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Args:
        data: Signal data (n_channels, n_samples)
        sfreq: Sampling frequency
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        order: Filter order
        
    Returns:
        Filtered data
    """
    nyq = sfreq / 2
    low = low_freq / nyq
    high = high_freq / nyq
    
    # Ensure valid frequency range
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter along time axis
    filtered = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    width: float = 2.0
) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.
    
    Args:
        data: Signal data (n_channels, n_samples)
        sfreq: Sampling frequency
        freq: Notch frequency (e.g., 50 or 60 Hz)
        width: Notch width in Hz
        
    Returns:
        Filtered data
    """
    nyq = sfreq / 2
    
    if freq >= nyq:
        logger.warning(f"Notch frequency {freq} Hz >= Nyquist {nyq} Hz, skipping")
        return data
    
    # Design notch filter
    Q = freq / width
    b, a = signal.iirnotch(freq, Q, sfreq)
    
    filtered = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered


def resample_signal(
    data: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float
) -> np.ndarray:
    """
    Resample signal to target frequency.
    
    Args:
        data: Signal data (n_channels, n_samples)
        orig_sfreq: Original sampling frequency
        target_sfreq: Target sampling frequency
        
    Returns:
        Resampled data
    """
    if orig_sfreq == target_sfreq:
        return data
    
    n_samples_new = int(data.shape[-1] * target_sfreq / orig_sfreq)
    resampled = signal.resample(data, n_samples_new, axis=-1)
    
    return resampled


def check_signal_quality(
    data: np.ndarray,
    max_nan_ratio: float = 0.1,
    max_flat_ratio: float = 0.1,
    amplitude_threshold: float = 500.0
) -> Dict:
    """
    Check signal quality and return QC metrics.
    
    Args:
        data: Signal data (n_channels, n_samples)
        max_nan_ratio: Maximum allowed ratio of NaN values
        max_flat_ratio: Maximum allowed ratio of flat signal
        amplitude_threshold: Maximum amplitude for clipping detection
        
    Returns:
        Dictionary with QC results
    """
    n_channels, n_samples = data.shape
    
    # Check for NaN
    nan_ratio = np.isnan(data).sum() / data.size
    
    # Check for flat signal (std < threshold)
    flat_channels = []
    for i in range(n_channels):
        if np.std(data[i]) < 1e-6:
            flat_channels.append(i)
    flat_ratio = len(flat_channels) / n_channels
    
    # Check for clipping
    clipped_ratio = (np.abs(data) > amplitude_threshold).sum() / data.size
    
    # Overall quality
    is_good = (
        nan_ratio <= max_nan_ratio and
        flat_ratio <= max_flat_ratio and
        clipped_ratio < 0.01
    )
    
    return {
        'is_good': is_good,
        'nan_ratio': nan_ratio,
        'flat_ratio': flat_ratio,
        'flat_channels': flat_channels,
        'clipped_ratio': clipped_ratio,
        'mean_amplitude': np.nanmean(np.abs(data)),
        'max_amplitude': np.nanmax(np.abs(data))
    }


def preprocess_edf(
    edf_path: Path,
    channels: Optional[List[str]] = None,
    target_sfreq: float = 256.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 50.0,
    notch_freq: Optional[float] = 60.0,
    notch_width: float = 2.0,
    run_qc: bool = True
) -> Tuple[np.ndarray, float, List[str], Optional[Dict]]:
    """
    Full preprocessing pipeline for EDF file.
    
    Args:
        edf_path: Path to EDF file
        channels: Channels to load
        target_sfreq: Target sampling frequency
        bandpass_low: Bandpass low cutoff
        bandpass_high: Bandpass high cutoff
        notch_freq: Notch filter frequency (None to skip)
        notch_width: Notch filter width
        run_qc: Whether to run quality check
        
    Returns:
        Tuple of (data, sfreq, channel_names, qc_results)
    """
    # Load data
    data, orig_sfreq, ch_names = load_edf_channels(edf_path, channels)
    
    # Handle NaN values
    if np.isnan(data).any():
        logger.warning(f"NaN values found in {edf_path.name}, interpolating")
        data = np.nan_to_num(data, nan=0.0)
    
    # Bandpass filter
    data = bandpass_filter(data, orig_sfreq, bandpass_low, bandpass_high)
    
    # Notch filter
    if notch_freq is not None:
        data = notch_filter(data, orig_sfreq, notch_freq, notch_width)
    
    # Resample
    data = resample_signal(data, orig_sfreq, target_sfreq)
    
    # Quality check
    qc_results = None
    if run_qc:
        qc_results = check_signal_quality(data)
        if not qc_results['is_good']:
            logger.warning(f"QC failed for {edf_path.name}: {qc_results}")
    
    return data, target_sfreq, ch_names, qc_results


if __name__ == "__main__":
    # Quick test
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config['paths']['data_root'])
    test_edf = data_root / "chb01" / "chb01_03.edf"
    
    if test_edf.exists():
        data, sfreq, ch_names, qc = preprocess_edf(
            test_edf,
            channels=config['channels']['selected'],
            target_sfreq=config['preprocessing']['target_sfreq'],
            bandpass_low=config['preprocessing']['bandpass_low'],
            bandpass_high=config['preprocessing']['bandpass_high'],
            notch_freq=config['preprocessing']['notch_freq']
        )
        
        print(f"Data shape: {data.shape}")
        print(f"Sampling rate: {sfreq}")
        print(f"Channels: {ch_names}")
        print(f"QC: {qc}")
