"""
Transfer Learning Pipeline for Seizure Prediction.

This pipeline:
1. Loads and preprocesses EEG data from all patients
2. Extracts raw windows (for CNN) and features (for hybrid approach)
3. Pretrains a global CNN-LSTM model on all patients
4. Fine-tunes on each patient individually
5. Evaluates and compares with baseline
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import gc

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from data.index_builder import build_seizure_index, build_file_index
from data.preprocessing import preprocess_edf
from data.segmentation import segment_signal
from data.labeling import label_windows
from models.deep_model import (
    SeizurePredictorDL,
    pretrain_global_model,
    finetune_patient_model,
    find_optimal_threshold,
    ensemble_predict,
    train_ensemble_models,
    reject_artifacts,
    EEGAugmentation,
    FocalLoss,
    MAMLTrainer
)
from evaluation.alarm_logic import generate_alarms
from evaluation.metrics import compute_event_metrics
from features.extractor import compute_entropy_features


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_raw_windows(
    windows_df: pd.DataFrame,
    window_data: Dict[int, np.ndarray],
    n_timepoints: int = 1024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare raw EEG windows for CNN input.
    
    Args:
        windows_df: DataFrame with window metadata
        window_data: Dict of window_idx -> raw data array
        n_timepoints: Target number of timepoints (will pad/truncate)
        
    Returns:
        X: Raw data (n_samples, n_channels, n_timepoints)
        y: Labels (n_samples,)
        indices: Original window indices
    """
    # Filter to preictal and interictal only (1 = preictal, 0 = interictal, -1 = excluded)
    mask = windows_df['label'].isin([0, 1])
    filtered_df = windows_df[mask].copy()
    
    X_list = []
    y_list = []
    indices = []
    
    # Determine expected number of channels from first valid window
    expected_channels = None
    for win_idx in window_data:
        expected_channels = window_data[win_idx].shape[0]
        break
    
    if expected_channels is None:
        return np.array([]), np.array([]), np.array([])
    
    skipped_channels = 0
    for idx, row in filtered_df.iterrows():
        win_idx = row['window_idx']
        if win_idx not in window_data:
            continue
        
        data = window_data[win_idx]
        n_channels, n_samples = data.shape
        
        # Skip windows with different channel count
        if n_channels != expected_channels:
            skipped_channels += 1
            continue
        
        # Pad or truncate to target length
        if n_samples < n_timepoints:
            # Pad with zeros
            padded = np.zeros((n_channels, n_timepoints))
            padded[:, :n_samples] = data
            data = padded
        elif n_samples > n_timepoints:
            # Truncate
            data = data[:, :n_timepoints]
        
        X_list.append(data)
        y_list.append(1 if row['label'] == 1 else 0)
        indices.append(idx)
    
    if skipped_channels > 0:
        logger.warning(f"Skipped {skipped_channels} windows with inconsistent channel count")
    
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    indices = np.array(indices)
    
    return X, y, indices


def compute_entropy_for_windows(X: np.ndarray, sfreq: float = 256.0) -> np.ndarray:
    """
    Compute entropy features for all windows.
    
    Args:
        X: Raw data (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency
        
    Returns:
        Entropy features (n_samples, n_channels * 2) - SampEn and PermEn per channel
    """
    n_samples = X.shape[0]
    n_channels = X.shape[1]
    entropy_features = np.zeros((n_samples, n_channels * 2), dtype=np.float32)
    
    for i in range(n_samples):
        entropy_features[i] = compute_entropy_features(X[i], sfreq)
    
    return entropy_features


def sample_patient_for_pretraining(
    X: np.ndarray,
    y: np.ndarray,
    target_samples: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample patient windows while preserving class balance as much as possible.
    """
    n_samples = len(y)
    if target_samples >= n_samples:
        return X, y
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0")

    pos_idx = np.where(y >= 0.5)[0]
    neg_idx = np.where(y < 0.5)[0]

    # Fallback: single-class patient data
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        selected = rng.choice(n_samples, size=target_samples, replace=False)
        return X[selected], y[selected]

    pos_ratio = len(pos_idx) / n_samples
    n_pos = int(round(target_samples * pos_ratio))
    n_pos = min(max(n_pos, 0), len(pos_idx))
    n_neg = target_samples - n_pos
    n_neg = min(max(n_neg, 0), len(neg_idx))

    # Fill missing quota if one class hit availability limit
    used = n_pos + n_neg
    if used < target_samples:
        remaining = target_samples - used
        pos_room = len(pos_idx) - n_pos
        add_pos = min(remaining, pos_room)
        n_pos += add_pos
        remaining -= add_pos

        neg_room = len(neg_idx) - n_neg
        add_neg = min(remaining, neg_room)
        n_neg += add_neg

    # Guarantee at least one sample from each class when possible
    if target_samples > 1:
        if n_pos == 0 and len(pos_idx) > 0 and n_neg > 1:
            n_pos, n_neg = 1, target_samples - 1
        elif n_neg == 0 and len(neg_idx) > 0 and n_pos > 1:
            n_neg, n_pos = 1, target_samples - 1

    selected_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=np.int64)
    selected_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=np.int64)
    selected = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(selected)

    return X[selected], y[selected]


def process_patient_for_dl(
    patient_id: str,
    config: dict,
    seizure_index: pd.DataFrame,
    file_index: pd.DataFrame
) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Process a single patient and return raw windows for DL.
    
    Returns:
        X: Raw data (n_samples, n_channels, n_timepoints)
        y: Labels
        windows_df: Window metadata
    """
    data_root = Path(config['paths']['data_root'])
    patient_dir = data_root / patient_id
    
    # Get patient files and seizures
    patient_files = file_index[file_index['patient'] == patient_id]
    patient_seizures = seizure_index[seizure_index['patient'] == patient_id]
    
    if len(patient_seizures) == 0:
        logger.warning(f"No seizures for {patient_id}, skipping")
        return None
    
    # Process each file
    all_windows = []
    all_window_data = {}
    window_counter = 0
    
    for _, file_row in tqdm(patient_files.iterrows(), total=len(patient_files), desc=f"  {patient_id}"):
        edf_path = Path(file_row['edf_path'])
        
        if not edf_path.exists():
            continue
        
        # Preprocessну с
        try:
            result = preprocess_edf(
                edf_path,
                channels=config['channels'].get('selected'),
                target_sfreq=config['preprocessing']['target_sfreq'],
                bandpass_low=config['preprocessing']['bandpass_low'],
                bandpass_high=config['preprocessing']['bandpass_high'],
                notch_freq=config['preprocessing']['notch_freq']
            )
            raw_data, sfreq, channels_out, qc = result[0], result[1], result[2], result[3]
            if qc['flat_ratio'] > config['qc']['max_flat_ratio']:
                logger.warning(f"  Skip {edf_path.name}: flat_ratio={qc['flat_ratio']} > {config['qc']['max_flat_ratio']}")
                continue
        except Exception as e:
            logger.warning(f"Failed to process {edf_path}: {e}")
            continue
        
        # Segment
        windows = list(segment_signal(
            raw_data,
            sfreq=sfreq,
            window_length=config['windowing']['window_length'],
            window_step=config['windowing']['window_step']
        ))
        
        # Store windows with file info - use file index as offset (each file ~1 hour)
        file_idx = patient_files.index.get_loc(file_row.name) if hasattr(file_row, 'name') else 0
        file_start = float(file_idx * 3600)  # Approximate 1 hour per file
        
        for window_data, t_start, t_end in windows:
            win = {
                'start_sec': t_start,
                'end_sec': t_end,
                'file_path': str(file_row['edf_path']),
                'file_start': file_start,
                'global_start': file_start + t_start,
                'global_end': file_start + t_end,
                'window_idx': window_counter
            }
            
            all_window_data[window_counter] = window_data
            all_windows.append(win)
            window_counter += 1
    
    if not all_windows:
        return None
    
    windows_df = pd.DataFrame(all_windows)
    
    # Label windows - add required columns
    windows_df['patient'] = patient_id
    windows_df['edf_file'] = windows_df['file_path'].apply(lambda x: Path(x).name)
    windows_df['t_start'] = windows_df['start_sec']
    windows_df['t_end'] = windows_df['end_sec']
    
    windows_df = label_windows(
        windows_df,
        patient_seizures,
        sph=config['timing']['sph'],
        preictal_duration=config['timing']['preictal_duration'],
        postictal_excluded=config['timing']['postictal_excluded'],
        interictal_gap=config['timing']['interictal_gap']
    )
    
    # Prepare for DL
    n_timepoints = int(config['windowing']['window_length'] * config['preprocessing']['target_sfreq'])
    X, y, indices = prepare_raw_windows(windows_df, all_window_data, n_timepoints)
    
    # Check if we have any valid windows
    if len(X) == 0:
        logger.warning(f"  {patient_id}: No valid windows after filtering")
        return None
    
    # Update windows_df to only include processed windows
    windows_df = windows_df.loc[indices].reset_index(drop=True)
    
    logger.info(f"  {patient_id}: {len(y)} windows, {y.sum():.0f} preictal, {len(y) - y.sum():.0f} interictal")
    
    return X, y, windows_df


def run_transfer_learning_pipeline(
    config_path: Path,
    patients: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    train_patients: Optional[List[str]] = None,
    test_patients: Optional[List[str]] = None,
    resume: bool = False
) -> Dict:
    """
    Run the full Transfer Learning pipeline.
    
    Args:
        config_path: Path to config file
        patients: List of patient IDs (None = all) - used if train/test not specified
        output_dir: Output directory
        train_patients: Patients for pretraining (if None, uses all patients)
        test_patients: Patients for testing only (not seen during training)
        resume: If True, skip pretraining and use existing pretrained model
        
    Returns:
        Results dictionary
    """
    start_time = datetime.now()
    
    # Load config
    config = load_config(config_path)
    data_root = Path(config['paths']['data_root'])
    
    if output_dir is None:
        output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING PIPELINE FOR SEIZURE PREDICTION")
    logger.info("=" * 60)
    
    # Step 1: Build indices
    logger.info("Step 1: Building seizure and file indices")
    seizure_index = build_seizure_index(data_root)
    
    # Determine train/test split
    all_patients = seizure_index['patient'].unique().tolist()
    
    if train_patients is not None and test_patients is not None:
        # Use explicit train/test split
        patients = train_patients + test_patients
        logger.info(f"Train patients ({len(train_patients)}): {train_patients}")
        logger.info(f"Test patients ({len(test_patients)}): {test_patients}")
    elif patients is not None:
        # Use provided patients for both train and test (old behavior)
        train_patients = patients
        test_patients = patients
    else:
        # Use all patients
        patients = all_patients
        train_patients = patients
        test_patients = patients
    
    file_index = build_file_index(data_root, patients)
    
    logger.info(f"Processing {len(patients)} patients total")
    
    # Step 2: Build/validate disk cache (do not keep all patient arrays in RAM)
    logger.info("Step 2: Preparing disk cache for all patients")
    
    cache_dir = output_dir / "preprocessed_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    patient_cache = {}
    
    for patient_id in patients:
        cache_file = cache_dir / f"{patient_id}_data.npz"
        windows_cache = cache_dir / f"{patient_id}_windows.csv"
        
        if cache_file.exists() and windows_cache.exists():
            logger.info(f"  {patient_id}: Cache ready")
        else:
            # Process and persist to cache
            result = process_patient_for_dl(patient_id, config, seizure_index, file_index)
            if result is None:
                continue
            X, y, windows_df = result
            
            np.savez_compressed(cache_file, X=X, y=y)
            windows_df.to_csv(windows_cache, index=False)
            logger.info(f"  {patient_id}: Saved to cache")

            # Explicit cleanup after cache write
            del X, y, windows_df
            gc.collect()

        if cache_file.exists() and windows_cache.exists():
            patient_cache[patient_id] = (cache_file, windows_cache)

    if not patient_cache:
        logger.error("No patient data available in cache!")
        return {}

    unavailable = [p for p in patients if p not in patient_cache]
    if unavailable:
        logger.warning(f"Skipping patients without cache/data: {unavailable}")

    # Keep deterministic order while filtering to available patients
    patients = [p for p in patients if p in patient_cache]
    train_patients = [p for p in train_patients if p in patient_cache]
    test_patients = [p for p in test_patients if p in patient_cache]
    
    # Step 3: Pretrain global model on TRAIN patients only
    selected_channels = config.get('channels', {}).get('selected', [])
    if selected_channels:
        n_channels = len(selected_channels)
    else:
        # Fallback: infer channel count from one cached patient
        ref_patient = patients[0]
        ref_cache_file, _ = patient_cache[ref_patient]
        with np.load(ref_cache_file) as cached_ref:
            ref_X = cached_ref['X']
            n_channels = ref_X.shape[1]

    n_timepoints = int(config['windowing']['window_length'] * config['preprocessing']['target_sfreq'])

    pretrained_model_path = output_dir / "pretrained_model.pt"
    
    if resume and pretrained_model_path.exists():
        logger.info("Step 3: SKIPPED - Loading existing pretrained model (--resume)")
        logger.info(f"  Loading from: {pretrained_model_path}")
        global_model = SeizurePredictorDL(n_channels=n_channels, n_timepoints=n_timepoints)
        global_model.load(pretrained_model_path)
    else:
        logger.info("Step 3: Pretraining global CNN-LSTM model on TRAIN patients")

        if not train_patients:
            logger.error("No train patients available for pretraining")
            return {}

        pretrain_max_samples = int(config.get('deep_learning', {}).get('pretrain_max_samples', 50000))
        seed = int(config.get('training', {}).get('random_seed', 42))
        rng = np.random.default_rng(seed)

        # Pass 1: count available samples without loading all arrays simultaneously
        train_counts = {}
        total_train_samples = 0
        total_train_preictal = 0

        for patient_id in train_patients:
            cache_file, _ = patient_cache[patient_id]
            with np.load(cache_file) as cached:
                y = cached['y']
            n_samples = int(len(y))
            if n_samples == 0:
                logger.warning(f"  {patient_id}: Empty cache, skipping")
                continue
            train_counts[patient_id] = n_samples
            total_train_samples += n_samples
            total_train_preictal += int(y.sum())

        if not train_counts:
            logger.error("No valid train data found in cache")
            return {}

        sample_ratio = 1.0
        if pretrain_max_samples > 0 and total_train_samples > pretrain_max_samples:
            sample_ratio = pretrain_max_samples / total_train_samples

        logger.info(
            f"  Train pool before subsampling: {total_train_samples} samples, "
            f"{total_train_preictal} preictal"
        )
        if sample_ratio < 1.0:
            logger.info(
                f"  Subsampling train pool to ~{pretrain_max_samples} samples "
                f"(ratio={sample_ratio:.3f})"
            )

        # Pass 2: load one patient at a time, subsample, and keep only sampled tensors
        train_data = {}
        for patient_id in train_patients:
            if patient_id not in train_counts:
                continue
            cache_file, _ = patient_cache[patient_id]
            with np.load(cache_file) as cached:
                X = cached['X']
                y = cached['y']

            target_samples = max(1, int(round(len(y) * sample_ratio))) if sample_ratio < 1.0 else len(y)
            if target_samples < len(y):
                X, y = sample_patient_for_pretraining(X, y, target_samples, rng)

            train_data[patient_id] = (X, y)
            logger.info(f"  {patient_id}: pretrain samples={len(y)}, preictal={int(y.sum())}")

        logger.info(f"  Using {len(train_data)} train patients for pretraining")
        
        global_model = pretrain_global_model(
            train_data,  # Only train patients!
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            epochs=config.get('deep_learning', {}).get('pretrain_epochs', 30),
            batch_size=config.get('deep_learning', {}).get('batch_size', 64),
            save_path=pretrained_model_path,
            max_samples=pretrain_max_samples
        )

        del train_data
        gc.collect()

    # We only need the checkpoint on disk for per-patient fine-tuning.
    del global_model
    gc.collect()
    
    # Step 4: Fine-tune and evaluate for each patient
    logger.info("Step 4: Fine-tuning and evaluating for each patient")
    
    results = {}
    train_results = {}
    test_results = {}
    
    for patient_id in patients:
        is_test_patient = patient_id in test_patients and patient_id not in train_patients
        patient_type = "TEST" if is_test_patient else "TRAIN"
        logger.info(f"  Processing {patient_id} [{patient_type}]")

        cache_file, windows_cache = patient_cache[patient_id]
        with np.load(cache_file) as cached:
            X = cached['X']
            y = cached['y']
        windows_df = pd.read_csv(windows_cache)
        
        # Normalize channel count to match model
        patient_channels = X.shape[1]
        if patient_channels < n_channels:
            padding = np.zeros((X.shape[0], n_channels - patient_channels, X.shape[2]), dtype=X.dtype)
            X = np.concatenate([X, padding], axis=1)
            logger.info(f"    Padded channels: {patient_channels}→{n_channels}")
        elif patient_channels > n_channels:
            X = X[:, :n_channels, :]
            logger.info(f"    Truncated channels: {patient_channels}→{n_channels}")
        
        # Artifact rejection (clean data before training)
        use_artifact_rejection = config.get('deep_learning', {}).get('artifact_rejection', True)
        valid_indices = None
        if use_artifact_rejection:
            X_clean, y_clean, valid_indices = reject_artifacts(
                X, y,
                max_amplitude=config.get('deep_learning', {}).get('max_amplitude', 200.0),
                flat_threshold=config.get('deep_learning', {}).get('flat_threshold', 1e-6),
                max_reject_ratio=0.3
            )
            # Update windows_df to match cleaned data
            windows_df_clean = windows_df.iloc[valid_indices].reset_index(drop=True)
        else:
            X_clean, y_clean = X, y
            windows_df_clean = windows_df
        
        # Compute entropy features if enabled
        use_entropy = config.get('deep_learning', {}).get('include_entropy', False)
        entropy_features = None
        n_entropy_features = 0
        
        if use_entropy:
            entropy_cache = cache_dir / f"{patient_id}_entropy.npy"
            if entropy_cache.exists():
                entropy_features = np.load(entropy_cache)
                logger.info(f"    Loaded entropy from cache ({entropy_features.shape[1]} features)")
            else:
                logger.info(f"    Computing entropy features...")
                entropy_features = compute_entropy_for_windows(X, config['preprocessing']['target_sfreq'])
                np.save(entropy_cache, entropy_features)
                logger.info(f"    Saved entropy to cache ({entropy_features.shape[1]} features)")
            n_entropy_features = entropy_features.shape[1]
        
        # Create patient output directory
        patient_output = output_dir / patient_id
        patient_output.mkdir(parents=True, exist_ok=True)
        
        # Load fresh copy of pretrained model for fine-tuning
        patient_model = SeizurePredictorDL(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_entropy_features=n_entropy_features
        )
        patient_model.load(pretrained_model_path)
        
        # Fine-tune with file-based split and adaptive parameters (use cleaned data)
        finetune_history = finetune_patient_model(
            patient_model,
            X_clean, y_clean,
            epochs=config.get('deep_learning', {}).get('finetune_epochs', 20),
            batch_size=config.get('deep_learning', {}).get('batch_size', 32),
            lr=config.get('deep_learning', {}).get('finetune_lr', 1e-4),
            freeze_cnn=True,
            windows_df=windows_df_clean,
            entropy_features=entropy_features[valid_indices] if entropy_features is not None and use_artifact_rejection else entropy_features,
            adaptive_lr=True,
            adaptive_epochs=True,
            use_focal_loss=config.get('deep_learning', {}).get('use_focal_loss', True),
            focal_alpha=config.get('deep_learning', {}).get('focal_alpha', 0.75),
            focal_gamma=config.get('deep_learning', {}).get('focal_gamma', 2.0)
        )
        
        # Save fine-tuned model
        patient_model.save(patient_output / "model_finetuned.pt")
        
        # Predict on ALL data (including artifacts) for evaluation
        probs = patient_model.predict_proba(X, entropy_features=entropy_features)
        
        # Find optimal personalized threshold for this patient
        optimal_threshold, threshold_metrics = find_optimal_threshold(y, probs, metric='f1')
        logger.info(f"    Optimal threshold: {optimal_threshold:.2f} (F1={threshold_metrics['f1']:.3f})")
        
        windows_df['prob_preictal'] = probs
        windows_df['pred_label'] = (probs >= optimal_threshold).astype(int)
        
        # Generate alarms with personalized threshold
        alarms_df = generate_alarms(
            windows_df,
            threshold=optimal_threshold,  # Use personalized threshold
            smoothing_window=int(config['alarm']['smoothing_window'] / config['windowing']['window_step']),
            refractory_sec=config['alarm']['refractory_period']
        )
        
        # Compute simple metrics
        # Get list of files that were actually processed (passed QC)
        processed_files = windows_df['edf_file'].unique()
        
        # Filter seizures: only count seizures from files that were successfully processed
        all_patient_seizures = seizure_index[seizure_index['patient'] == patient_id]
        patient_seizures = all_patient_seizures[all_patient_seizures['edf_file'].isin(processed_files)]
        
        # Log if any seizures were excluded due to QC failures
        n_excluded = len(all_patient_seizures) - len(patient_seizures)
        if n_excluded > 0:
            excluded_files = set(all_patient_seizures['edf_file']) - set(processed_files)
            logger.warning(
                f"    {patient_id}: {n_excluded} seizures excluded (files failed QC or missing channels): "
                f"{', '.join(sorted(excluded_files))}"
            )
        
        # Simple event-based metrics
        n_seizures = len(patient_seizures)
        n_alarms = len(alarms_df)
        total_hours = len(windows_df) * config['windowing']['window_step'] / 3600
        
        # Count detected seizures (alarm within SOP before seizure onset)
        # Compare times WITHIN THE SAME FILE (both in file-relative coordinates)
        sph = config['timing']['sph']
        sop = config['timing']['sop']
        # Use personalized threshold instead of config threshold
        detected = 0
        
        for _, seizure in patient_seizures.iterrows():
            onset = seizure['onset_sec']  # time within file
            edf_file = seizure['edf_file']
            
            # Get windows for THIS FILE only
            file_windows = windows_df[windows_df['edf_file'] == edf_file]
            if file_windows.empty:
                continue
            
            # Check if any high-probability window in valid prediction window
            # Valid window: [onset - sop, onset - sph] (e.g., 30 min to 1 min before seizure)
            for _, win in file_windows.iterrows():
                if win['prob_preictal'] >= optimal_threshold:
                    win_time = win['t_start']  # time within file (same coordinates as onset)
                    if (onset - sop) <= win_time <= (onset - sph):
                        detected += 1
                        break
        
        sensitivity = detected / n_seizures if n_seizures > 0 else 0
        # FA = alarms that are NOT true detections (must be >= 0)
        false_alarms = max(0, n_alarms - detected)
        fa_per_24h = false_alarms / total_hours * 24 if total_hours > 0 else 0
        
        event_metrics = {
            'sensitivity': sensitivity,
            'n_detected': detected,
            'n_seizures': n_seizures,
            'n_alarms': n_alarms,
            'fa_per_24h': fa_per_24h,
            'total_hours': total_hours,
            'optimal_threshold': optimal_threshold,
            'threshold_f1': threshold_metrics['f1']
        }
        
        # Save results
        windows_df.to_csv(patient_output / "windows_dl.csv", index=False)
        alarms_df.to_csv(patient_output / "alarms_dl.csv", index=False)
        
        with open(patient_output / "metrics_dl.json", 'w') as f:
            json.dump(event_metrics, f, indent=2)
        
        patient_result = {
            'n_seizures': len(patient_seizures),
            'n_windows': len(windows_df),
            'n_preictal': int(y.sum()),
            'sensitivity': event_metrics.get('sensitivity', 0),
            'fa_per_24h': event_metrics.get('fa_per_24h', 0),
            'finetune_val_auc': finetune_history['val_auc'][-1] if finetune_history['val_auc'] else 0,
            'is_test': is_test_patient
        }
        
        results[patient_id] = patient_result
        if is_test_patient:
            test_results[patient_id] = patient_result
        else:
            train_results[patient_id] = patient_result
        
        logger.info(
            f"    {patient_id} [{patient_type}]: Sensitivity={patient_result['sensitivity']:.1%}, "
            f"FA/24h={patient_result['fa_per_24h']:.2f}, "
            f"AUC={patient_result['finetune_val_auc']:.3f}"
        )

        # Explicit cleanup keeps resident memory stable across long runs.
        del X, y, windows_df, X_clean, y_clean, windows_df_clean
        del entropy_features, patient_model, finetune_history, probs, alarms_df
        gc.collect()
    
    # Step 5: Generate summary report
    logger.info("Step 5: Generating summary report")
    
    summary_lines = [
        "=" * 60,
        "TRANSFER LEARNING SEIZURE PREDICTION - SUMMARY",
        f"Generated: {datetime.now()}",
        "=" * 60,
        "",
    ]
    
    # Helper function to compute metrics for a subset
    def compute_summary_metrics(result_dict, label):
        lines = []
        if not result_dict:
            return lines
        
        lines.append(f"{label} RESULTS")
        lines.append("-" * 50)
        lines.append(f"{'Patient':<10} {'Seizures':<10} {'Sensitivity':<12} {'FA/24h':<10} {'AUC':<10}")
        lines.append("-" * 50)
        
        total_sz = 0
        total_det = 0
        sens_list = []
        fa_list = []
        auc_list = []
        
        for pid, res in result_dict.items():
            lines.append(
                f"{pid:<10} {res['n_seizures']:<10} "
                f"{res['sensitivity']*100:>6.1f}%     "
                f"{res['fa_per_24h']:<10.2f} "
                f"{res['finetune_val_auc']:<10.3f}"
            )
            total_sz += res['n_seizures']
            total_det += int(res['sensitivity'] * res['n_seizures'])
            sens_list.append(res['sensitivity'])
            fa_list.append(res['fa_per_24h'])
            auc_list.append(res['finetune_val_auc'])
        
        lines.append("-" * 50)
        lines.append(f"  {label} Seizures: {total_sz}, Detected: {total_det}")
        if total_sz > 0:
            lines.append(f"  {label} Sensitivity: {total_det/total_sz*100:.1f}%")
        lines.append(f"  {label} Mean Sensitivity: {np.mean(sens_list)*100:.1f}% ± {np.std(sens_list)*100:.1f}%")
        lines.append(f"  {label} Mean FA/24h: {np.mean(fa_list):.2f} ± {np.std(fa_list):.2f}")
        lines.append(f"  {label} Mean AUC: {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
        lines.append("")
        
        return lines
    
    # Add TRAIN results
    summary_lines.extend(compute_summary_metrics(train_results, "TRAIN"))
    
    # Add TEST results (most important!)
    summary_lines.extend(compute_summary_metrics(test_results, "TEST (UNSEEN PATIENTS)"))
    
    summary_lines.extend([
        f"Pipeline completed in {datetime.now() - start_time}",
        "=" * 60
    ])
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_dir / "summary_transfer_learning.txt", 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    # Save full results
    with open(output_dir / "results_transfer_learning.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Pipeline completed in {datetime.now() - start_time}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfer Learning Pipeline for Seizure Prediction")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file path")
    parser.add_argument("--patients", type=str, nargs="+", help="Patient IDs to process")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    output_dir = Path(args.output) if args.output else None
    
    results = run_transfer_learning_pipeline(
        config_path=config_path,
        patients=args.patients,
        output_dir=output_dir
    )
