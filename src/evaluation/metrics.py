"""
Module for event-based evaluation metrics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def compute_event_metrics(
    alarms_df: pd.DataFrame,
    seizure_index: pd.DataFrame,
    sph: float = 60.0,
    sop: float = 600.0,
    total_recording_hours: Optional[float] = None
) -> Dict:
    """
    Compute event-based metrics for seizure prediction.
    
    Args:
        alarms_df: DataFrame with alarms (patient, edf_file, alarm_time)
        seizure_index: DataFrame with seizures (patient, edf_file, onset_sec, offset_sec)
        sph: Seizure Prediction Horizon in seconds
        sop: Seizure Occurrence Period in seconds
        total_recording_hours: Total recording time in hours (for FA rate)
        
    Returns:
        Dictionary with metrics:
            - sensitivity: Seizure-wise sensitivity
            - n_seizures_detected: Number of seizures with successful alarm
            - n_seizures_total: Total number of seizures
            - n_alarms: Total number of alarms
            - n_false_alarms: Number of false alarms
            - false_alarm_rate: False alarms per 24 hours
            - time_in_warning: Fraction of time in warning state
    """
    results = {
        'n_seizures_total': 0,
        'n_seizures_detected': 0,
        'n_alarms': 0,
        'n_false_alarms': 0,
        'n_true_alarms': 0,
        'sensitivity': 0.0,
        'false_alarm_rate_24h': 0.0,
        'time_in_warning': 0.0,
        'detected_seizures': [],
        'missed_seizures': [],
        'alarm_details': []
    }
    
    if seizure_index.empty:
        logger.warning("No seizures in index")
        return results
    
    # Process each patient separately
    patients = seizure_index['patient'].unique()
    
    all_seizure_detected = []
    all_alarm_success = []
    total_warning_time = 0.0
    total_recording_time = 0.0
    
    for patient in patients:
        patient_seizures = seizure_index[seizure_index['patient'] == patient]
        patient_alarms = alarms_df[alarms_df['patient'] == patient] if not alarms_df.empty else pd.DataFrame()
        
        # Track which seizures are detected
        for _, seizure in patient_seizures.iterrows():
            seizure_detected = False
            edf_file = seizure['edf_file']
            onset = seizure['onset_sec']
            seizure_id = seizure.get('seizure_id', 0)
            
            # Check alarms for this file
            file_alarms = patient_alarms[patient_alarms['edf_file'] == edf_file]
            
            for _, alarm in file_alarms.iterrows():
                alarm_time = alarm['alarm_time']
                
                # Valid prediction window: [alarm_time + SPH, alarm_time + SOP]
                valid_start = alarm_time + sph
                valid_end = alarm_time + sop
                
                if valid_start <= onset <= valid_end:
                    seizure_detected = True
                    break
            
            all_seizure_detected.append({
                'patient': patient,
                'edf_file': edf_file,
                'seizure_id': seizure_id,
                'onset_sec': onset,
                'detected': seizure_detected
            })
        
        # Track which alarms are successful
        for _, alarm in patient_alarms.iterrows():
            alarm_time = alarm['alarm_time']
            edf_file = alarm['edf_file']
            
            # Check if any seizure falls in valid window
            file_seizures = patient_seizures[patient_seizures['edf_file'] == edf_file]
            
            alarm_success = False
            matched_seizure = None
            
            for _, seizure in file_seizures.iterrows():
                onset = seizure['onset_sec']
                valid_start = alarm_time + sph
                valid_end = alarm_time + sop
                
                if valid_start <= onset <= valid_end:
                    alarm_success = True
                    matched_seizure = onset
                    break
            
            all_alarm_success.append({
                'patient': patient,
                'edf_file': edf_file,
                'alarm_time': alarm_time,
                'success': alarm_success,
                'matched_seizure': matched_seizure
            })
            
            # Add warning time
            if 'alarm_duration' in alarm:
                total_warning_time += alarm['alarm_duration']
            else:
                total_warning_time += sop  # Assume SOP duration
    
    # Compute metrics
    n_seizures = len(all_seizure_detected)
    n_detected = sum(1 for s in all_seizure_detected if s['detected'])
    n_alarms = len(all_alarm_success)
    n_true_alarms = sum(1 for a in all_alarm_success if a['success'])
    n_false_alarms = n_alarms - n_true_alarms
    
    results['n_seizures_total'] = n_seizures
    results['n_seizures_detected'] = n_detected
    results['n_alarms'] = n_alarms
    results['n_true_alarms'] = n_true_alarms
    results['n_false_alarms'] = n_false_alarms
    
    # Sensitivity
    if n_seizures > 0:
        results['sensitivity'] = n_detected / n_seizures
    
    # False alarm rate per 24 hours
    if total_recording_hours is not None and total_recording_hours > 0:
        results['false_alarm_rate_24h'] = (n_false_alarms / total_recording_hours) * 24
    
    # Time in warning
    if total_recording_hours is not None and total_recording_hours > 0:
        total_recording_seconds = total_recording_hours * 3600
        results['time_in_warning'] = total_warning_time / total_recording_seconds
    
    # Details
    results['detected_seizures'] = [s for s in all_seizure_detected if s['detected']]
    results['missed_seizures'] = [s for s in all_seizure_detected if not s['detected']]
    results['alarm_details'] = all_alarm_success
    
    return results


def compute_sensitivity(
    alarms_df: pd.DataFrame,
    seizure_index: pd.DataFrame,
    sph: float = 60.0,
    sop: float = 600.0
) -> float:
    """
    Compute seizure-wise sensitivity.
    
    Args:
        alarms_df: DataFrame with alarms
        seizure_index: DataFrame with seizures
        sph: Seizure Prediction Horizon
        sop: Seizure Occurrence Period
        
    Returns:
        Sensitivity (0-1)
    """
    metrics = compute_event_metrics(alarms_df, seizure_index, sph, sop)
    return metrics['sensitivity']


def compute_false_alarm_rate(
    alarms_df: pd.DataFrame,
    seizure_index: pd.DataFrame,
    total_recording_hours: float,
    sph: float = 60.0,
    sop: float = 600.0
) -> float:
    """
    Compute false alarm rate per 24 hours.
    
    Args:
        alarms_df: DataFrame with alarms
        seizure_index: DataFrame with seizures
        total_recording_hours: Total recording time in hours
        sph: Seizure Prediction Horizon
        sop: Seizure Occurrence Period
        
    Returns:
        False alarms per 24 hours
    """
    metrics = compute_event_metrics(
        alarms_df, seizure_index, sph, sop, total_recording_hours
    )
    return metrics['false_alarm_rate_24h']


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute window-level classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def generate_metrics_report(
    event_metrics: Dict,
    window_metrics: Optional[Dict] = None,
    cv_results: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> str:
    """
    Generate a text report of metrics.
    
    Args:
        event_metrics: Event-based metrics
        window_metrics: Window-level metrics (optional)
        cv_results: Cross-validation results (optional)
        config: Configuration parameters (optional)
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SEIZURE PREDICTION EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Configuration
    if config:
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"  SPH: {config.get('sph', 'N/A')} seconds")
        lines.append(f"  SOP: {config.get('sop', 'N/A')} seconds")
        lines.append(f"  Preictal Duration: {config.get('preictal_duration', 'N/A')} seconds")
        lines.append(f"  Window Length: {config.get('window_length', 'N/A')} seconds")
        lines.append(f"  Window Step: {config.get('window_step', 'N/A')} seconds")
        lines.append("")
    
    # Event-based metrics
    lines.append("EVENT-BASED METRICS")
    lines.append("-" * 40)
    lines.append(f"  Seizures Total: {event_metrics['n_seizures_total']}")
    lines.append(f"  Seizures Detected: {event_metrics['n_seizures_detected']}")
    lines.append(f"  Sensitivity: {event_metrics['sensitivity']:.1%}")
    lines.append(f"  Total Alarms: {event_metrics['n_alarms']}")
    lines.append(f"  True Alarms: {event_metrics['n_true_alarms']}")
    lines.append(f"  False Alarms: {event_metrics['n_false_alarms']}")
    lines.append(f"  False Alarm Rate: {event_metrics['false_alarm_rate_24h']:.2f} / 24h")
    lines.append(f"  Time in Warning: {event_metrics['time_in_warning']:.1%}")
    lines.append("")
    
    # Window-level metrics
    if window_metrics:
        lines.append("WINDOW-LEVEL METRICS")
        lines.append("-" * 40)
        lines.append(f"  Accuracy: {window_metrics['accuracy']:.3f}")
        lines.append(f"  Precision: {window_metrics['precision']:.3f}")
        lines.append(f"  Recall: {window_metrics['recall']:.3f}")
        lines.append(f"  F1 Score: {window_metrics['f1']:.3f}")
        if 'auc' in window_metrics:
            lines.append(f"  AUC: {window_metrics['auc']:.3f}")
        lines.append("")
    
    # CV results
    if cv_results and 'mean_f1' in cv_results:
        lines.append("CROSS-VALIDATION RESULTS (LOSO)")
        lines.append("-" * 40)
        lines.append(f"  Folds: {cv_results['n_folds']}")
        lines.append(f"  Mean Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        lines.append(f"  Mean Precision: {cv_results['mean_precision']:.3f} ± {cv_results['std_precision']:.3f}")
        lines.append(f"  Mean Recall: {cv_results['mean_recall']:.3f} ± {cv_results['std_recall']:.3f}")
        lines.append(f"  Mean F1: {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
        lines.append("")
    
    # Missed seizures
    if event_metrics.get('missed_seizures'):
        lines.append("MISSED SEIZURES")
        lines.append("-" * 40)
        for s in event_metrics['missed_seizures']:
            lines.append(f"  {s['patient']} / {s['edf_file']} @ {s['onset_sec']}s")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic data
    
    # Create synthetic seizure index
    seizure_index = pd.DataFrame([
        {'patient': 'chb01', 'edf_file': 'chb01_03.edf', 'seizure_id': 1, 'onset_sec': 2996, 'offset_sec': 3036},
        {'patient': 'chb01', 'edf_file': 'chb01_04.edf', 'seizure_id': 2, 'onset_sec': 1467, 'offset_sec': 1494},
    ])
    
    # Create synthetic alarms
    alarms_df = pd.DataFrame([
        {'patient': 'chb01', 'edf_file': 'chb01_03.edf', 'alarm_time': 2800, 'alarm_duration': 60},  # Should detect seizure 1
        {'patient': 'chb01', 'edf_file': 'chb01_04.edf', 'alarm_time': 500, 'alarm_duration': 60},   # False alarm
    ])
    
    # Compute metrics
    metrics = compute_event_metrics(
        alarms_df, seizure_index,
        sph=60, sop=600,
        total_recording_hours=2.0
    )
    
    print("Event Metrics:")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  False Alarms: {metrics['n_false_alarms']}")
    print(f"  FA Rate: {metrics['false_alarm_rate_24h']:.2f} / 24h")
    
    # Generate report
    report = generate_metrics_report(
        metrics,
        config={'sph': 60, 'sop': 600, 'preictal_duration': 1800}
    )
    print("\n" + report)
