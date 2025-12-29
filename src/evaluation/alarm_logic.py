"""
Module for alarm generation logic.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class AlarmGenerator:
    """
    Generates alarms from probability stream based on configurable logic.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        smoothing_window: int = 15,
        n_consecutive: int = 5,
        refractory_period: float = 300.0,
        window_step: float = 2.0
    ):
        """
        Args:
            threshold: Probability threshold for alarm
            smoothing_window: Number of windows for moving average smoothing
            n_consecutive: Number of consecutive windows above threshold
            refractory_period: Refractory period after alarm in seconds
            window_step: Time step between windows in seconds
        """
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.n_consecutive = n_consecutive
        self.refractory_period = refractory_period
        self.window_step = window_step
    
    def smooth_probabilities(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing to probabilities.
        
        Args:
            proba: Raw probability array
            
        Returns:
            Smoothed probability array
        """
        if self.smoothing_window <= 1:
            return proba
        
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = np.convolve(proba, kernel, mode='same')
        
        return smoothed
    
    def generate_alarms(
        self,
        proba: np.ndarray,
        times: np.ndarray
    ) -> List[Dict]:
        """
        Generate alarms from probability stream.
        
        Args:
            proba: Probability array (n_windows,)
            times: Time array in seconds (n_windows,)
            
        Returns:
            List of alarm dictionaries with 'time', 'proba', 'duration'
        """
        # Smooth probabilities
        smoothed = self.smooth_probabilities(proba)
        
        # Find windows above threshold
        above_threshold = smoothed >= self.threshold
        
        alarms = []
        last_alarm_time = -float('inf')
        consecutive_count = 0
        alarm_start_idx = None
        
        for i in range(len(smoothed)):
            if above_threshold[i]:
                consecutive_count += 1
                if alarm_start_idx is None:
                    alarm_start_idx = i
                
                # Check if we have enough consecutive windows
                if consecutive_count >= self.n_consecutive:
                    alarm_time = times[alarm_start_idx]
                    
                    # Check refractory period
                    if alarm_time - last_alarm_time >= self.refractory_period:
                        alarms.append({
                            'time': alarm_time,
                            'end_time': times[i],
                            'proba': float(smoothed[alarm_start_idx]),
                            'duration': times[i] - alarm_time,
                            'window_idx': alarm_start_idx
                        })
                        last_alarm_time = alarm_time
                        
                        # Reset after alarm
                        consecutive_count = 0
                        alarm_start_idx = None
            else:
                consecutive_count = 0
                alarm_start_idx = None
        
        return alarms
    
    def generate_alarms_for_file(
        self,
        proba: np.ndarray,
        window_times: List[Tuple[float, float]],
        patient: str,
        edf_file: str
    ) -> pd.DataFrame:
        """
        Generate alarms for a single file.
        
        Args:
            proba: Probability array
            window_times: List of (t_start, t_end) tuples
            patient: Patient ID
            edf_file: EDF filename
            
        Returns:
            DataFrame with alarm information
        """
        times = np.array([t[0] for t in window_times])
        alarms = self.generate_alarms(proba, times)
        
        records = []
        for alarm in alarms:
            records.append({
                'patient': patient,
                'edf_file': edf_file,
                'alarm_time': alarm['time'],
                'alarm_end_time': alarm['end_time'],
                'alarm_proba': alarm['proba'],
                'alarm_duration': alarm['duration']
            })
        
        return pd.DataFrame(records)


def apply_alarm_logic(
    predictions_df: pd.DataFrame,
    threshold: float = 0.5,
    smoothing_window: int = 15,
    n_consecutive: int = 5,
    refractory_period: float = 300.0,
    window_step: float = 2.0
) -> pd.DataFrame:
    """
    Apply alarm logic to prediction DataFrame.
    
    Args:
        predictions_df: DataFrame with columns:
            - patient, edf_file, t_start, t_end, proba
        threshold: Probability threshold
        smoothing_window: Smoothing window size
        n_consecutive: Consecutive windows required
        refractory_period: Refractory period in seconds
        window_step: Window step in seconds
        
    Returns:
        DataFrame with alarms
    """
    generator = AlarmGenerator(
        threshold=threshold,
        smoothing_window=smoothing_window,
        n_consecutive=n_consecutive,
        refractory_period=refractory_period,
        window_step=window_step
    )
    
    all_alarms = []
    
    # Process each file separately
    for (patient, edf_file), group in predictions_df.groupby(['patient', 'edf_file']):
        group = group.sort_values('t_start')
        
        proba = group['proba'].values
        window_times = list(zip(group['t_start'], group['t_end']))
        
        alarms_df = generator.generate_alarms_for_file(
            proba, window_times, patient, edf_file
        )
        all_alarms.append(alarms_df)
    
    if all_alarms:
        return pd.concat(all_alarms, ignore_index=True)
    else:
        return pd.DataFrame(columns=[
            'patient', 'edf_file', 'alarm_time', 'alarm_end_time',
            'alarm_proba', 'alarm_duration'
        ])


def evaluate_alarm(
    alarm_time: float,
    seizures: List[Dict],
    sph: float = 60.0,
    sop: float = 600.0
) -> Dict:
    """
    Evaluate if an alarm is successful.
    
    An alarm is successful if a seizure onset occurs:
    - After SPH (alarm_time + SPH)
    - Within SOP (alarm_time + SOP)
    
    Args:
        alarm_time: Time of alarm in seconds
        seizures: List of seizure dicts with 'onset_sec'
        sph: Seizure Prediction Horizon
        sop: Seizure Occurrence Period
        
    Returns:
        Dictionary with evaluation results
    """
    valid_start = alarm_time + sph
    valid_end = alarm_time + sop
    
    for seizure in seizures:
        onset = seizure['onset_sec']
        
        if valid_start <= onset <= valid_end:
            return {
                'success': True,
                'seizure_onset': onset,
                'time_to_seizure': onset - alarm_time,
                'within_sph': False
            }
        
        # Check if alarm is too close (within SPH)
        if alarm_time <= onset < valid_start:
            return {
                'success': False,
                'reason': 'within_sph',
                'seizure_onset': onset,
                'time_to_seizure': onset - alarm_time
            }
    
    return {
        'success': False,
        'reason': 'no_seizure_in_sop',
        'seizure_onset': None,
        'time_to_seizure': None
    }


def generate_alarms(
    windows_df: pd.DataFrame,
    threshold: float = 0.5,
    smoothing_window: int = 15,
    refractory_sec: float = 300.0,
    n_consecutive: int = 5,
    window_step: float = 2.0
) -> pd.DataFrame:
    """
    Convenience function to generate alarms from windows DataFrame.
    
    Args:
        windows_df: DataFrame with 'prob_preictal' and 'global_start' columns
        threshold: Probability threshold
        smoothing_window: Smoothing window size
        refractory_sec: Refractory period in seconds
        n_consecutive: Consecutive windows above threshold
        window_step: Window step in seconds
        
    Returns:
        DataFrame with alarms
    """
    if 'prob_preictal' not in windows_df.columns:
        logger.warning("No prob_preictal column in windows_df")
        return pd.DataFrame(columns=['time', 'proba', 'duration'])
    
    generator = AlarmGenerator(
        threshold=threshold,
        smoothing_window=smoothing_window,
        n_consecutive=n_consecutive,
        refractory_period=refractory_sec,
        window_step=window_step
    )
    
    proba = windows_df['prob_preictal'].values
    times = windows_df['global_start'].values if 'global_start' in windows_df.columns else np.arange(len(proba)) * window_step
    
    alarms = generator.generate_alarms(proba, times)
    
    return pd.DataFrame(alarms)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Simulate 1 hour of predictions
    n_windows = 1800  # 4s windows, 2s step = 1800 windows per hour
    times = np.arange(n_windows) * 2.0  # 2s step
    
    # Generate probabilities with a spike before a "seizure" at 2000s
    proba = np.random.uniform(0.1, 0.3, n_windows)
    
    # Add preictal pattern before seizure at 2000s
    seizure_onset = 2000
    preictal_start = seizure_onset - 1800  # 30 min before
    preictal_end = seizure_onset - 60  # SPH
    
    for i, t in enumerate(times):
        if preictal_start <= t <= preictal_end:
            # Gradually increase probability
            progress = (t - preictal_start) / (preictal_end - preictal_start)
            proba[i] = 0.3 + 0.5 * progress + np.random.uniform(-0.1, 0.1)
    
    proba = np.clip(proba, 0, 1)
    
    # Generate alarms
    generator = AlarmGenerator(
        threshold=0.5,
        smoothing_window=15,
        n_consecutive=5,
        refractory_period=300
    )
    
    alarms = generator.generate_alarms(proba, times)
    
    print(f"Generated {len(alarms)} alarms")
    for alarm in alarms:
        print(f"  Alarm at t={alarm['time']:.0f}s, proba={alarm['proba']:.2f}")
        
        # Evaluate
        result = evaluate_alarm(
            alarm['time'],
            [{'onset_sec': seizure_onset}],
            sph=60,
            sop=600
        )
        print(f"    Success: {result['success']}, Time to seizure: {result.get('time_to_seizure')}")
