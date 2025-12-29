"""
Main pipeline for seizure prediction.
Runs the complete workflow: index → preprocess → segment → label → features → train → evaluate
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from data.index_builder import build_seizure_index, build_file_index, save_index
from data.preprocessing import preprocess_edf, load_edf_channels
from data.segmentation import create_window_index, get_window_data, WindowGenerator
from data.labeling import WindowLabeler, label_windows, get_label_statistics
from features.extractor import extract_features, get_feature_names
from models.classifier import SeizureClassifier, loso_cross_validation
from evaluation.alarm_logic import AlarmGenerator, apply_alarm_logic
from evaluation.metrics import compute_event_metrics, generate_metrics_report


class SeizurePredictionPipeline:
    """
    End-to-end pipeline for seizure prediction.
    """
    
    def __init__(self, config_path: Optional[Path] = None, config: Optional[Dict] = None):
        """
        Args:
            config_path: Path to YAML config file
            config: Config dictionary (overrides config_path)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.data_root = Path(self.config['paths']['data_root'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.labeler = WindowLabeler(
            sph=self.config['timing']['sph'],
            preictal_duration=self.config['timing']['preictal_duration'],
            postictal_excluded=self.config['timing']['postictal_excluded'],
            interictal_gap=self.config['timing']['interictal_gap']
        )
        
        self.window_generator = WindowGenerator(
            window_length=self.config['windowing']['window_length'],
            window_step=self.config['windowing']['window_step'],
            sfreq=self.config['preprocessing']['target_sfreq']
        )
        
        self.alarm_generator = AlarmGenerator(
            threshold=self.config['alarm']['threshold'],
            smoothing_window=self.config['alarm']['smoothing_window'],
            n_consecutive=self.config['alarm']['n_consecutive'],
            refractory_period=self.config['alarm']['refractory_period'],
            window_step=self.config['windowing']['window_step']
        )
        
        # Results storage
        self.seizure_index = None
        self.file_index = None
        self.results = {}
    
    def run(self, patients: Optional[List[str]] = None) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            patients: List of patient IDs to process. If None, processes all.
            
        Returns:
            Dictionary with results
        """
        logger.info("Starting Seizure Prediction Pipeline")
        start_time = datetime.now()
        
        # Step 1: Build seizure index
        logger.info("Step 1: Building seizure index")
        self.seizure_index = build_seizure_index(self.data_root, patients)
        save_index(self.seizure_index, self.output_dir / "seizure_index.csv")
        
        if self.seizure_index.empty:
            logger.error("No seizures found, aborting")
            return {'error': 'no_seizures'}
        
        # Step 2: Build file index
        logger.info("Step 2: Building file index")
        self.file_index = build_file_index(self.data_root, patients)
        
        # Get list of patients to process
        if patients is None:
            patients = self.seizure_index['patient'].unique().tolist()
        
        # Step 3-9: Process each patient
        all_results = {}
        for patient in patients:
            logger.info(f"Processing patient: {patient}")
            try:
                patient_results = self.process_patient(patient)
                all_results[patient] = patient_results
            except Exception as e:
                logger.error(f"Error processing {patient}: {e}")
                all_results[patient] = {'error': str(e)}
        
        # Step 10: Generate summary report
        logger.info("Step 10: Generating summary report")
        self.generate_summary_report(all_results)
        
        elapsed = datetime.now() - start_time
        logger.info(f"Pipeline completed in {elapsed}")
        
        self.results = all_results
        return all_results
    
    def process_patient(self, patient: str) -> Dict:
        """
        Process a single patient.
        
        Args:
            patient: Patient ID (e.g., 'chb01')
            
        Returns:
            Dictionary with patient results
        """
        patient_dir = self.output_dir / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Get patient's seizures and files
        patient_seizures = self.seizure_index[self.seizure_index['patient'] == patient]
        patient_files = self.file_index[self.file_index['patient'] == patient]
        
        logger.info(f"  {patient}: {len(patient_seizures)} seizures, {len(patient_files)} files")
        
        # Step 3-4: Preprocess and segment
        all_windows = []
        all_features = []
        all_labels = []
        all_seizure_ids = []
        
        for _, file_row in tqdm(patient_files.iterrows(), total=len(patient_files), desc=f"  {patient}"):
            edf_path = Path(file_row['edf_path'])
            
            if not edf_path.exists():
                continue
            
            try:
                # Preprocess
                data, sfreq, ch_names, qc = preprocess_edf(
                    edf_path,
                    channels=self.config['channels']['selected'],
                    target_sfreq=self.config['preprocessing']['target_sfreq'],
                    bandpass_low=self.config['preprocessing']['bandpass_low'],
                    bandpass_high=self.config['preprocessing']['bandpass_high'],
                    notch_freq=self.config['preprocessing']['notch_freq']
                )
                
                if qc and not qc['is_good']:
                    logger.warning(f"  QC failed for {edf_path.name}")
                
                # Get seizures for this file
                file_seizures = patient_seizures[
                    patient_seizures['edf_file'] == file_row['edf_file']
                ][['onset_sec', 'offset_sec', 'seizure_id']].to_dict('records')
                
                # Segment and label
                window_index = create_window_index(
                    data, sfreq,
                    self.config['windowing']['window_length'],
                    self.config['windowing']['window_step'],
                    patient, file_row['edf_file']
                )
                
                for _, win_row in window_index.iterrows():
                    # Get label
                    seizures_for_label = [{'onset_sec': s['onset_sec'], 'offset_sec': s['offset_sec']} 
                                          for s in file_seizures]
                    label, reason = self.labeler.get_window_label(
                        win_row['t_start'], win_row['t_end'], seizures_for_label
                    )
                    
                    # Skip excluded windows
                    if label == -1:
                        continue
                    
                    # Extract features
                    window_data = get_window_data(
                        data, win_row['start_sample'], win_row['end_sample']
                    )
                    
                    features = extract_features(
                        window_data, sfreq,
                        bands=self.config['features']['bands']
                    )
                    
                    # Determine seizure ID for LOSO
                    seizure_id = 0  # interictal
                    if label == 1:  # preictal
                        for s in file_seizures:
                            if win_row['t_end'] <= s['onset_sec'] - self.config['timing']['sph']:
                                seizure_id = s['seizure_id']
                                break
                    
                    all_windows.append({
                        'patient': patient,
                        'edf_file': file_row['edf_file'],
                        't_start': win_row['t_start'],
                        't_end': win_row['t_end']
                    })
                    all_features.append(features)
                    all_labels.append(label)
                    all_seizure_ids.append(seizure_id)
                    
            except Exception as e:
                logger.warning(f"  Error processing {edf_path.name}: {e}")
                continue
        
        if len(all_features) == 0:
            logger.warning(f"  No valid windows for {patient}")
            return {'error': 'no_windows'}
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        seizure_ids = np.array(all_seizure_ids)
        windows_df = pd.DataFrame(all_windows)
        
        # Log statistics
        stats = get_label_statistics(pd.DataFrame({'label': y}))
        logger.info(f"  Windows: {stats['total_windows']} (preictal: {stats['preictal']}, interictal: {stats['interictal']})")
        
        # Step 7: LOSO Cross-validation
        logger.info(f"  Running LOSO cross-validation")
        cv_results = loso_cross_validation(
            X, y, seizure_ids,
            model_type=self.config['training']['model_type'],
            random_seed=self.config['training']['random_seed']
        )
        
        # Step 8: Train final model on all data
        logger.info(f"  Training final model")
        classifier = SeizureClassifier(
            model_type=self.config['training']['model_type'],
            random_seed=self.config['training']['random_seed']
        )
        classifier.fit(X, y)
        classifier.save(patient_dir / "model.joblib")
        
        # Step 9: Generate predictions and alarms
        logger.info(f"  Generating predictions and alarms")
        proba = classifier.predict_proba(X)[:, 1]
        windows_df['proba'] = proba
        windows_df['label'] = y
        
        alarms_df = apply_alarm_logic(
            windows_df,
            threshold=self.config['alarm']['threshold'],
            smoothing_window=self.config['alarm']['smoothing_window'],
            n_consecutive=self.config['alarm']['n_consecutive'],
            refractory_period=self.config['alarm']['refractory_period']
        )
        
        # Compute event metrics
        total_hours = len(patient_files)  # Approximate: 1 hour per file
        event_metrics = compute_event_metrics(
            alarms_df, patient_seizures,
            sph=self.config['timing']['sph'],
            sop=self.config['timing']['sop'],
            total_recording_hours=total_hours
        )
        
        # Save results
        windows_df.to_csv(patient_dir / "windows.csv", index=False)
        alarms_df.to_csv(patient_dir / "alarms.csv", index=False)
        
        with open(patient_dir / "cv_results.json", 'w') as f:
            # Convert numpy types for JSON serialization
            cv_json = json.loads(json.dumps(cv_results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x))
            json.dump(cv_json, f, indent=2)
        
        with open(patient_dir / "event_metrics.json", 'w') as f:
            event_json = json.loads(json.dumps(event_metrics, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x))
            json.dump(event_json, f, indent=2)
        
        # Generate patient report
        report = generate_metrics_report(
            event_metrics,
            window_metrics=cv_results.get('overall'),
            cv_results=cv_results,
            config={
                'sph': self.config['timing']['sph'],
                'sop': self.config['timing']['sop'],
                'preictal_duration': self.config['timing']['preictal_duration'],
                'window_length': self.config['windowing']['window_length'],
                'window_step': self.config['windowing']['window_step']
            }
        )
        
        with open(patient_dir / "report.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"  {patient} complete: Sensitivity={event_metrics['sensitivity']:.1%}, FA/24h={event_metrics['false_alarm_rate_24h']:.2f}")
        
        return {
            'n_windows': len(windows_df),
            'n_preictal': stats['preictal'],
            'n_interictal': stats['interictal'],
            'cv_results': cv_results,
            'event_metrics': event_metrics
        }
    
    def generate_summary_report(self, all_results: Dict) -> None:
        """Generate summary report across all patients."""
        lines = []
        lines.append("=" * 70)
        lines.append("SEIZURE PREDICTION PIPELINE - SUMMARY REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        
        # Configuration summary
        lines.append("CONFIGURATION")
        lines.append("-" * 50)
        lines.append(f"  Data Root: {self.config['paths']['data_root']}")
        lines.append(f"  SPH: {self.config['timing']['sph']}s")
        lines.append(f"  SOP: {self.config['timing']['sop']}s")
        lines.append(f"  Preictal Duration: {self.config['timing']['preictal_duration']}s")
        lines.append(f"  Model: {self.config['training']['model_type']}")
        lines.append("")
        
        # Per-patient results
        lines.append("PATIENT RESULTS")
        lines.append("-" * 50)
        lines.append(f"{'Patient':<10} {'Seizures':<10} {'Sensitivity':<12} {'FA/24h':<10} {'CV F1':<10}")
        lines.append("-" * 50)
        
        total_seizures = 0
        total_detected = 0
        all_sensitivities = []
        all_fa_rates = []
        all_f1s = []
        
        for patient, result in sorted(all_results.items()):
            if 'error' in result:
                lines.append(f"{patient:<10} ERROR: {result['error']}")
                continue
            
            em = result['event_metrics']
            cv = result['cv_results']
            
            sens = em['sensitivity']
            fa_rate = em['false_alarm_rate_24h']
            f1 = cv.get('mean_f1', 0)
            
            total_seizures += em['n_seizures_total']
            total_detected += em['n_seizures_detected']
            all_sensitivities.append(sens)
            all_fa_rates.append(fa_rate)
            all_f1s.append(f1)
            
            lines.append(f"{patient:<10} {em['n_seizures_total']:<10} {sens:<12.1%} {fa_rate:<10.2f} {f1:<10.3f}")
        
        lines.append("-" * 50)
        
        # Aggregate metrics
        if all_sensitivities:
            lines.append("")
            lines.append("AGGREGATE METRICS")
            lines.append("-" * 50)
            lines.append(f"  Total Seizures: {total_seizures}")
            lines.append(f"  Total Detected: {total_detected}")
            lines.append(f"  Overall Sensitivity: {total_detected/total_seizures:.1%}" if total_seizures > 0 else "  Overall Sensitivity: N/A")
            lines.append(f"  Mean Sensitivity: {np.mean(all_sensitivities):.1%} ± {np.std(all_sensitivities):.1%}")
            lines.append(f"  Mean FA/24h: {np.mean(all_fa_rates):.2f} ± {np.std(all_fa_rates):.2f}")
            lines.append(f"  Mean CV F1: {np.mean(all_f1s):.3f} ± {np.std(all_f1s):.3f}")
        
        lines.append("")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        # Save report
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {self.output_dir / 'summary_report.txt'}")
        print(report)


def run_pipeline(
    config_path: Optional[Path] = None,
    patients: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to run the pipeline.
    
    Args:
        config_path: Path to config file
        patients: List of patients to process
        
    Returns:
        Results dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    
    pipeline = SeizurePredictionPipeline(config_path=config_path)
    return pipeline.run(patients=patients)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Seizure Prediction Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--patients", type=str, nargs="+", default=None, help="Patient IDs to process")
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    results = run_pipeline(config_path=config_path, patients=args.patients)
