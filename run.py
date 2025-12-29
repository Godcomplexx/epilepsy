"""
Entry point script for running the seizure prediction pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import run_pipeline, SeizurePredictionPipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Seizure Prediction Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--patients", 
        type=str, 
        nargs="+", 
        default=None,
        help="Patient IDs to process (e.g., chb01 chb02). If not specified, processes all."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Single patient ID to process"
    )
    
    args = parser.parse_args()
    
    # Handle single patient argument
    patients = args.patients
    if args.patient:
        patients = [args.patient]
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    print(f"Config: {config_path}")
    print(f"Patients: {patients or 'all'}")
    print()
    
    results = run_pipeline(config_path=config_path, patients=patients)
