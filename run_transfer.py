#!/usr/bin/env python
"""
Run Transfer Learning Pipeline for Seizure Prediction.

Usage:
    python run_transfer.py --patients chb01 chb02 chb03
    python run_transfer.py  # Run on all available patients
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline_transfer import run_transfer_learning_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Learning Pipeline for Seizure Prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        help="Patient IDs to process (e.g., chb01 chb02). If not specified, processes all."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/transfer_learning",
        help="Output directory"
    )
    parser.add_argument(
        "--train",
        type=str,
        nargs="+",
        help="Patient IDs for training (e.g., chb01 chb02)"
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="+",
        help="Patient IDs for testing (unseen during training)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from pretrained model (skip pretraining, only do fine-tuning)"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    print(f"Config: {config_path.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    if args.train and args.test:
        print(f"Train patients: {args.train}")
        print(f"Test patients: {args.test}")
    else:
        print(f"Patients: {args.patients if args.patients else 'all'}")
    print()
    
    results = run_transfer_learning_pipeline(
        config_path=config_path,
        patients=args.patients,
        output_dir=output_dir,
        train_patients=args.train,
        test_patients=args.test,
        resume=args.resume
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
