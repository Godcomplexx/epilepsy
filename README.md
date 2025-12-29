# Seizure Prediction System

Patient-specific seizure prediction using Transfer Learning on EEG data from the CHB-MIT dataset.

## Overview

This system predicts epileptic seizures using a **CNN-LSTM architecture** with transfer learning:
1. **Pretrain** a global model on multiple patients
2. **Fine-tune** on each patient individually
3. **Personalized thresholds** for optimal sensitivity/specificity trade-off

### Key Features
- **Transfer Learning**: Global pretraining + patient-specific fine-tuning
- **Adaptive Learning Rate**: Automatically adjusts based on patient data size
- **Focal Loss**: Better handling of class imbalance (preictal << interictal)
- **Artifact Rejection**: Automatic removal of noisy windows
- **Personalized Thresholds**: Optimal threshold search per patient

## Project Structure

```
epilepsy/
├── config/
│   └── default.yaml          # Configuration parameters
├── src/
│   ├── data/
│   │   ├── index_builder.py  # Build seizure/file indices from CHB-MIT
│   │   ├── labeling.py       # Label windows (preictal/interictal)
│   │   ├── preprocessing.py  # EDF loading, filtering, resampling
│   │   └── segmentation.py   # Sliding window segmentation
│   ├── features/
│   │   └── extractor.py      # PSD, statistical, Hjorth, entropy features
│   ├── models/
│   │   ├── classifier.py     # Traditional ML classifiers (RF, XGB, SVM)
│   │   └── deep_model.py     # CNN-LSTM with attention, Focal Loss, MAML
│   ├── evaluation/
│   │   ├── alarm_logic.py    # Alarm generation with smoothing
│   │   └── metrics.py        # Sensitivity, FA/24h, event-based metrics
│   ├── pipeline.py           # Traditional ML pipeline
│   └── pipeline_transfer.py  # Transfer Learning pipeline
├── run.py                    # Run traditional ML pipeline
├── run_transfer.py           # Run Transfer Learning pipeline
└── requirements.txt          # Python dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data

This project uses the **CHB-MIT Scalp EEG Database**:
- 24 patients with intractable seizures
- 844 hours of continuous EEG recordings
- 198 seizures annotated

Download from: https://physionet.org/content/chbmit/1.0.0/

Place data in the path specified in `config/default.yaml`:
```yaml
paths:
  data_root: "path/to/chbmit/1.0.0"
```

## Usage

### Transfer Learning Pipeline (Recommended)

```bash
# Full pipeline with train/test split
python run_transfer.py --config config/default.yaml

# Specific patients only
python run_transfer.py --patients chb01 chb02 chb03

# Resume from pretrained model
python run_transfer.py --resume
```

### Traditional ML Pipeline

```bash
python run.py --config config/default.yaml --patients chb01 chb02
```

## Configuration

Key parameters in `config/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timing.sph` | 60s | Seizure Prediction Horizon |
| `timing.preictal_duration` | 1800s | Preictal period (30 min) |
| `windowing.window_length` | 4s | EEG window size |
| `deep_learning.pretrain_epochs` | 30 | Global model epochs |
| `deep_learning.finetune_epochs` | 20 | Patient fine-tuning epochs |
| `deep_learning.use_focal_loss` | true | Use Focal Loss for imbalance |

## Model Architecture

```
Input: (batch, 17 channels, 1024 samples)
    ↓
CNN Block 1: Conv1d(17→32) + BatchNorm + ReLU + MaxPool
CNN Block 2: Conv1d(32→64) + BatchNorm + ReLU + MaxPool  
CNN Block 3: Conv1d(64→128) + BatchNorm + ReLU + MaxPool
    ↓
LSTM: 2 layers, hidden=64, bidirectional
    ↓
Attention: Self-attention over time steps
    ↓
FC: 64 → 1 (sigmoid)
    ↓
Output: P(preictal)
```

## Results

### Latest Run (24 patients)

| Group | Patients | Seizures | Detected | Sensitivity | Mean FA/24h |
|-------|----------|----------|----------|-------------|-------------|
| TRAIN | 19 | 182 | 117 | 64.3% | 45.27 |
| TEST | 5 | 16 | 13 | **81.2%** | 34.31 |

**Best performers:**
- chb02: Sensitivity=100%, FA/24h=0.00, AUC=0.877
- chb10: Sensitivity=100%, FA/24h=0.00, AUC=0.777
- chb11 (TEST): Sensitivity=100%, FA/24h=2.92, AUC=0.776

## Evaluation Metrics

- **Sensitivity**: % of seizures with at least one alarm in prediction window
- **FA/24h**: False alarms per 24 hours of recording
- **AUC**: Area under ROC curve (window-level classification)

Prediction window: `[onset - SOP, onset - SPH]` = `[onset - 10min, onset - 1min]`

## Known Issues

Some patients show poor performance due to:
- **Atypical seizure patterns** (chb15: 0% sensitivity with 20 seizures)
- **Insufficient data** (chb18: only 1 file)
- **High inter-seizure variability** (chb12: 40 seizures, 47.5% sensitivity)

## Future Improvements

- [ ] Multi-task learning across patients
- [ ] Seizure type classification
- [ ] Online learning / adaptation
- [ ] Reduce false alarm rate with post-processing
- [ ] Add more entropy-based features

## References

1. CHB-MIT Database: Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection and treatment.
2. Focal Loss: Lin, T. Y., et al. (2017). Focal loss for dense object detection.

## License

MIT License
