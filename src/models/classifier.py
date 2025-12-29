"""
Module for seizure prediction model training and validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class SeizureClassifier:
    """
    Classifier for seizure prediction (preictal vs interictal).
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        random_seed: int = 42,
        **model_params
    ):
        """
        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'xgboost', 'mlp')
            random_seed: Random seed for reproducibility
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'class_weight': 'balanced',
                'random_state': self.random_seed,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)
            
        elif self.model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'class_weight': 'balanced',
                'probability': True,
                'random_state': self.random_seed
            }
            default_params.update(self.model_params)
            self.model = SVC(**default_params)
            
        elif self.model_type == 'xgboost':
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not installed")
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_seed,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBClassifier(**default_params)
            
        elif self.model_type == 'mlp':
            default_params = {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'max_iter': 500,
                'random_state': self.random_seed
            }
            default_params.update(self.model_params)
            self.model = MLPClassifier(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SeizureClassifier':
        """
        Fit the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Trained {self.model_type} on {len(y)} samples")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # AUC only if both classes present
        if len(np.unique(y)) > 1:
            metrics['auc'] = roc_auc_score(y, y_proba)
        else:
            metrics['auc'] = None
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances if available."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            return None
    
    def save(self, path: Path) -> None:
        """Save model to file."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'SeizureClassifier':
        """Load model from file."""
        import joblib
        data = joblib.load(path)
        
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.is_fitted = data['is_fitted']
        
        return classifier


def train_patient_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'random_forest',
    random_seed: int = 42,
    **model_params
) -> Tuple[SeizureClassifier, Dict]:
    """
    Train a patient-specific model.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of classifier
        random_seed: Random seed
        **model_params: Model parameters
        
    Returns:
        Tuple of (trained classifier, training metrics)
    """
    classifier = SeizureClassifier(
        model_type=model_type,
        random_seed=random_seed,
        **model_params
    )
    
    classifier.fit(X, y)
    metrics = classifier.evaluate(X, y)
    
    return classifier, metrics


def loso_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    seizure_ids: np.ndarray,
    model_type: str = 'random_forest',
    random_seed: int = 42,
    **model_params
) -> Dict:
    """
    Leave-One-Seizure-Out cross-validation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        seizure_ids: Seizure ID for each sample (n_samples,)
                     Samples with seizure_id=0 are interictal
        model_type: Type of classifier
        random_seed: Random seed
        **model_params: Model parameters
        
    Returns:
        Dictionary with CV results
    """
    unique_seizures = np.unique(seizure_ids[seizure_ids > 0])
    n_seizures = len(unique_seizures)
    
    if n_seizures < 2:
        logger.warning(f"Only {n_seizures} seizures, cannot perform LOSO CV")
        return {'error': 'insufficient_seizures'}
    
    logger.info(f"Running LOSO CV with {n_seizures} seizures")
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold_idx, test_seizure in enumerate(unique_seizures):
        # Test set: windows from this seizure
        test_mask = seizure_ids == test_seizure
        
        # Train set: all other windows (including interictal)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Skip if no test samples or only one class in train
        if len(y_test) == 0 or len(np.unique(y_train)) < 2:
            logger.warning(f"Skipping fold {fold_idx}: insufficient data")
            continue
        
        # Train model
        classifier = SeizureClassifier(
            model_type=model_type,
            random_seed=random_seed,
            **model_params
        )
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]
        
        fold_metrics = {
            'fold': fold_idx,
            'test_seizure': int(test_seizure),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        fold_results.append(fold_metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    results = {
        'n_folds': len(fold_results),
        'fold_results': fold_results,
        'overall': {
            'accuracy': accuracy_score(all_y_true, all_y_pred),
            'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
            'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
            'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist()
        }
    }
    
    if len(np.unique(all_y_true)) > 1:
        results['overall']['auc'] = roc_auc_score(all_y_true, all_y_proba)
    
    # Mean metrics across folds
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = [f[metric] for f in fold_results]
        results[f'mean_{metric}'] = np.mean(values)
        results[f'std_{metric}'] = np.std(values)
    
    return results


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 100
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (20% preictal)
    y = np.zeros(n_samples, dtype=int)
    y[:200] = 1
    
    # Generate seizure IDs (5 seizures, 40 preictal samples each)
    seizure_ids = np.zeros(n_samples, dtype=int)
    for i in range(5):
        seizure_ids[i*40:(i+1)*40] = i + 1
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    X, y, seizure_ids = X[perm], y[perm], seizure_ids[perm]
    
    # Test LOSO CV
    results = loso_cross_validation(X, y, seizure_ids)
    
    print("LOSO CV Results:")
    print(f"  Folds: {results['n_folds']}")
    print(f"  Mean Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"  Mean F1: {results['mean_f1']:.3f} ± {results['std_f1']:.3f}")
    print(f"  Overall AUC: {results['overall'].get('auc', 'N/A')}")
