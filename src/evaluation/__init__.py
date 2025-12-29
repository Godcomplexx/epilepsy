"""Evaluation and alarm logic modules."""

from .alarm_logic import AlarmGenerator, apply_alarm_logic
from .metrics import compute_event_metrics, compute_sensitivity, compute_false_alarm_rate
