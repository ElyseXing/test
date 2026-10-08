"""Shared classification reports and transparent activity proxies."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def spike_activity(spikes: np.ndarray) -> dict[str, float]:
    """Summarize event activity; these values are not measured energy."""
    return {
        "spike_count": float(spikes.sum()),
        "mean_firing_rate": float(spikes.mean()),
        "zero_spike_fraction": float(np.mean(spikes == 0)),
    }
