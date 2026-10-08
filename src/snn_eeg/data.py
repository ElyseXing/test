"""Sleep-EDF loading and compact EEG feature extraction."""

from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

STAGE_EVENT_IDS = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 5,
    "Sleep stage R": 6,
}
EVENT_TO_STAGE = {1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "N3", 6: "REM"}
BANDS_HZ = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
}


def bandpower_features(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute log band power averaged over EEG channels for each epoch."""
    frequencies, power = welch(epochs, fs=sfreq, nperseg=min(256, epochs.shape[-1]))
    channel_mean_power = power.mean(axis=1)
    features = []
    for low, high in BANDS_HZ.values():
        mask = (frequencies >= low) & (frequencies < high)
        if not mask.any():
            raise ValueError(f"No Welch bins found for band {low}-{high} Hz")
        features.append(np.log(channel_mean_power[:, mask].mean(axis=1) + 1e-12))
    return np.column_stack(features)


def load_sleep_edf(
    subjects: Sequence[int] = (0, 1),
    recording: int = 1,
    data_path: str | Path | None = None,
    l_freq: float = 0.3,
    h_freq: float = 35.0,
    reject_peak_to_peak_uv: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download/read Sleep-EDF and return features, stage labels, subject IDs."""
    files = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(subjects), recording=[recording], path=data_path
    )
    feature_parts, label_parts, group_parts = [], [], []

    for subject, (psg_path, hypnogram_path) in zip(subjects, files):
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
        raw.set_annotations(mne.read_annotations(hypnogram_path), emit_warning=False)
        raw.pick(picks="eeg")
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

        events, _ = mne.events_from_annotations(
            raw, event_id=STAGE_EVENT_IDS, verbose="ERROR"
        )
        epochs = mne.Epochs(
            raw,
            events,
            event_id=STAGE_EVENT_IDS,
            tmin=0,
            tmax=30 - 1 / raw.info["sfreq"],
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            reject={"eeg": reject_peak_to_peak_uv * 1e-6},
            verbose="ERROR",
        )
        epoch_data = epochs.get_data(picks="eeg")
        feature_parts.append(bandpower_features(epoch_data, raw.info["sfreq"]))
        label_parts.append(np.array([EVENT_TO_STAGE[int(code)] for code in epochs.events[:, 2]]))
        group_parts.append(np.full(len(epochs), subject))

    return (
        np.concatenate(feature_parts),
        np.concatenate(label_parts),
        np.concatenate(group_parts),
    )
