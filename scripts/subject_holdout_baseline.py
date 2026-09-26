"""Evaluate the current EEG ridge baseline with one participant held out.

This is an exploratory baseline, not the paper's full preprocessing pipeline.
Each trial uses 0.3 s of pre-movement EEG and 1.0 s of wrist displacement.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt


PAPER_CHANNELS = (
    "F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "O1", "Oz", "O2",
)
WRIST_AXES = ("Px4", "Py4", "Pz4")
FS = 500
N_BEFORE = 150
N_AFTER = 500
RIDGE_ALPHA = 100.0


def read_mat(path, key):
    return loadmat(path, struct_as_record=False, squeeze_me=True)[key]


def extract_subject(data_root, subject):
    directory = data_root / f"P{subject}"
    events = read_mat(directory / f"P{subject}_AllLifts.mat", "P")
    columns = list(events.ColNames)
    run_col, lift_col, start_col, onset_col = (
        columns.index(name)
        for name in ("Run", "Lift", "StartTime", "tHandStart")
    )
    filter_sos = butter(
        4, [0.1, 40], btype="bandpass", fs=FS, output="sos"
    )
    eeg_trials, wrist_trials, counts = [], [], []

    for run in range(1, 10):
        hs = read_mat(directory / f"HS_P{subject}_S{run}.mat", "hs")
        ws = read_mat(directory / f"WS_P{subject}_S{run}.mat", "ws")
        eeg_names = list(hs.eeg.names)
        kin_names = list(hs.kin.names)
        eeg_cols = [eeg_names.index(name) for name in PAPER_CHANNELS]
        wrist_cols = [
            next(
                i for i, name in enumerate(kin_names)
                if str(name).startswith(axis + " ")
            )
            for axis in WRIST_AXES
        ]
        c3_col = eeg_names.index("C3")

        # Average reference and forward-only filter on the continuous run.
        eeg = hs.eeg.sig[:, eeg_cols].astype(float)
        eeg -= eeg.mean(axis=1, keepdims=True)
        eeg = sosfilt(filter_sos, eeg, axis=0)
        rows = events.AllLifts[events.AllLifts[:, run_col] == run]
        if len(rows) != len(ws.win):
            raise ValueError(f"P{subject} S{run}: event/window count mismatch")
        counts.append(len(rows))

        for row in rows:
            lift = int(row[lift_col])
            window = ws.win[lift - 1]
            onset = float(row[onset_col])
            if not np.isfinite(onset):
                raise ValueError(f"P{subject} S{run} lift {lift}: missing onset")

            window_start = round((float(row[start_col]) - 2.0) * FS)
            if not np.array_equal(
                hs.eeg.sig[window_start:window_start + 10, c3_col],
                window.eeg[:10, c3_col],
            ):
                raise ValueError(f"P{subject} S{run} lift {lift}: EEG misaligned")
            onset_in_window = np.searchsorted(window.eeg_t, onset)
            onset_in_run = window_start + onset_in_window

            x = eeg[onset_in_run - N_BEFORE:onset_in_run]
            y = window.kin[
                onset_in_window:onset_in_window + N_AFTER, wrist_cols
            ]
            if x.shape != (N_BEFORE, len(PAPER_CHANNELS)) or y.shape != (N_AFTER, 3):
                raise ValueError(f"P{subject} S{run} lift {lift}: short trial")
            y = y - y[0]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"P{subject} S{run} lift {lift}: non-finite value")
            eeg_trials.append(x)
            wrist_trials.append(y)

    return np.stack(eeg_trials), np.stack(wrist_trials), counts


def make_features(eeg):
    # Three 0.1 s windows, with one log-standard-deviation per channel.
    n_trials = len(eeg)
    return np.log(
        np.std(eeg.reshape(n_trials, 3, 50, len(PAPER_CHANNELS)), axis=2)
        + 1e-8
    ).reshape(n_trials, -1)


def evaluate(features_train, targets_train, features_test, targets_test):
    mean = features_train.mean(axis=0)
    scale = features_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (features_train - mean) / scale
    test = (features_test - mean) / scale

    baseline = np.broadcast_to(targets_train.mean(axis=0), targets_test.shape)
    target_flat = targets_train.reshape(len(targets_train), -1)
    target_mean = target_flat.mean(axis=0)
    weights = np.linalg.solve(
        train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]),
        train.T @ (target_flat - target_mean),
    )
    prediction = (test @ weights + target_mean).reshape(targets_test.shape)

    baseline_error = np.abs(baseline - targets_test)
    model_error = np.abs(prediction - targets_test)
    return (
        baseline_error.mean(axis=(0, 1)),
        model_error.mean(axis=(0, 1)),
        int(np.sum(
            model_error.mean(axis=(1, 2))
            < baseline_error.mean(axis=(1, 2))
        )),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    args = parser.parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    if len(subjects) < 2:
        parser.error("Provide at least two distinct subjects")

    data = {}
    for subject in subjects:
        eeg, targets, counts = extract_subject(args.data_root, subject)
        data[subject] = (make_features(eeg), targets)
        print(f"P{subject}: {len(eeg)} trials, runs={counts}")

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        train_features = np.concatenate([data[s][0] for s in train_subjects])
        train_targets = np.concatenate([data[s][1] for s in train_subjects])
        test_features, test_targets = data[test_subject]
        baseline, model, wins = evaluate(
            train_features, train_targets, test_features, test_targets
        )
        print(
            f"test=P{test_subject} train={train_subjects} "
            f"baseline_xyz={np.round(baseline, 3)} "
            f"eeg_xyz={np.round(model, 3)} "
            f"overall={baseline.mean():.3f}/{model.mean():.3f} "
            f"wins={wins}/{len(test_targets)}"
        )


if __name__ == "__main__":
    main()
