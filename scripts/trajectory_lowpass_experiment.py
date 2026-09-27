r"""只改变训练目标的 2 Hz FIR 低通，保持 EEG、岭回归和留一被试划分不变。

运行：python scripts/trajectory_lowpass_experiment.py --data-root D:\biosignal-data\WAY-EEG-GAL

重要：所有主要 MAE 都相对于同一份未经低通的测试轨迹计算，避免仅因
测试轨迹被平滑就造成指标“改善”。这是探索性单变量实验，不是论文
完整预处理复现；论文未给出足以确定本实验 FIR 长度的全部实现细节。
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import firwin

from subject_holdout_baseline import (
    FS,
    N_AFTER,
    RIDGE_ALPHA,
    WRIST_AXES,
    extract_subject,
    make_features,
    read_mat,
)


SUBJECTS = (1, 2, 3, 4)
FIR_TAPS = 501  # 奇数长度、Hamming 窗；只平滑运动学，不动 EEG。
FIR_COEFF = firwin(FIR_TAPS, cutoff=2.0, fs=FS, window="hamming")


def filtered_targets(data_root, subject):
    """先平滑完整的约 9 秒 WS 轨迹，再截运动开始后的 1 秒。"""
    directory = data_root / f"P{subject}"
    events = read_mat(directory / f"P{subject}_AllLifts.mat", "P")
    columns = list(events.ColNames)
    run_col = columns.index("Run")
    lift_col = columns.index("Lift")
    onset_col = columns.index("tHandStart")
    trials = []

    for run in range(1, 10):
        hs = read_mat(directory / f"HS_P{subject}_S{run}.mat", "hs")
        ws = read_mat(directory / f"WS_P{subject}_S{run}.mat", "ws")
        kin_names = list(hs.kin.names)
        wrist_cols = [
            next(i for i, name in enumerate(kin_names) if str(name).startswith(axis + " "))
            for axis in WRIST_AXES
        ]
        rows = events.AllLifts[events.AllLifts[:, run_col] == run]
        if len(rows) != len(ws.win):
            raise ValueError(f"P{subject} S{run}: 事件数与窗口数不同")

        for row in rows:
            lift = int(row[lift_col])
            window = ws.win[lift - 1]
            onset = float(row[onset_col])
            index = np.searchsorted(window.eeg_t, onset)
            # 对完整窗口一次性做对称 FIR 卷积；只使用目标轨迹数据，
            # 并用镜像边界处理，避免在截出的 1 秒两端直接滤波。
            wrist = window.kin[:, wrist_cols].astype(float)
            smooth = convolve1d(wrist, FIR_COEFF, axis=0, mode="reflect")
            target = smooth[index:index + N_AFTER]
            if target.shape != (N_AFTER, 3) or not np.isfinite(target).all():
                raise ValueError(f"P{subject} S{run} lift {lift}: 滤波目标无效")
            trials.append(target - target[0])

    return np.stack(trials)


def predict(train_features, train_targets, test_features):
    """与既有基线脚本完全一致的训练集标准化、固定 alpha 岭回归。"""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale

    target_flat = train_targets.reshape(len(train_targets), -1)
    target_mean = target_flat.mean(axis=0)
    weights = np.linalg.solve(
        train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]),
        train.T @ (target_flat - target_mean),
    )
    baseline = train_targets.mean(axis=0)
    model = (test @ weights + target_mean).reshape(len(test_features), N_AFTER, 3)
    return baseline, model


def scores(baseline, prediction, raw_test):
    """两种模型都用同一原始测试轨迹打分。"""
    base_error = np.abs(baseline[None, :, :] - raw_test)
    model_error = np.abs(prediction - raw_test)
    return (
        base_error.mean(axis=(0, 1)),
        model_error.mean(axis=(0, 1)),
        np.sum(model_error.mean(axis=(1, 2)) < base_error.mean(axis=(1, 2))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data = {}

    for subject in SUBJECTS:
        eeg, raw, runs = extract_subject(args.data_root, subject)
        smooth = filtered_targets(args.data_root, subject)
        if smooth.shape != raw.shape:
            raise ValueError(f"P{subject}: 平滑前后试次没有对齐")
        data[subject] = (make_features(eeg), raw, smooth)
        difference = np.abs(raw - smooth).mean(axis=(0, 1))
        print(
            f"P{subject}: {len(raw)} 次，runs={runs}，"
            f"低通前后平均绝对差(XYZ)={np.round(difference, 3)}"
        )

    for test_subject in SUBJECTS:
        others = [s for s in SUBJECTS if s != test_subject]
        x_train = np.concatenate([data[s][0] for s in others])
        x_test, raw_test, _ = data[test_subject]
        print(f"\n留出 P{test_subject}，两方案均对同一原始测试目标评分：")
        for label, target_index in (("原始目标", 1), ("仅训练目标2Hz低通", 2)):
            y_train = np.concatenate([data[s][target_index] for s in others])
            baseline, prediction = predict(x_train, y_train, x_test)
            base_xyz, eeg_xyz, wins = scores(baseline, prediction, raw_test)
            print(
                f"  {label}: 基线XYZ={np.round(base_xyz, 3)}；"
                f"EEG XYZ={np.round(eeg_xyz, 3)}；"
                f"整体={base_xyz.mean():.3f}/{eeg_xyz.mean():.3f}；"
                f"EEG获益={wins}/{len(raw_test)}"
            )


if __name__ == "__main__":
    main()
