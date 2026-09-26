"""用“留一受试者”方式评估当前 EEG 岭回归入门基线。

示例：python scripts/subject_holdout_baseline.py --data-root 数据目录 --subjects 1 2 3
每一轮留出一整个人测试，其他人的抓握全部用于训练；随后换人重复。
每次抓握用运动前 0.3 秒 EEG 预测运动后 1 秒的三维手腕位移。

这是探索性基线：尚未实现论文的全部频带、重采样、轨迹处理和模型。
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt


# 按名称选论文所用的 21 个 EEG 通道；不依赖不同文件里的列顺序。
PAPER_CHANNELS = (
    "F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "O1", "Oz", "O2",
)
WRIST_AXES = ("Px4", "Py4", "Pz4")  # 手腕在三个坐标方向的位置
FS = 500             # 原始 EEG 和运动学数据每秒均有 500 个采样点
N_BEFORE = 150       # 运动前 150/500 = 0.3 秒 EEG
N_AFTER = 500        # 运动后 500/500 = 1.0 秒手腕轨迹
RIDGE_ALPHA = 100.0  # 与先前 P1 基线一致，预先固定；不根据测试者调参


def read_mat(path, key):
    """读取 MATLAB 文件中名为 key 的结构体，例如 hs、ws 或 P。"""
    # squeeze_me 去掉多余的单元素维度，便于使用 hs.eeg.sig 等属性。
    return loadmat(path, struct_as_record=False, squeeze_me=True)[key]


def extract_subject(data_root, subject):
    """返回某人的 (试次×150×21 EEG, 试次×500×3 位移, 各段试次数)。"""
    directory = data_root / f"P{subject}"
    # AllLifts 每行是一回抓握，ColNames 给出 Run、Lift 和事件时间所在列。
    events = read_mat(directory / f"P{subject}_AllLifts.mat", "P")
    columns = list(events.ColNames)
    run_col, lift_col, start_col, onset_col = (
        columns.index(name)
        for name in ("Run", "Lift", "StartTime", "tHandStart")
    )
    # 4 阶 Butterworth 0.1–40 Hz 带通；sos 是数值较稳定的滤波器表示。
    filter_sos = butter(
        4, [0.1, 40], btype="bandpass", fs=FS, output="sos"
    )
    eeg_trials, wrist_trials, counts = [], [], []

    # S1–S9 是九段正式记录；ST 是另一种记录，这里不纳入当前基线。
    for run in range(1, 10):
        # HS 是整段连续记录；WS 是已经按抓握切开的窗口。
        hs = read_mat(directory / f"HS_P{subject}_S{run}.mat", "hs")
        ws = read_mat(directory / f"WS_P{subject}_S{run}.mat", "ws")
        eeg_names = list(hs.eeg.names)
        kin_names = list(hs.kin.names)
        eeg_cols = [eeg_names.index(name) for name in PAPER_CHANNELS]
        # 运动学列名可能带有额外后缀，所以匹配 "Px4 " 等前缀。
        wrist_cols = [
            next(
                i for i, name in enumerate(kin_names)
                if str(name).startswith(axis + " ")
            )
            for axis in WRIST_AXES
        ]
        c3_col = eeg_names.index("C3")

        # 原始矩阵的形状是“时间点 × 通道”。每个时间点减去 21 通道均值，
        # 得到平均重参考后的信号；这一步不会改变时间点或通道数量。
        eeg = hs.eeg.sig[:, eeg_cols].astype(float)
        eeg -= eeg.mean(axis=1, keepdims=True)
        # 在连续记录上从前往后滤波，再切试次；sosfilt 不使用未来采样。
        # 当前九段模型未使用 Notebook 中仅对 P1 S1 探索过的 ICA 清理。
        eeg = sosfilt(filter_sos, eeg, axis=0)
        # 必须按这段自己的事件行数遍历：例如 P2 S1 是 28 次，P1 S1 是 34 次。
        rows = events.AllLifts[events.AllLifts[:, run_col] == run]
        if len(rows) != len(ws.win):
            raise ValueError(f"P{subject} S{run}: event/window count mismatch")
        counts.append(len(rows))

        for row in rows:
            lift = int(row[lift_col])
            # MATLAB 的 Lift 从 1 开始；Python 数组索引从 0 开始。
            window = ws.win[lift - 1]
            # tHandStart 是相对于本次 WS 窗口时间轴的运动开始时刻。
            onset = float(row[onset_col])
            if not np.isfinite(onset):
                raise ValueError(f"P{subject} S{run} lift {lift}: missing onset")

            # 已核对过的数据约定：WS 窗口从 StartTime 前 2 秒开始。
            # 将“秒”乘采样率，换成该窗口在 HS 连续记录中的起点索引。
            window_start = round((float(row[start_col]) - 2.0) * FS)
            # 对比 C3 的前 10 个原始采样点，防止事件行和窗口对错位。
            if not np.array_equal(
                hs.eeg.sig[window_start:window_start + 10, c3_col],
                window.eeg[:10, c3_col],
            ):
                raise ValueError(f"P{subject} S{run} lift {lift}: EEG misaligned")
            # 同一个运动开始时刻有两种索引：在 WS 窗口中、在 HS 连续记录中。
            onset_in_window = np.searchsorted(window.eeg_t, onset)
            onset_in_run = window_start + onset_in_window

            # 输入 x 只取运动前样本，截止于运动开始的前一个采样点。
            # 目标 y 从运动开始取 1 秒，三个列依次对应 X、Y、Z。
            x = eeg[onset_in_run - N_BEFORE:onset_in_run]
            y = window.kin[
                onset_in_window:onset_in_window + N_AFTER, wrist_cols
            ]
            if x.shape != (N_BEFORE, len(PAPER_CHANNELS)) or y.shape != (N_AFTER, 3):
                raise ValueError(f"P{subject} S{run} lift {lift}: short trial")
            # 每次抓握都减去自己的起始位置：预测的是位移而非绝对坐标。
            y = y - y[0]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"P{subject} S{run} lift {lift}: non-finite value")
            eeg_trials.append(x)
            wrist_trials.append(y)

    # list 中每个元素是一回抓握；stack 后在最前面增加“试次”维度。
    return np.stack(eeg_trials), np.stack(wrist_trials), counts


def make_features(eeg):
    """把每回 (150,21) EEG 压缩为 3×21=63 个波动强度特征。"""
    n_trials = len(eeg)
    # 150 点分成三个连续的 50 点时间窗，每窗恰好 0.1 秒。
    # reshape 后各轴依次为：试次、时间窗、窗内时间点、通道。
    # axis=2 表示沿窗内 50 个点计算标准差，结果是 (试次,3,21)。
    # 取 log 压缩不同通道的数值范围；1e-8 防止对零取对数。
    return np.log(
        np.std(eeg.reshape(n_trials, 3, 50, len(PAPER_CHANNELS)), axis=2)
        + 1e-8
    ).reshape(n_trials, -1)


def evaluate(features_train, targets_train, features_test, targets_test):
    """只用训练者拟合基线和岭回归，返回两者的 MAE 及模型胜出的试次数。"""
    # 下面的均值和标准差只看训练者，测试者不能参与参数估计。
    mean = features_train.mean(axis=0)
    scale = features_train.std(axis=0)
    # 恒定特征的标准差可能为零，改成 1 可避免除零。
    scale[scale < 1e-12] = 1.0
    train = (features_train - mean) / scale
    test = (features_test - mean) / scale

    # 基线不看 EEG：对所有训练抓握的轨迹逐时间点求平均，
    # 再把这条相同的“平均轨迹”用于每一回测试抓握。
    baseline = np.broadcast_to(targets_train.mean(axis=0), targets_test.shape)
    # 每回目标原为 500 时间点 × 3 方向；摊平为 1500 个输出数值。
    target_flat = targets_train.reshape(len(targets_train), -1)
    target_mean = target_flat.mean(axis=0)
    # 岭回归同时学习从 63 个 EEG 特征到 1500 个轨迹数值的映射。
    # 等价于求解 (XᵀX + αI)W = Xᵀ(Y - 训练目标均值)，
    # solve 比显式求逆更稳定；target_mean 充当每个输出的截距。
    weights = np.linalg.solve(
        train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]),
        train.T @ (target_flat - target_mean),
    )
    prediction = (test @ weights + target_mean).reshape(targets_test.shape)

    # 先对每个试次、时间点、方向求绝对误差。
    baseline_error = np.abs(baseline - targets_test)
    model_error = np.abs(prediction - targets_test)
    return (
        # axis=(0,1)：对试次和时间求平均，留下 X/Y/Z 各一个 MAE。
        baseline_error.mean(axis=(0, 1)),
        model_error.mean(axis=(0, 1)),
        # axis=(1,2)：每次抓握在全部时间点、三方向上的整体 MAE。
        # 逐次比较模型与基线，再统计模型误差更小的抓握数。
        int(np.sum(
            model_error.mean(axis=(1, 2))
            < baseline_error.mean(axis=(1, 2))
        )),
    )


def main():
    """读取命令行参数，提取数据，再轮流留出每一位指定受试者。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # data-root 是 P1、P2 等文件夹共同所在的目录。
    parser.add_argument("--data-root", type=Path, required=True)
    # 可以传 --subjects 1 2 3；以后有新数据再追加编号。
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    args = parser.parse_args()
    # 保留输入顺序，同时去掉重复编号。
    subjects = list(dict.fromkeys(args.subjects))
    if len(subjects) < 2:
        parser.error("Provide at least two distinct subjects")

    data = {}
    for subject in subjects:
        eeg, targets, counts = extract_subject(args.data_root, subject)
        # 缓存每人的 63 维特征和真实轨迹，避免每轮重新读大型 .mat 文件。
        data[subject] = (make_features(eeg), targets)
        print(f"P{subject}: {len(eeg)} trials, runs={counts}")

    for test_subject in subjects:
        # 这一轮只用其他受试者训练；当前受试者的所有抓握都用作测试。
        train_subjects = [s for s in subjects if s != test_subject]
        train_features = np.concatenate([data[s][0] for s in train_subjects])
        train_targets = np.concatenate([data[s][1] for s in train_subjects])
        test_features, test_targets = data[test_subject]
        baseline, model, wins = evaluate(
            train_features, train_targets, test_features, test_targets
        )
        # overall 是三个方向 MAE 的平均：前一个数为基线，后一个为模型。
        # wins 是模型逐次 MAE 更小的抓握数，而非“预测正确的类别数”。
        print(
            f"test=P{test_subject} train={train_subjects} "
            f"baseline_xyz={np.round(baseline, 3)} "
            f"eeg_xyz={np.round(model, 3)} "
            f"overall={baseline.mean():.3f}/{model.mean():.3f} "
            f"wins={wins}/{len(test_targets)}"
        )


if __name__ == "__main__":
    main()
