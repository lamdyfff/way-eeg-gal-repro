r"""EEG 输入 500→100 Hz 对照；全部模型对同一原始500点轨迹评分。

运行：python -X utf8 scripts/sampling_rate_experiment.py --data-root D:\biosignal-data\WAY-EEG-GAL

三种输入：原500Hz、抗混叠后500Hz、同一抗混叠信号每5点取1点得到100Hz。
三个0.1秒特征窗、21通道、63个特征、岭回归alpha和留一被试划分一致。
本轮只检查输入采样率，不包含论文的运动学降采样/归一化和完整模型。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, ellip, ellipord, sosfilt, sosfreqz

from subject_holdout_baseline import (
    FS, N_BEFORE, N_AFTER, PAPER_CHANNELS, WRIST_AXES, RIDGE_ALPHA,
    read_mat, make_features, evaluate,
)


OUTPUT_FS = 100
FACTOR = FS // OUTPUT_FS
CONFIGS = ("original_500hz", "antialias_500hz", "antialias_100hz")
LABELS = ("Original 500 Hz", "AA + 500 Hz", "AA + 100 Hz")
BANDPASS = butter(4, [0.1, 40], btype="bandpass", fs=FS, output="sos")
# 新100Hz信号的奈奎斯特频率为50Hz，抽取前需压低50Hz及以上的成分。
# 参数预先固定：通带到40Hz，通带纹波0.1dB，50Hz起阻带衰减至少60dB。
AA_ORDER, AA_CUTOFF = ellipord(40, 50, 0.1, 60, fs=FS)
ANTIALIAS = ellip(AA_ORDER, 0.1, 60, AA_CUTOFF, fs=FS, output="sos")


def preprocess(signal):
    """在连续EEG上参考及因果滤波，避免每个短窗口单独滤波的边缘效应。"""
    referenced = signal.astype(float) - signal.mean(axis=1, keepdims=True)
    original = sosfilt(BANDPASS, referenced, axis=0)
    antialias = sosfilt(ANTIALIAS, original, axis=0)
    # 使用sosfilt，只用当前及过去采样。新增滤波器有频率相关延迟，
    # 本轮不以未来采样补偿；AA+500Hz对照用于观察它自身的影响。
    return original, antialias


def features_at_rate(eeg, sampling_rate):
    """每个0.1秒窗分别用50点或10点，输出均为63个log标准差特征。"""
    points_per_window = sampling_rate // 10
    expected = (3 * points_per_window, len(PAPER_CHANNELS))
    if eeg.shape[1:] != expected:
        raise ValueError(f"{sampling_rate}Hz输入形状异常：{eeg.shape}")
    return np.log(
        np.std(eeg.reshape(len(eeg), 3, points_per_window, len(PAPER_CHANNELS)), axis=2)
        + 1e-8
    ).reshape(len(eeg), -1)


def verify_signal_handling():
    """检查特征一致性、因果性以及滤波器在50Hz后的抗混叠衰减。"""
    rng = np.random.default_rng(20260930)
    sample = rng.normal(size=(3, N_BEFORE, len(PAPER_CHANNELS)))
    np.testing.assert_allclose(features_at_rate(sample, FS), make_features(sample))
    assert features_at_rate(sample[:, ::FACTOR], OUTPUT_FS).shape == (3, 63)
    # 大幅改变运动开始之后的数据，之前的两种滤波输出必须保持一致。
    signal = rng.normal(size=(900, len(PAPER_CHANNELS)))
    changed = signal.copy()
    changed[600:] += rng.normal(0, 100, size=changed[600:].shape)
    for before, after in zip(preprocess(signal), preprocess(changed)):
        np.testing.assert_array_equal(before[:600], after[:600])
    frequency, response = sosfreqz(ANTIALIAS, worN=16384, fs=FS)
    gain = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
    if np.max(gain[frequency >= 50]) > -59.99:
        raise AssertionError("抗混叠阻带未达到预定60dB衰减")


def extract_variants(data_root, subject):
    """一次读取每段，生成三种输入及共同目标，并沿用原基线试次顺序。"""
    directory = data_root / f"P{subject}"
    events = read_mat(directory / f"P{subject}_AllLifts.mat", "P")
    columns = list(events.ColNames)
    run_col, lift_col, start_col, onset_col = (
        columns.index(name) for name in ("Run", "Lift", "StartTime", "tHandStart")
    )
    trials = {config: [] for config in CONFIGS}
    targets, counts = [], []
    for run in range(1, 10):
        hs = read_mat(directory / f"HS_P{subject}_S{run}.mat", "hs")
        ws = read_mat(directory / f"WS_P{subject}_S{run}.mat", "ws")
        eeg_names, kin_names = list(hs.eeg.names), list(hs.kin.names)
        eeg_cols = [eeg_names.index(name) for name in PAPER_CHANNELS]
        wrist_cols = [
            next(i for i, name in enumerate(kin_names) if str(name).startswith(axis + " "))
            for axis in WRIST_AXES
        ]
        c3 = eeg_names.index("C3")
        original, antialias = preprocess(hs.eeg.sig[:, eeg_cols])
        rows = events.AllLifts[events.AllLifts[:, run_col] == run]
        if len(rows) != len(ws.win):
            raise ValueError(f"P{subject} S{run}: 事件/窗口数量不同")
        counts.append(len(rows))
        for row in rows:
            lift, onset = int(row[lift_col]), float(row[onset_col])
            window = ws.win[lift - 1]
            if not np.isfinite(onset):
                raise ValueError(f"P{subject} S{run} lift{lift}: 无运动起始时间")
            start = round((float(row[start_col]) - 2.0) * FS)
            np.testing.assert_array_equal(
                hs.eeg.sig[start:start + 10, c3], window.eeg[:10, c3]
            )
            onset_window = int(np.searchsorted(window.eeg_t, onset))
            onset_run = start + onset_window
            if onset_run < N_BEFORE or onset_window + N_AFTER > len(window.kin):
                raise ValueError(f"P{subject} S{run} lift{lift}: 输入/目标越界")
            x_original = original[onset_run - N_BEFORE:onset_run]
            x_aa = antialias[onset_run - N_BEFORE:onset_run]
            # 按运动起点对齐抽取，100Hz点对应-0.30、-0.29…-0.01秒。
            # 最后一点仍在运动开始之前，不包含0秒或运动后的EEG。
            x_100 = x_aa[::FACTOR]
            y = window.kin[onset_window:onset_window + N_AFTER, wrist_cols].astype(float)
            y = y - y[0]
            values = (x_original, x_aa, x_100)
            if any(not np.isfinite(value).all() for value in (*values, y)):
                raise ValueError(f"P{subject} S{run} lift{lift}: 非有限数值")
            if x_original.shape != (N_BEFORE, 21) or y.shape != (N_AFTER, 3):
                raise ValueError(f"P{subject} S{run} lift{lift}: 试次形状异常")
            for config, value in zip(CONFIGS, values):
                trials[config].append(value)
            targets.append(y)
    features = {}
    for config in CONFIGS:
        array = np.stack(trials[config])
        rate = OUTPUT_FS if config == CONFIGS[-1] else FS
        features[config] = features_at_rate(array, rate)
    return features, np.stack(targets), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    if len(subjects) < 2:
        parser.error("至少需要两位受试者")
    verify_signal_handling()
    print(f"检查通过；抗混叠滤波器阶数={AA_ORDER}；预先固定alpha={RIDGE_ALPHA}", flush=True)
    data, counts = {}, {}
    for subject in subjects:
        features, targets, runs = extract_variants(args.data_root, subject)
        data[subject] = features, targets
        counts[str(subject)] = {"trials": len(targets), "runs": runs}
        print(
            f"P{subject}: EEG500=({len(targets)},150,21)，"
            f"EEG100=({len(targets)},30,21)，共同目标={targets.shape}，runs={runs}",
            flush=True,
        )

    rows = []
    for test_subject in subjects:
        others = [s for s in subjects if s != test_subject]
        y_train = np.concatenate([data[s][1] for s in others])
        y_test = data[test_subject][1]
        baseline_reference = None
        print(f"\n留出P{test_subject}；训练={others}；共同原始500Hz目标评分：", flush=True)
        for config in CONFIGS:
            x_train = np.concatenate([data[s][0][config] for s in others])
            x_test = data[test_subject][0][config]
            base, eeg, wins = evaluate(x_train, y_train, x_test, y_test)
            if baseline_reference is None:
                baseline_reference = base
            else:
                np.testing.assert_array_equal(base, baseline_reference)
            rows.append({
                "test_subject": test_subject, "train_subjects": others,
                "config": config, "trials": len(y_test),
                "baseline_xyz": base.tolist(), "eeg_xyz": eeg.tolist(),
                "baseline_overall": float(base.mean()), "eeg_overall": float(eeg.mean()),
                "eeg_wins": wins,
            })
            print(f"  {config}: 基线XYZ={np.round(base, 3)}；EEG XYZ={np.round(eeg, 3)}；"
                  f"整体={base.mean():.3f}/{eeg.mean():.3f}；获益={wins}/{len(y_test)}", flush=True)

    aggregates = {}
    print("\n跨被试加权平均：", flush=True)
    for config in CONFIGS:
        selected = [row for row in rows if row["config"] == config]
        weights = [row["trials"] for row in selected]
        mean_base = float(np.average([row["baseline_overall"] for row in selected], weights=weights))
        mean_eeg = float(np.average([row["eeg_overall"] for row in selected], weights=weights))
        aggregates[config] = {"baseline_overall": mean_base, "eeg_overall": mean_eeg}
        print(f"  {config}: 基线={mean_base:.3f}；EEG={mean_eeg:.3f}", flush=True)

    root = Path(__file__).resolve().parent.parent
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EEG_input_sampling_rate_control", "subjects": subjects,
        "source_fs": FS, "downsampled_input_fs": OUTPUT_FS, "target_fs": FS,
        "input_duration_seconds": 0.3, "target_duration_seconds": 1.0,
        "features": {"channels": 21, "windows": 3, "window_seconds": 0.1,
                     "dimension": 63, "statistic": "log standard deviation"},
        "ridge_alpha": RIDGE_ALPHA,
        "antialias": {"type": "causal elliptic", "order": int(AA_ORDER),
                      "passband_hz": 40, "stopband_hz": 50, "ripple_db": 0.1,
                      "attenuation_db": 60, "delay_compensation": False},
        "counts": counts, "results": rows, "aggregates": aggregates,
    }
    output = report_dir / "sampling_rate_experiment.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    position = np.arange(len(subjects))
    width = 0.19
    base_values = [next(row for row in rows if row["test_subject"] == s)["baseline_overall"]
                   for s in subjects]
    axes[0].bar(position - 1.5 * width, base_values, width, label="Mean trajectory")
    for i, (config, label) in enumerate(zip(CONFIGS, LABELS)):
        values = [next(row for row in rows if row["test_subject"] == s and row["config"] == config)
                  ["eeg_overall"] for s in subjects]
        axes[0].bar(position + (i - 0.5) * width, values, width, label=label)
    axes[0].set_xticks(position, [f"P{s}" for s in subjects])
    axes[0].set_ylabel("Overall MAE (raw position units)")
    axes[0].legend(fontsize=8)
    for axis, name in enumerate(("X", "Y", "Z")):
        difference = []
        for subject in subjects:
            selected = {row["config"]: row for row in rows if row["test_subject"] == subject}
            difference.append(selected[CONFIGS[2]]["eeg_xyz"][axis]
                              - selected[CONFIGS[1]]["eeg_xyz"][axis])
        axes[1].bar(position + (axis - 1) * 0.24, difference, 0.24, label=name)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(position, [f"P{s}" for s in subjects])
    axes[1].set_ylabel("AA+100Hz MAE minus AA+500Hz MAE")
    axes[1].set_title("Positive means downsampling is worse")
    axes[1].legend()
    fig.suptitle("Input sampling-rate control; same original 500 Hz targets")
    fig.tight_layout()
    figure_dir = root / "figures" / "diagnostics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure = figure_dir / "sampling_rate_control_mae.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    print(f"结果已保存：{output}\n图已保存：{figure}", flush=True)


if __name__ == "__main__":
    main()
