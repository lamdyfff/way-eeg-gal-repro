r"""按记录段检查 P3 的 Y 和 P4 的 Z 误差；只读，不改模型或原始数据。

运行：python -X utf8 scripts/diagnose_runwise_errors.py --data-root D:\biosignal-data\WAY-EEG-GAL

与 subject_holdout_baseline.py 相同：全部 21 EEG 通道、固定岭回归参数、
留出完整测试被试，特征标准化仅由训练被试拟合。测试数据仅用于诊断。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subject_holdout_baseline import RIDGE_ALPHA, extract_subject, make_features


SUBJECTS = (1, 2, 3, 4)
TARGETS = ((3, 1, "Y"), (4, 2, "Z"))


def fit_and_predict(train_x, train_y, test_x):
    """保持原基线的训练过程不变，返回平均轨迹与 EEG 预测。"""
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = (train_x - mean) / scale
    test = (test_x - mean) / scale
    flat = train_y.reshape(len(train_y), -1)
    target_mean = flat.mean(axis=0)
    weights = np.linalg.solve(
        x.T @ x + RIDGE_ALPHA * np.eye(x.shape[1]),
        x.T @ (flat - target_mean),
    )
    baseline = train_y.mean(axis=0)
    prediction = (test @ weights + target_mean).reshape(len(test_x), *train_y.shape[1:])
    return baseline, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    for subject in SUBJECTS:
        eeg, y, run_counts = extract_subject(args.data_root, subject)
        data[subject] = (make_features(eeg), y, run_counts)

    output_dir = Path(__file__).resolve().parent.parent / "figures" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    for subject, axis, name in TARGETS:
        train_subjects = [s for s in SUBJECTS if s != subject]
        train_x = np.concatenate([data[s][0] for s in train_subjects])
        train_y = np.concatenate([data[s][1] for s in train_subjects])
        test_x, test_y, run_counts = data[subject]
        baseline, prediction = fit_and_predict(train_x, train_y, test_x)
        # 每次抓握单独计算 1 秒、单轴的 MAE；正的 delta 代表 EEG 更差。
        baseline_error = np.abs(test_y[:, :, axis] - baseline[:, axis]).mean(axis=1)
        eeg_error = np.abs(test_y[:, :, axis] - prediction[:, :, axis]).mean(axis=1)
        delta = eeg_error - baseline_error
        endpoints = test_y[:, -1, axis]
        if sum(run_counts) != len(test_y):
            raise ValueError(f"P{subject}: 九段试次数之和与测试试次数不符")
        print(f"\nP{subject} {name}: 基线={baseline_error.mean():.3f}, "
              f"EEG={eeg_error.mean():.3f}, 差值={delta.mean():+.3f}")
        print("段  试次  终点均值  终点标准差  基线MAE  EEG MAE  差值  EEG获益")
        start = 0
        run_delta = []
        run_endpoints = []
        for run, count in enumerate(run_counts, start=1):
            stop = start + count
            selected = slice(start, stop)
            run_delta.append(delta[selected].mean())
            run_endpoints.append(endpoints[selected])
            wins = int(np.sum(delta[selected] < 0))
            print(
                f"S{run:<2} {count:>4}  {endpoints[selected].mean():>8.3f}  "
                f"{endpoints[selected].std():>10.3f}  "
                f"{baseline_error[selected].mean():>7.3f}  "
                f"{eeg_error[selected].mean():>7.3f}  "
                f"{delta[selected].mean():>+6.3f}  {wins}/{count}"
            )
            start = stop

        # 同时给出原试次号及所属段，避免把第 272 次误认为 S272。
        print("相对基线退步最大的 5 次（全被试试次号从 1 开始）：")
        cumulative = np.cumsum(run_counts)
        for index in np.argsort(delta)[-5:][::-1]:
            run = int(np.searchsorted(cumulative, index, side="right")) + 1
            prev = 0 if run == 1 else cumulative[run - 2]
            print(
                f"  第{index + 1}次 / S{run}第{index - prev + 1}次："
                f"终点={endpoints[index]:.3f}，基线={baseline_error[index]:.3f}，"
                f"EEG={eeg_error[index]:.3f}，差值={delta[index]:+.3f}"
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(np.arange(1, 10), run_delta,
                    color=["tab:orange" if value > 0 else "tab:green" for value in run_delta])
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Run")
        axes[0].set_ylabel("EEG MAE - baseline MAE")
        axes[0].set_title("Positive means EEG is worse")
        axes[0].set_xticks(range(1, 10))
        axes[1].boxplot(run_endpoints, tick_labels=[str(run) for run in range(1, 10)])
        axes[1].set_xlabel("Run")
        axes[1].set_ylabel(f"{name} displacement at 1 s (raw units)")
        axes[1].set_title("Target endpoint distribution")
        fig.suptitle(f"P{subject} {name}: run-wise held-out diagnosis")
        fig.tight_layout()
        output = output_dir / f"diagnostics_P{subject}_{name}_runwise.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print(f"图已保存：{output}")

        # EEG 模型 = 训练者平均轨迹 + EEG 修正。真实轨迹与平均轨迹之差，
        # 则是该测试者实际“需要”的修正。这里只观察，不使用测试者拟合参数。
        needed = test_y[:, :, axis] - baseline[:, axis]
        applied = prediction[:, :, axis] - baseline[:, axis]
        print("\n相对平均轨迹的修正诊断（数值仍为原始位置单位）：")
        print(
            f"  1秒终点：训练者平均轨迹={baseline[-1, axis]:+.3f}，"
            f"测试者真实均值={endpoints.mean():+.3f}，"
            f"真实所需修正={needed[:, -1].mean():+.3f}，"
            f"EEG实际修正={applied[:, -1].mean():+.3f}"
        )
        print(
            f"  全1秒平均：真实所需修正={needed.mean():+.3f}，"
            f"EEG实际修正={applied.mean():+.3f}"
        )
        print("段   终点真实所需修正   终点EEG实际修正")
        start = 0
        for run, count in enumerate(run_counts, start=1):
            stop = start + count
            print(
                f"S{run:<2} {needed[start:stop, -1].mean():>+15.3f}"
                f" {applied[start:stop, -1].mean():>+19.3f}"
            )
            start = stop

        # 去掉每段、每时间点的均值后，再看同段内逐试次变化是否对得上；
        # 否则共同的时间趋势会让相关系数看起来虚高。
        needed_within = needed.copy()
        applied_within = applied.copy()
        start = 0
        for count in run_counts:
            stop = start + count
            needed_within[start:stop] -= needed[start:stop].mean(axis=0)
            applied_within[start:stop] -= applied[start:stop].mean(axis=0)
            start = stop
        within_corr = np.corrcoef(needed_within.ravel(), applied_within.ravel())[0, 1]
        print(f"  去除段内平均轨迹后的试次变化相关系数={within_corr:+.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        time = np.arange(test_y.shape[1]) / 500
        axes[0].plot(time, needed.mean(axis=0), label="True needed correction")
        axes[0].plot(time, applied.mean(axis=0), label="EEG applied correction")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Seconds after movement onset")
        axes[0].set_ylabel(f"{name} correction (raw units)")
        axes[0].legend()
        axes[1].scatter(needed[:, -1], applied[:, -1], s=12, alpha=0.65)
        axes[1].set_xlabel("True needed correction at 1 s")
        axes[1].set_ylabel("EEG applied correction at 1 s")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].axvline(0, color="black", linewidth=0.8)
        fig.suptitle(f"P{subject} {name}: correction relative to training baseline")
        fig.tight_layout()
        output = output_dir / f"diagnostics_P{subject}_{name}_correction.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print(f"图已保存：{output}")


if __name__ == "__main__":
    main()
