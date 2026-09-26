"""检查 P3/P4 留一受试者预测、逐时间点误差及 EEG 特征偏移。

运行：python scripts/diagnose_subject_holdout.py --data-root 数据目录
图像写到运行命令时所在的目录；不会修改原始数据或基线训练脚本。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from subject_holdout_baseline import (
    PAPER_CHANNELS,
    RIDGE_ALPHA,
    extract_subject,
    make_features,
)

# Windows 上显式加载中文字体；其他系统没有该字体时沿用 Matplotlib 默认字体。
font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
if font_path.is_file():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True, help="含 P1–P4 子目录的数据目录")
args = parser.parse_args()
DATA_ROOT = args.data_root
data = {}

# 读取四位被试；每人的轨迹形状应为 (294, 500, 3)。
for subject in (1, 2, 3, 4):
    eeg, y, _ = extract_subject(DATA_ROOT, subject)
    data[subject] = (make_features(eeg), y)
    endpoint_y = y[:, -1, 1]  # 每次抓握结束时的 Y 位移
    print(
        f"P{subject} Y终点："
        f"均值={endpoint_y.mean():.3f}，"
        f"标准差={endpoint_y.std():.3f}，"
        f"范围=[{endpoint_y.min():.3f}, {endpoint_y.max():.3f}]"
    )

for test_subject in (3, 4):
    train_subjects = [s for s in (1, 2, 3, 4) if s != test_subject]
    x_train = np.concatenate([data[s][0] for s in train_subjects])
    y_train = np.concatenate([data[s][1] for s in train_subjects])
    x_test, y_test = data[test_subject]

    # 完全沿用原脚本：标准化参数只由训练被试计算。
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale

    # 比较训练者与留出被试的 63 个 EEG 特征。每个特征对应
    # 一个运动前 0.1 秒时间窗和一个 EEG 通道。
    train_median = np.median(train, axis=0)
    test_median = np.median(test, axis=0)
    median_shift = test_median - train_median
    train_distance = np.sqrt(np.mean(train ** 2, axis=1))
    test_distance = np.sqrt(np.mean(test ** 2, axis=1))
    train_p95 = np.percentile(train_distance, 95)
    print(
        f"P{test_subject} 特征距离：训练者中位数={np.median(train_distance):.2f}，"
        f"测试者中位数={np.median(test_distance):.2f}；"
        f"测试试次超过训练者第95百分位的比例="
        f"{np.mean(test_distance > train_p95):.1%}"
    )
    for feature in np.argsort(np.abs(median_shift))[-8:][::-1]:
        window = feature // len(PAPER_CHANNELS) + 1
        channel = PAPER_CHANNELS[feature % len(PAPER_CHANNELS)]
        print(
            f"  偏移特征：运动前第{window}个0.1秒窗，{channel}，"
            f"中位数差={median_shift[feature]:+.2f}个训练标准差"
        )

    fig, axes_shift = plt.subplots(1, 2, figsize=(15, 4.5))
    heatmap = axes_shift[0].imshow(
        median_shift.reshape(3, len(PAPER_CHANNELS)),
        aspect="auto", cmap="coolwarm", vmin=-3, vmax=3,
    )
    axes_shift[0].set_xticks(range(len(PAPER_CHANNELS)), PAPER_CHANNELS, rotation=90)
    axes_shift[0].set_yticks(range(3), ["-0.3至-0.2秒", "-0.2至-0.1秒", "-0.1至0秒"])
    axes_shift[0].set_title("测试者与训练者的特征中位数差")
    fig.colorbar(heatmap, ax=axes_shift[0], label="训练标准差单位")
    axes_shift[1].hist(train_distance, bins=30, density=True, alpha=0.6, label="训练者")
    axes_shift[1].hist(test_distance, bins=30, density=True, alpha=0.6, label=f"P{test_subject}")
    axes_shift[1].axvline(train_p95, color="black", linestyle="--", label="训练者第95百分位")
    axes_shift[1].set_xlabel("每次抓握的标准化特征距离")
    axes_shift[1].set_ylabel("密度")
    axes_shift[1].set_title("EEG特征整体分布")
    axes_shift[1].legend()
    fig.suptitle(f"P{test_subject} EEG输入特征偏移检查")
    fig.tight_layout()
    shift_output = Path(f"diagnostics_P{test_subject}_feature_shift.png")
    fig.savefig(shift_output, dpi=160)
    plt.close(fig)
    print(f"已保存：{shift_output.resolve()}")

    flat = y_train.reshape(len(y_train), -1)
    target_mean = flat.mean(axis=0)
    weights = np.linalg.solve(
        train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]),
        train.T @ (flat - target_mean),
    )
    pred = (test @ weights + target_mean).reshape(y_test.shape)
    baseline = np.broadcast_to(y_train.mean(axis=0), y_test.shape)

    # 找出“EEG 相对基线退步最多”的两次抓握，方便优先排查。
    base_y_error = np.abs(baseline[:, :, 1] - y_test[:, :, 1]).mean(axis=1)
    eeg_y_error = np.abs(pred[:, :, 1] - y_test[:, :, 1]).mean(axis=1)
    worst = np.argsort(eeg_y_error - base_y_error)[-2:][::-1]

    print(f"\nP{test_subject}：Y方向退步最多的试次（从1开始）：{worst + 1}")
    for i in worst:
        print(
            f"第{i + 1}次：基线Y MAE={base_y_error[i]:.3f}，"
            f"EEG Y MAE={eeg_y_error[i]:.3f}"
        )

    # 每张图显示最差的一次抓握的 X/Y/Z 三个方向。
    i = worst[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    time = np.arange(y_test.shape[1]) / 500
    for axis, name in enumerate(("X", "Y", "Z")):
        axes[axis].plot(time, y_test[i, :, axis], label="真实")
        axes[axis].plot(time, baseline[i, :, axis], label="平均轨迹基线")
        axes[axis].plot(time, pred[i, :, axis], label="EEG预测")
        axes[axis].set_ylabel(f"{name} 位移")
        axes[axis].legend()
    axes[-1].set_xlabel("运动开始后的时间（秒）")
    fig.suptitle(f"P{test_subject} 第{i + 1}次抓握")
    fig.tight_layout()
    output = Path(f"diagnostics_P{test_subject}_worstY.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"已保存：{output.resolve()}")

    # 汇总全部 294 次，而不是只依据误差最大的单次抓握判断。
    # 第一个轴按试次求平均，留下 500 个时间点和 X/Y/Z 三个方向。
    true_mean = y_test.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    baseline_mean = baseline[0]
    baseline_mae_t = np.abs(baseline - y_test).mean(axis=0)
    eeg_mae_t = np.abs(pred - y_test).mean(axis=0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, name in enumerate(("X", "Y", "Z")):
        axes[axis, 0].plot(time, true_mean[:, axis], label="真实均值")
        axes[axis, 0].plot(time, baseline_mean[:, axis], label="平均轨迹基线")
        axes[axis, 0].plot(time, pred_mean[:, axis], label="EEG预测均值")
        axes[axis, 0].set_ylabel(f"{name} 位移")
        axes[axis, 0].legend()

        axes[axis, 1].plot(time, baseline_mae_t[:, axis], label="基线 MAE")
        axes[axis, 1].plot(time, eeg_mae_t[:, axis], label="EEG MAE")
        axes[axis, 1].set_ylabel(f"{name} MAE")
        axes[axis, 1].legend()

        # 单方向逐试次比较：只有 EEG 的该方向 MAE 更低才算获益。
        trial_base = np.abs(baseline[:, :, axis] - y_test[:, :, axis]).mean(axis=1)
        trial_eeg = np.abs(pred[:, :, axis] - y_test[:, :, axis]).mean(axis=1)
        wins = np.sum(trial_eeg < trial_base)
        print(
            f"P{test_subject} {name}：基线MAE={trial_base.mean():.3f}，"
            f"EEG MAE={trial_eeg.mean():.3f}，"
            f"EEG获益={wins}/{len(y_test)}次"
        )

    axes[0, 0].set_title("全部试次的平均轨迹")
    axes[0, 1].set_title("全部试次在每个时间点的平均绝对误差")
    axes[-1, 0].set_xlabel("运动开始后的时间（秒）")
    axes[-1, 1].set_xlabel("运动开始后的时间（秒）")
    fig.suptitle(f"P{test_subject} 全部 {len(y_test)} 次抓握")
    fig.tight_layout()
    output = Path(f"diagnostics_P{test_subject}_all_trials.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"已保存：{output.resolve()}")
