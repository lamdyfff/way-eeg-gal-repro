r"""比较四种 EEG 输入通道配置，全部使用相同的留一受试者划分。

运行示例：
python scripts/channel_ablation.py --data-root D:\biosignal-data\WAY-EEG-GAL

这是探索性对照实验：F4/FC5 候选来自已看过的 P3 测试结果，
因此不能把其中最好的配置称为无偏的最终泛化成绩。
"""

import argparse
from pathlib import Path

import numpy as np

from subject_holdout_baseline import PAPER_CHANNELS, evaluate, extract_subject, make_features


SUBJECTS = (1, 2, 3, 4)
# 空元组表示保留全部 21 通道；其余配置仅从模型输入特征中去掉指定通道。
CONFIGS = (
    ("全部21通道", ()),
    ("去F4", ("F4",)),
    ("去FC5", ("FC5",)),
    ("去F4和FC5", ("F4", "FC5")),
)


def feature_mask(dropped):
    """返回 63 维特征的布尔掩码；每通道在三个 0.1 秒窗中各占一个特征。"""
    keep_channel = np.array([name not in dropped for name in PAPER_CHANNELS])
    return np.tile(keep_channel, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    # 数据只读一次，所有配置使用完全相同的试次、滤波、平均重参考和目标。
    # 因此这里的“去通道”只改模型输入，不重新计算 EEG 参考或滤波。
    data = {}
    for subject in SUBJECTS:
        eeg, targets, runs = extract_subject(args.data_root, subject)
        data[subject] = (make_features(eeg), targets)
        print(f"P{subject}: {len(eeg)} 次抓握，九段试次数={runs}")

    summary = {name: [] for name, _ in CONFIGS}
    for test_subject in SUBJECTS:
        train_subjects = [s for s in SUBJECTS if s != test_subject]
        x_train = np.concatenate([data[s][0] for s in train_subjects])
        y_train = np.concatenate([data[s][1] for s in train_subjects])
        x_test, y_test = data[test_subject]

        print(f"\n留出 P{test_subject}；训练 {train_subjects}：")
        for name, dropped in CONFIGS:
            mask = feature_mask(dropped)
            # evaluate 会在每种配置下重新拟合训练集标准化和岭回归权重；
            # 测试受试者始终不参与这些参数的估计。
            baseline_xyz, eeg_xyz, wins = evaluate(
                x_train[:, mask], y_train, x_test[:, mask], y_test
            )
            baseline_mae = baseline_xyz.mean()
            eeg_mae = eeg_xyz.mean()
            summary[name].append((baseline_mae, eeg_mae, wins, len(y_test)))
            print(
                f"  {name}: 基线XYZ={np.round(baseline_xyz, 3)}；"
                f"EEG XYZ={np.round(eeg_xyz, 3)}；"
                f"整体={baseline_mae:.3f}/{eeg_mae:.3f}；"
                f"EEG获益={wins}/{len(y_test)}"
            )

    # 四人各 294 次；这里仍明确按试次数加权，便于以后扩展到不同试次数。
    print("\n四位受试者汇总（整体 MAE；前者基线、后者 EEG）：")
    for name, rows in summary.items():
        total = sum(row[3] for row in rows)
        baseline = sum(row[0] * row[3] for row in rows) / total
        eeg = sum(row[1] * row[3] for row in rows) / total
        wins = sum(row[2] for row in rows)
        print(f"  {name}: {baseline:.3f}/{eeg:.3f}；EEG获益={wins}/{total}")


if __name__ == "__main__":
    main()
