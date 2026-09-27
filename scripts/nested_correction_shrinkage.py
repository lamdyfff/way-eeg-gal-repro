r"""只用训练被试内层验证选择 EEG 修正强度，再做四轮留一被试测试。

运行：python -X utf8 scripts/nested_correction_shrinkage.py --data-root D:\biosignal-data\WAY-EEG-GAL

预测 = 训练者平均轨迹 + strength × (原 EEG 岭回归预测 - 训练者平均轨迹)。
strength=0 是平均轨迹基线，1 是原 EEG 模型。本实验沿用 21 通道、
63 个特征、固定岭回归 alpha=100；仅探索一个全方向共用的修正强度。
四位被试此前已用于诊断，结果只能作为探索性分析，不能视为全新测试集。
"""

import argparse
from pathlib import Path

import numpy as np

from subject_holdout_baseline import RIDGE_ALPHA, extract_subject, make_features


SUBJECTS = (1, 2, 3, 4)
STRENGTHS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])


def fit_predict(train_x, train_y, test_x):
    """复制原基线的训练集标准化与岭回归公式，不触碰验证/测试目标。"""
    feature_mean = train_x.mean(axis=0)
    feature_scale = train_x.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    x = (train_x - feature_mean) / feature_scale
    test = (test_x - feature_mean) / feature_scale
    flat = train_y.reshape(len(train_y), -1)
    target_mean = flat.mean(axis=0)
    weights = np.linalg.solve(
        x.T @ x + RIDGE_ALPHA * np.eye(x.shape[1]),
        x.T @ (flat - target_mean),
    )
    baseline = train_y.mean(axis=0)
    eeg_prediction = (test @ weights + target_mean).reshape(
        len(test_x), *train_y.shape[1:]
    )
    return baseline, eeg_prediction


def gather(data, subjects):
    return (
        np.concatenate([data[subject][0] for subject in subjects]),
        np.concatenate([data[subject][1] for subject in subjects]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    outer_results = {label: [] for label in ("基线", "原EEG", "内层选择")}
    for subject in SUBJECTS:
        eeg, y, _ = extract_subject(args.data_root, subject)
        data[subject] = make_features(eeg), y

    for outer_test in SUBJECTS:
        train_subjects = [s for s in SUBJECTS if s != outer_test]
        # 内层每次留出训练者之一，只用剩余两人训练；绝不读取外层
        # 测试者的真实轨迹来选 strength。
        error_sums = np.zeros(len(STRENGTHS))
        value_count = 0
        for inner_valid in train_subjects:
            inner_train = [s for s in train_subjects if s != inner_valid]
            x_train, y_train = gather(data, inner_train)
            x_valid, y_valid = data[inner_valid]
            baseline, eeg_prediction = fit_predict(x_train, y_train, x_valid)
            correction = eeg_prediction - baseline[None, :, :]
            for i, strength in enumerate(STRENGTHS):
                prediction = baseline[None, :, :] + strength * correction
                error_sums[i] += np.abs(prediction - y_valid).sum()
            value_count += y_valid.size
        inner_mae = error_sums / value_count
        # np.argmin 在完全并列时选更小的 strength（预定义数组升序）。
        chosen = STRENGTHS[np.argmin(inner_mae)]

        x_train, y_train = gather(data, train_subjects)
        x_test, y_test = data[outer_test]
        baseline, eeg_prediction = fit_predict(x_train, y_train, x_test)
        candidates = {
            "基线": np.broadcast_to(baseline, y_test.shape),
            "原EEG": eeg_prediction,
            "内层选择": baseline[None, :, :] + chosen * (eeg_prediction - baseline[None, :, :]),
        }
        print(f"\n留出 P{outer_test}；内层验证者={train_subjects}")
        print(
            "内层 MAE（强度:MAE）："
            + ", ".join(f"{s:g}:{m:.3f}" for s, m in zip(STRENGTHS, inner_mae))
            + f"；所选强度={chosen:g}"
        )
        for label, prediction in candidates.items():
            xyz = np.abs(prediction - y_test).mean(axis=(0, 1))
            outer_results[label].append(xyz.mean())
            print(f"  {label}: XYZ={np.round(xyz, 3)}；整体={xyz.mean():.3f}")

    print("\n四轮平均整体 MAE（每位测试者均为 294 次抓握）：")
    for label, values in outer_results.items():
        print(f"  {label}: {np.mean(values):.3f}")


if __name__ == "__main__":
    main()
