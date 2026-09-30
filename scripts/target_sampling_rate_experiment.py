r"""目标输出时间网格对照：固定原500Hz EEG、21通道、63特征和alpha=100。

运行：python -X utf8 scripts/target_sampling_rate_experiment.py --data-root D:\biosignal-data\WAY-EEG-GAL

主比较：500点输出 vs 100Hz网格的100点输出，统一对原始测试轨迹前496点
（0–0.990秒，包含两端）评分。第二组增加原500Hz末点作为第101个输出，
统一对全部500点（0–0.998秒）评分。第二组不是严格均匀的100Hz采样。

这里只稀疏表示目标并线性插值，不增加运动学滤波，不是带抗混叠处理的
连续信号重采样，也不是论文完整轨迹预处理。测试标签仅参与最终评分及
独立的标签重建诊断，不参与训练、插值权重或参数选择。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subject_holdout_baseline import (
    FS, N_AFTER, PAPER_CHANNELS, RIDGE_ALPHA,
    extract_subject, make_features, evaluate,
)


TARGET_FS = 100
# 原目标500点的时间为0/500、1/500……499/500，并没有1.000秒这个点。
KNOTS_100 = np.arange(0, N_AFTER, FS // TARGET_FS)  # 0,5……495，共100点
KNOTS_ENDPOINT = np.append(KNOTS_100, N_AFTER - 1)  # 再保留0.998秒，共101点
COMMON_GRID = np.arange(KNOTS_100[-1] + 1)        # 原始前496点，到0.990秒
FULL_GRID = np.arange(N_AFTER)                  # 原始全部500点，到0.998秒


def fit_predict(x_train, y_train, x_test):
    """独立重训岭回归，允许目标为500/100/101点；标准化只使用训练者。"""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    flat = y_train.reshape(len(y_train), -1)
    target_mean = flat.mean(axis=0)
    weights = np.linalg.solve(
        train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]),
        train.T @ (flat - target_mean),
    )
    prediction = (test @ weights + target_mean).reshape(
        len(x_test), *y_train.shape[1:]
    )
    baseline = y_train.mean(axis=0)
    return baseline, prediction


def interpolate(values, knots, grid):
    """沿倒数第二维（时间）线性插值；明确禁止任何末端外推。

    可接收(时间,XYZ)基线或(试次,时间,XYZ)预测。权重只由固定时间索引
    确定，不读取测试目标。整数索引统一表示原始500Hz时间，乘1/500即秒。
    """
    knots, grid = np.asarray(knots), np.asarray(grid)
    if (knots.ndim != 1 or grid.ndim != 1 or len(knots) < 2
            or len(grid) == 0 or np.any(np.diff(knots) <= 0)
            or values.shape[-2] != len(knots)):
        raise ValueError("插值时间网格或数组形状无效")
    if grid.min() < knots[0] or grid.max() > knots[-1]:
        raise ValueError("评分网格超过输出端点：本实验禁止外推")
    right = np.clip(np.searchsorted(knots, grid, side="right"), 1, len(knots) - 1)
    left = right - 1
    fraction = ((grid - knots[left]) / (knots[right] - knots[left]))[:, None]
    return values[..., left, :] * (1 - fraction) + values[..., right, :] * fraction


def score(baseline, prediction, truth):
    """两组都对同一未滤波、未插值的原始测试标签评分。"""
    base_error = np.abs(baseline[None] - truth)
    eeg_error = np.abs(prediction - truth)
    base_xyz = base_error.mean(axis=(0, 1))
    eeg_xyz = eeg_error.mean(axis=(0, 1))
    return {
        "baseline_xyz": base_xyz.tolist(), "eeg_xyz": eeg_xyz.tolist(),
        "baseline_overall": float(base_xyz.mean()),
        "eeg_overall": float(eeg_xyz.mean()),
        "eeg_wins": int(np.sum(eeg_error.mean(axis=(1, 2)) < base_error.mean(axis=(1, 2)))),
    }


def verify():
    """检查时间点数、端点、线性插值和与原岭回归的输出一致性。"""
    assert len(KNOTS_100) == 100 and len(KNOTS_ENDPOINT) == 101
    assert len(COMMON_GRID) == 496 and KNOTS_100[-1] / FS == 0.99
    assert KNOTS_ENDPOINT[-1] / FS == 0.998
    affine = np.stack([KNOTS_ENDPOINT, 2 * KNOTS_ENDPOINT, -KNOTS_ENDPOINT], axis=-1)
    expected = np.stack([FULL_GRID, 2 * FULL_GRID, -FULL_GRID], axis=-1)
    np.testing.assert_allclose(interpolate(affine, KNOTS_ENDPOINT, FULL_GRID), expected)
    try:
        interpolate(affine[:-1], KNOTS_100, FULL_GRID)
    except ValueError:
        pass
    else:
        raise AssertionError("100个输出点不应允许对0.992–0.998秒外推")
    rng = np.random.default_rng(20260930)
    train_x, test_x = rng.normal(size=(80, 63)), rng.normal(size=(5, 63))
    train_y = rng.normal(size=(80, N_AFTER, 3))
    test_y = rng.normal(size=(5, N_AFTER, 3))
    base, dense = fit_predict(train_x, train_y, test_x)
    original_base, original_eeg, original_wins = evaluate(train_x, train_y, test_x, test_y)
    metrics = score(base, dense, test_y)
    np.testing.assert_allclose(metrics["baseline_xyz"], original_base)
    np.testing.assert_allclose(metrics["eeg_xyz"], original_eeg)
    assert metrics["eeg_wins"] == original_wins
    # 当前岭回归各输出独立：删掉其他时间的标签不会改变保留时间点的预测。
    # 检查该性质，避免把重训后应有的一致性误判成“实验没有真正运行”。
    sparse_base, sparse = fit_predict(train_x, train_y[:, KNOTS_100], test_x)
    np.testing.assert_allclose(sparse, dense[:, KNOTS_100], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sparse_base, base[KNOTS_100], rtol=1e-10, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--check-only", action="store_true", help="只运行合成数据检查")
    args = parser.parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    if len(subjects) < 2:
        parser.error("至少需要两位受试者")
    verify()
    print(f"检查通过；固定原500Hz EEG；alpha={RIDGE_ALPHA}；禁止末端外推", flush=True)
    if args.check_only:
        return
    data, counts = {}, {}
    for subject in subjects:
        eeg, target, runs = extract_subject(args.data_root, subject)
        data[subject] = make_features(eeg), target
        counts[str(subject)] = {"trials": len(target), "runs": runs}
        print(f"P{subject}: EEG={eeg.shape}；原目标={target.shape}；"
              f"100Hz网格={target[:, KNOTS_100].shape}；"
              f"保留末点={target[:, KNOTS_ENDPOINT].shape}；runs={runs}", flush=True)

    rows, reconstruction, knot_differences = [], {}, {}
    for subject in subjects:
        others = [s for s in subjects if s != subject]
        x_train = np.concatenate([data[s][0] for s in others])
        y_train = np.concatenate([data[s][1] for s in others])
        x_test, y_test = data[subject]
        dense_base, dense = fit_predict(x_train, y_train, x_test)
        sparse_base, sparse = fit_predict(x_train, y_train[:, KNOTS_100], x_test)
        end_base, end_prediction = fit_predict(x_train, y_train[:, KNOTS_ENDPOINT], x_test)
        # 每个方案确实单独拟合；保留时间点的输出理论上应与密集输出一致。
        np.testing.assert_allclose(sparse, dense[:, KNOTS_100], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(end_prediction, dense[:, KNOTS_ENDPOINT], rtol=1e-9, atol=1e-9)
        knot_differences[str(subject)] = float(np.max(np.abs(sparse - dense[:, KNOTS_100])))
        sparse_common = interpolate(sparse, KNOTS_100, COMMON_GRID)
        sparse_common_base = interpolate(sparse_base, KNOTS_100, COMMON_GRID)
        endpoint_full = interpolate(end_prediction, KNOTS_ENDPOINT, FULL_GRID)
        endpoint_full_base = interpolate(end_base, KNOTS_ENDPOINT, FULL_GRID)
        configs = (
            ("common_0_to_0.990", "original_500hz", dense_base[COMMON_GRID], dense[:, COMMON_GRID], COMMON_GRID),
            ("common_0_to_0.990", "grid_100hz", sparse_common_base, sparse_common, COMMON_GRID),
            ("full_0_to_0.998", "original_500hz", dense_base, dense, FULL_GRID),
            ("full_0_to_0.998", "grid_100hz_plus_endpoint", endpoint_full_base, endpoint_full, FULL_GRID),
        )
        print(f"\n留出P{subject}；训练={others}：", flush=True)
        for interval, config, baseline, prediction, grid in configs:
            metrics = score(baseline, prediction, y_test[:, grid])
            rows.append({"test_subject": subject, "train_subjects": others,
                         "interval": interval, "config": config, "trials": len(y_test),
                         "scored_points": len(grid), **metrics})
            print(f"  {interval} / {config}: 基线XYZ={np.round(metrics['baseline_xyz'], 3)}；"
                  f"EEG XYZ={np.round(metrics['eeg_xyz'], 3)}；"
                  f"整体={metrics['baseline_overall']:.6f}/{metrics['eeg_overall']:.6f}；"
                  f"获益={metrics['eeg_wins']}/{len(y_test)}", flush=True)
        # 标签的重建误差是表示能力诊断，不能冒充模型预测MAE；不会回馈训练。
        rebuilt_common = interpolate(y_test[:, KNOTS_100], KNOTS_100, COMMON_GRID)
        rebuilt_full = interpolate(y_test[:, KNOTS_ENDPOINT], KNOTS_ENDPOINT, FULL_GRID)
        reconstruction[str(subject)] = {
            "common_grid_xyz_mae": np.abs(rebuilt_common - y_test[:, COMMON_GRID]).mean(axis=(0, 1)).tolist(),
            "full_endpoint_xyz_mae": np.abs(rebuilt_full - y_test).mean(axis=(0, 1)).tolist(),
        }
        print(f"  标签线性重建误差XYZ（非模型成绩）："
              f"{np.round(reconstruction[str(subject)]['full_endpoint_xyz_mae'], 6)}；"
              f"重训后共同输出点最大差={knot_differences[str(subject)]:.3e}", flush=True)

    aggregates = {}
    print("\n跨被试加权平均（仅在相同区间内比较）：", flush=True)
    for interval, config in dict.fromkeys((r["interval"], r["config"]) for r in rows):
        selected = [r for r in rows if r["interval"] == interval and r["config"] == config]
        weights = [r["trials"] for r in selected]
        metrics = {key: float(np.average([r[key] for r in selected], weights=weights))
                   for key in ("baseline_overall", "eeg_overall")}
        aggregates[f"{interval}/{config}"] = metrics
        print(f"  {interval} / {config}: 基线={metrics['baseline_overall']:.6f}；"
              f"EEG={metrics['eeg_overall']:.6f}", flush=True)

    root = Path(__file__).resolve().parent.parent
    report = {
        "experiment": "target_output_time_grid_control",
        "subjects": subjects, "eeg_fs": FS, "nominal_sparse_target_fs": TARGET_FS,
        "input_points": 150, "channels": list(PAPER_CHANNELS), "feature_dimension": 63,
        "ridge_alpha": RIDGE_ALPHA, "target_filter": None,
        "interpolation": "linear; no extrapolation; no test labels used to reconstruct predictions",
        "target_knots_500hz_indices": {"grid_100hz": KNOTS_100.tolist(),
                                       "grid_100hz_plus_endpoint": KNOTS_ENDPOINT.tolist()},
        "scoring": {"common": {"points": 496, "last_time_seconds": 0.990},
                    "full": {"points": 500, "last_time_seconds": 0.998},
                    "truth": "original unsmoothed 500Hz samples"},
        "counts": counts, "results": rows, "aggregates": aggregates,
        "target_reconstruction_diagnostic": reconstruction,
        "retained_prediction_max_difference": knot_differences,
        "scope": "Output grid density only, not anti-aliased signal resampling or full paper reproduction",
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "target_sampling_rate_experiment.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 图全部使用英文标签，以避免其他计算机没有中文字体时显示方框。
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, interval, sparse_config, title in zip(
        axes, ("common_0_to_0.990", "full_0_to_0.998"),
        ("grid_100hz", "grid_100hz_plus_endpoint"),
        ("Common interval: 0-0.990 s (496 points)", "Full interval: 0-0.998 s (500 points)"),
    ):
        pos = np.arange(len(subjects))
        for offset, config, key, label in (
            (-1.5, "original_500hz", "baseline_overall", "Dense baseline"),
            (-0.5, "original_500hz", "eeg_overall", "Dense EEG"),
            (0.5, sparse_config, "baseline_overall", "Interpolated sparse baseline"),
            (1.5, sparse_config, "eeg_overall", "Interpolated sparse EEG"),
        ):
            values = [next(r for r in rows if r["interval"] == interval
                           and r["config"] == config and r["test_subject"] == s)[key] for s in subjects]
            ax.bar(pos + offset * 0.19, values, 0.19, label=label)
        ax.set_xticks(pos, [f"P{s}" for s in subjects])
        ax.set_title(title)
        ax.set_ylabel("MAE (raw position units)")
        ax.legend(fontsize=7)
    fig.suptitle("Target output-grid control; unchanged 500 Hz EEG")
    fig.tight_layout()
    figure_dir = root / "figures" / "diagnostics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "target_sampling_rate_control_mae.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"\n结果已保存：{report_path}\n图已保存：{figure_path}", flush=True)


if __name__ == "__main__":
    main()
