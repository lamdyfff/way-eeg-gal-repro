r"""固定训练P1–P4，测试新被试P5；不执行五人留一被试或测试集调参。

运行：python -X utf8 scripts/fixed_subject_test.py --data-root D:\biosignal-data\WAY-EEG-GAL

沿用原500Hz EEG、21通道、3个0.1秒窗的63维log标准差特征、alpha=100，
预测运动后原始500点XYZ位移。先记录配置，再训练，最后读取测试数据。
首次测试结果与后续同配置复跑需要区分；单个新被试不是群体泛化证明。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_kinematics_alignment import audit_subject
from subject_holdout_baseline import (
    FS, N_BEFORE, N_AFTER, PAPER_CHANNELS, RIDGE_ALPHA,
    extract_subject, make_features, evaluate,
)


def required_files(data_root, subject):
    """九段HS/WS和事件文件共19个；ST文件不使用。"""
    directory = data_root / f"P{subject}"
    names = [f"P{subject}_AllLifts.mat"]
    for run in range(1, 10):
        names.extend([f"HS_P{subject}_S{run}.mat", f"WS_P{subject}_S{run}.mat"])
    return [directory / name for name in names]


def fit_model(features, targets):
    """仅接受训练数据：测试者不参与标准化、平均轨迹、截距或权重拟合。"""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (features - mean) / scale
    flat = targets.reshape(len(targets), -1)
    target_mean = flat.mean(axis=0)
    weights = np.linalg.solve(
        train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]),
        train.T @ (flat - target_mean),
    )
    return {"mean": mean, "scale": scale, "weights": weights,
            "target_mean": target_mean, "baseline": targets.mean(axis=0)}


def predict(model, features):
    """预测函数不接收测试轨迹；标签只能在预测完成后用于评分。"""
    test = (features - model["mean"]) / model["scale"]
    return (test @ model["weights"] + model["target_mean"]).reshape(len(test), N_AFTER, 3)


def verify_model():
    """用合成数据检查新脚本与原基线的训练、评分完全一致。"""
    rng = np.random.default_rng(20260930)
    train_x, test_x = rng.normal(size=(80, 63)), rng.normal(size=(7, 63))
    train_y, test_y = rng.normal(size=(80, 500, 3)), rng.normal(size=(7, 500, 3))
    model = fit_model(train_x, train_y)
    prediction = predict(model, test_x)
    base_error = np.abs(model["baseline"][None] - test_y)
    eeg_error = np.abs(prediction - test_y)
    reference_base, reference_eeg, reference_wins = evaluate(train_x, train_y, test_x, test_y)
    np.testing.assert_allclose(base_error.mean(axis=(0, 1)), reference_base, rtol=1e-12)
    np.testing.assert_allclose(eeg_error.mean(axis=(0, 1)), reference_eeg, rtol=1e-12)
    assert int(np.sum(eeg_error.mean(axis=(1, 2)) < base_error.mean(axis=(1, 2)))) == reference_wins
    assert np.isfinite(prediction).all()


def save_json(path, value):
    """生成报告，不保存原始EEG或手腕时间序列。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--test-subject", type=int, default=5)
    parser.add_argument("--check-only", action="store_true", help="只跑合成数据检查，不读取原始数据")
    args = parser.parse_args()
    train_subjects = list(dict.fromkeys(args.train_subjects))
    if not train_subjects or args.test_subject in train_subjects:
        parser.error("训练者必须非空，且不能包含测试者")
    verify_model()
    print("合成检查通过：新脚本与原基线一致；训练/测试被试严格分离", flush=True)
    if args.check_only:
        return

    # 这里只检查文件存在性，不读取P5的标签或根据其表现作决定。
    subjects = train_subjects + [args.test_subject]
    inventory = {}
    for subject in subjects:
        paths = required_files(args.data_root, subject)
        missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError("缺少或为空的文件：\n" + "\n".join(missing))
        inventory[f"P{subject}"] = [{"name": p.name, "bytes": p.stat().st_size} for p in paths]
    root = Path(__file__).resolve().parent.parent
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fixed_train_{'_'.join(map(str, train_subjects))}_test_P{args.test_subject}"
    # 源码哈希帮助后续确认是否确实按同一实现复跑；数据清单仅记文件大小，
    # 不能替代数据文件的完整内容哈希，不宣称其为严格数据完整性校验。
    source_names = ("fixed_subject_test.py", "subject_holdout_baseline.py", "audit_kinematics_alignment.py")
    source_hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
                     for name in source_names}
    configuration = {
        "train_subjects": train_subjects, "test_subject": args.test_subject,
        "runs": list(range(1, 10)), "fs": FS, "input_points": N_BEFORE,
        "target_points": N_AFTER, "channels": list(PAPER_CHANNELS),
        "eeg_processing": "21-channel average reference; continuous causal fourth-order Butterworth 0.1-40Hz; no ICA",
        "features": "63 log std features from three consecutive 50-point windows",
        "ridge_alpha": RIDGE_ALPHA, "standardization": "training subjects only",
        "target_processing": "raw wrist XYZ; subtract own onset position; no target filter/resampling",
        "primary_metric": "MAE over all trials, 500 time samples and XYZ",
        "secondary_metrics": ["XYZ MAE", "trial wins vs training mean trajectory", "runwise MAE"],
        "plots": "mean trajectories and first trial; no best/worst trial selection",
        "source_sha256": source_hashes, "file_size_inventory": inventory,
        "numpy_version": np.__version__,
    }
    protocol_path = report_dir / f"{stem}_protocol.json"
    rerun = protocol_path.exists()
    if rerun:
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous["configuration"] != configuration:
            raise ValueError("已有协议与当前配置/源码/文件大小不一致；请核查差异，不自动覆盖原协议")
        print("已有协议匹配；本次为同配置复跑，不是另一次独立测试", flush=True)
    else:
        save_json(protocol_path, {"recorded_before_test_utc": datetime.now(timezone.utc).isoformat(),
                                  "configuration": configuration})
    print(f"配置已记录：{protocol_path}", flush=True)

    train_features, train_targets, counts = [], [], {}
    for subject in train_subjects:
        eeg, targets, runs = extract_subject(args.data_root, subject)
        train_features.append(make_features(eeg))
        train_targets.append(targets)
        counts[f"P{subject}"] = {"trials": len(targets), "runs": runs}
        print(f"训练P{subject}: EEG={eeg.shape}；目标={targets.shape}；runs={runs}", flush=True)
    x_train, y_train = np.concatenate(train_features), np.concatenate(train_targets)
    model = fit_model(x_train, y_train)
    print(f"模型已拟合：训练{len(y_train)}次；alpha={RIDGE_ALPHA}；现在检查并读取测试者", flush=True)

    # 只用预先固定的内部一致性规则检查P5；不按误差或运动幅度删试次。
    # 检查失败时停止，不自动移动时间索引、翻转坐标或重选参数。
    audit_subject(args.data_root, args.test_subject)
    eeg_test, y_test, runs = extract_subject(args.data_root, args.test_subject)
    counts[f"P{args.test_subject}"] = {"trials": len(y_test), "runs": runs}
    prediction = predict(model, make_features(eeg_test))
    if not np.isfinite(prediction).all():
        raise ValueError("预测包含非有限数值，停止评分")
    baseline = model["baseline"]
    base_error, eeg_error = np.abs(baseline[None] - y_test), np.abs(prediction - y_test)
    base_trial, eeg_trial = base_error.mean(axis=(1, 2)), eeg_error.mean(axis=(1, 2))
    baseline_xyz, eeg_xyz = base_error.mean(axis=(0, 1)), eeg_error.mean(axis=(0, 1))
    wins = int(np.sum(eeg_trial < base_trial))
    print(f"\n测试P{args.test_subject}: EEG={eeg_test.shape}；目标={y_test.shape}；runs={runs}", flush=True)
    print(f"基线XYZ={np.round(baseline_xyz, 6)}\nEEG XYZ={np.round(eeg_xyz, 6)}", flush=True)
    print(f"整体基线/EEG={baseline_xyz.mean():.6f}/{eeg_xyz.mean():.6f}；"
          f"EEG获益={wins}/{len(y_test)}", flush=True)

    run_rows, trial_rows, start = [], [], 0
    for run, number in enumerate(runs, 1):
        stop = start + number
        row = {"run": run, "trials": number,
               "baseline_xyz": base_error[start:stop].mean(axis=(0, 1)).tolist(),
               "eeg_xyz": eeg_error[start:stop].mean(axis=(0, 1)).tolist(),
               "baseline_overall": float(base_trial[start:stop].mean()),
               "eeg_overall": float(eeg_trial[start:stop].mean()),
               "eeg_wins": int(np.sum(eeg_trial[start:stop] < base_trial[start:stop]))}
        run_rows.append(row)
        print(f"  S{run}: {number}次；基线/EEG={row['baseline_overall']:.3f}/{row['eeg_overall']:.3f}；"
              f"获益={row['eeg_wins']}/{number}", flush=True)
        for i in range(start, stop):
            trial_rows.append({"trial": i + 1, "run": run, "trial_in_run": i - start + 1,
                               "baseline_mae": float(base_trial[i]), "eeg_mae": float(eeg_trial[i])})
        start = stop
    assert start == len(y_test)
    report_path = report_dir / f"{stem}.json"
    save_json(report_path, {
        "protocol": protocol_path.name, "configuration": configuration,
        "scored_utc": datetime.now(timezone.utc).isoformat(), "same_protocol_rerun": rerun,
        "alignment_audit": "passed internal time, Lift sequence and sampled HS/WS C3/XYZ checks; physical units unconfirmed",
        "counts": counts, "train_trials": len(y_train), "test_trials": len(y_test),
        "baseline_xyz": baseline_xyz.tolist(), "eeg_xyz": eeg_xyz.tolist(),
        "baseline_overall": float(baseline_xyz.mean()), "eeg_overall": float(eeg_xyz.mean()),
        "eeg_wins": wins, "runwise": run_rows, "trialwise": trial_rows,
        "scope": "one new subject, fixed prior configuration; not five-subject LOSO or full paper reproduction",
    })

    # 预先选定平均轨迹与第1次抓握作图，不挑有利试次；使用英文防止字体缺失。
    times = np.arange(N_AFTER) / FS
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for axis, name in enumerate(("X", "Y", "Z")):
        for col, truth, estimate in ((0, y_test.mean(axis=0), prediction.mean(axis=0)),
                                     (1, y_test[0], prediction[0])):
            ax = axes[axis, col]
            ax.plot(times, truth[:, axis], label="True", linewidth=1.8)
            ax.plot(times, baseline[:, axis], label="Train mean baseline", linewidth=1.5)
            ax.plot(times, estimate[:, axis], label="EEG prediction", linewidth=1.5)
            ax.set_ylabel(f"{name} displacement (raw units)")
            ax.grid(alpha=0.25)
            if axis == 0:
                ax.set_title("All test trials: mean" if col == 0 else "Test trial 1 (preselected)")
                ax.legend(fontsize=8)
            if axis == 2:
                ax.set_xlabel("Time after movement onset (s)")
    fig.suptitle(f"Fixed training P{', P'.join(map(str, train_subjects))}; test P{args.test_subject}")
    fig.tight_layout()
    figure_dir = root / "figures" / "diagnostics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{stem}.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"\n报告已保存：{report_path}\n图已保存：{figure_path}", flush=True)


if __name__ == "__main__":
    main()
