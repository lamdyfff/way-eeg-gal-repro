r"""只读核查 P1–P4 的运动开始索引、手腕坐标列与数值尺度。

运行：python -X utf8 scripts/audit_kinematics_alignment.py --data-root D:\biosignal-data\WAY-EEG-GAL

本脚本不会修改原始数据或重新训练模型。它能核查内部时间/列对齐，
但仅凭 .mat 的列名和数值不能证明手腕坐标的物理单位或实验室坐标方向。
"""

import argparse
from pathlib import Path

import numpy as np

from subject_holdout_baseline import FS, N_AFTER, N_BEFORE, WRIST_AXES, read_mat


def audit_subject(data_root, subject):
    directory = data_root / f"P{subject}"
    events = read_mat(directory / f"P{subject}_AllLifts.mat", "P")
    columns = list(events.ColNames)
    run_col, lift_col, start_col, onset_col = (
        columns.index(name) for name in ("Run", "Lift", "StartTime", "tHandStart")
    )
    timing_error_ms = []
    onset_samples = []
    start_positions = []
    end_displacements = []
    direction_counts = np.zeros(3, dtype=int)
    trials = 0
    names_per_run = []

    for run in range(1, 10):
        hs = read_mat(directory / f"HS_P{subject}_S{run}.mat", "hs")
        ws = read_mat(directory / f"WS_P{subject}_S{run}.mat", "ws")
        kin_names = list(hs.kin.names)
        wrist_cols = [
            next(i for i, name in enumerate(kin_names)
                 if str(name).startswith(axis + " "))
            for axis in WRIST_AXES
        ]
        names_per_run.append(tuple(str(kin_names[i]) for i in wrist_cols))
        c3_col = list(hs.eeg.names).index("C3")
        rows = events.AllLifts[events.AllLifts[:, run_col] == run]
        if len(rows) != len(ws.win):
            raise ValueError(f"P{subject} S{run}: 事件/窗口数不一致")
        lifts = rows[:, lift_col].astype(int)
        if not np.array_equal(np.sort(lifts), np.arange(1, len(rows) + 1)):
            raise ValueError(f"P{subject} S{run}: Lift 编号不连续或重复")

        for row in rows:
            lift = int(row[lift_col])
            window = ws.win[lift - 1]
            t = np.asarray(window.eeg_t, dtype=float)
            onset = float(row[onset_col])
            start = round((float(row[start_col]) - 2.0) * FS)
            index = int(np.searchsorted(t, onset))
            if not (N_BEFORE <= index and index + N_AFTER <= len(t)):
                raise ValueError(f"P{subject} S{run} lift {lift}: 运动前/后窗口越界")
            if len(t) != len(window.eeg) or len(t) != len(window.kin):
                raise ValueError(f"P{subject} S{run} lift {lift}: 窗口时间轴长度不一致")
            if not np.allclose(np.diff(t), 1 / FS, atol=1e-9):
                raise ValueError(f"P{subject} S{run} lift {lift}: 采样时间间隔异常")
            if not (t[index - 1] < onset <= t[index]):
                raise ValueError(f"P{subject} S{run} lift {lift}: onset 索引不正确")
            # HS 连续记录和 WS 切窗互相验证：EEG 与手腕三列都必须逐点相同。
            for hs_signal, ws_signal, cols, label in (
                (hs.eeg.sig, window.eeg, [c3_col], "C3"),
                (hs.kin.sig, window.kin, wrist_cols, "手腕 XYZ"),
            ):
                offsets = np.array([0, index, index + N_AFTER - 1])
                if start < 0 or start + offsets[-1] >= len(hs_signal):
                    raise ValueError(f"P{subject} S{run} lift {lift}: HS 索引越界")
                if not np.allclose(
                    hs_signal[start + offsets[:, None], cols],
                    ws_signal[offsets[:, None], cols],
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"P{subject} S{run} lift {lift}: HS/WS {label} 不一致"
                    )

            wrist = np.asarray(window.kin[index:index + N_AFTER, wrist_cols], dtype=float)
            if wrist.shape != (N_AFTER, 3) or not np.isfinite(wrist).all():
                raise ValueError(f"P{subject} S{run} lift {lift}: 手腕轨迹无效")
            displacement = wrist - wrist[0]
            timing_error_ms.append((t[index] - onset) * 1000)
            onset_samples.append(index)
            start_positions.append(wrist[0])
            end_displacements.append(displacement[-1])
            direction_counts += displacement[-1] > 0
            trials += 1

    if len(set(names_per_run)) != 1:
        raise ValueError(f"P{subject}: 各段手腕列名/顺序不同")
    starts = np.stack(start_positions)
    ends = np.stack(end_displacements)
    print(f"P{subject}: {trials} 次；手腕列={names_per_run[0]}")
    print(
        f"  运动开始索引范围={min(onset_samples)}–{max(onset_samples)}；"
        f"相对采样点的时间误差范围={min(timing_error_ms):.3f}–"
        f"{max(timing_error_ms):.3f} ms"
    )
    print(f"  起始绝对坐标均值 XYZ={np.round(starts.mean(axis=0), 3)}")
    print(f"  终点相对位移均值 XYZ={np.round(ends.mean(axis=0), 3)}")
    print(f"  终点正方向次数 XYZ={direction_counts.tolist()}/{trials}")
    print("  HS/WS 的 C3 和手腕 XYZ 抽样逐点核对：全部一致")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    for subject in args.subjects:
        audit_subject(args.data_root, subject)


if __name__ == "__main__":
    main()
