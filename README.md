# WAY-EEG-GAL 复现练习

这是 WAY-EEG-GAL 数据集的个人学习与论文复现工作记录。先阅读 [阶段总结](STAGE_SUMMARY.md) 了解当前结论，再查看 [实验结果](RESULTS.md) 中的完整数值与复跑命令。早期逐步学习过程保存在 [`notebooks/01_inspect_subject.ipynb`](notebooks/01_inspect_subject.ipynb)，后续多被试实验使用 [`scripts/`](scripts/) 中的脚本，图表见 [诊断图索引](figures/diagnostics/README.md)。

## 当前进度

- 检查 P1 的第 1 段记录（P1 S1），读取 EEG、手部运动轨迹和试次事件。
- 绘制 EEG 与三维位置，核对试次事件时间；Notebook 中较早的一幅事件图存在约 2 秒对齐偏移，后续使用 WS 时间窗的单元格为修正版本。
- 初步构造 34 次抓握的 EEG 输入和轨迹目标；从 32 个 EEG 通道选出论文使用的 21 个通道。
- 探索 0.1–40 Hz 滤波、平均重参考、坏通道与 ICA。P1 S1 中 T8 暂标为坏通道；在该段探索性地排除 ICA000，并检查清理前后波形。
- P1 的 9 段共提取 294 次抓握：运动前 0.3 秒的 21 通道 EEG 为 `(294, 150, 21)`，运动后 1 秒手腕三维位移为 `(294, 500, 3)`。训练基线采用因果 0.1–40 Hz Butterworth 滤波、平均重参考、3 个时间窗的对数标准差特征、固定 `alpha=100` 的岭回归。**九段训练基线没有使用前述 ICA 清理结果。**
- 使用留一段测试（其余 8 段训练）进行 P1 内跨记录验证；每轮标准化和平均轨迹基线只使用训练段。EEG 模型在 9 段中的 7 段优于平均轨迹基线，在 294 次抓握中的 173 次取得更低 MAE。逐段数值见 [实验结果](RESULTS.md)。
- 已完成 P1–P4 四轮留一受试者评估，共1176次抓握。EEG整体MAE在P1/P2优于平均轨迹基线，在P3/P4更差；四人平均为基线1.915、EEG1.952。
- 已完成时间与坐标内部核查、P3 Y/P4 Z按段误差和预测修正诊断，以及通道消融、训练轨迹低通、内层选择修正强度的探索性对照。P3 Y与P4 Z的平均修正方向相反，退步遍布九段；上述对照尚未形成稳定改善。
- 已完成EEG输入100Hz对照，共同500点轨迹评分：四人EEG平均MAE为原500Hz的1.952、抗混叠后500Hz的2.045、抗混叠后100Hz的2.053。参数和未舍入指标见[报告](reports/sampling_rate_experiment.json)。
- 已完成目标输出网格对照，固定原500Hz EEG。共同0–0.990秒区间的EEG平均MAE为500点输出1.950420、100点输出插值后1.950428；额外保留0.998秒末点的完整区间对照也几乎不变。它只检查输出网格密度，不包含目标抗混叠滤波，见[报告](reports/target_sampling_rate_experiment.json)。
- 已固定P1–P4训练、首次测试P5，共1176次训练/294次测试。测试前记录配置，原21通道、500Hz和alpha=100不变。P5整体MAE为基线1.893119、EEG1.874303，约改善0.99%；X/Z改善、Y退步，获益145/294次。该结果是单个新被试测试，不是五人留一被试汇总，见[报告](reports/fixed_train_1_2_3_4_test_P5.json)。
- 同一固定P1–P4训练配置的P6测试报告也已保存：基线1.417633、EEG1.752760，获益75/294次，XYZ均退步。该结果与P5分别保留，不根据测试结果调参，见[P6报告](reports/fixed_train_1_2_3_4_test_P6.json)。

目前已完成单受试者跨记录与四人跨受试者的探索性基线，尚未完整复现论文模型和处理流程，也尚未获得跨被试稳定获益。原始EEG与手腕坐标的物理单位仍需确认。早期探索单元使用过零相位滤波；当前基线采用因果滤波。原始数据需单独准备。

## 运行

1. 准备 WAY-EEG-GAL 原始数据；原始数据不提交到本仓库。
2. 使用 Python 3.12 环境，并安装 `requirements.txt` 中的依赖。本机使用的是 Conda 环境 `pytorch_new`。
3. 打开 Notebook，将首个数据路径单元格的 `D:\biosignal-data\WAY-EEG-GAL\P1` 改成你自己的 P1 数据位置，然后从头按顺序运行。

Notebook 的图表和运行结果已保存，可先直接阅读；重新运行前请确认输入文件路径、数据单位和事件时间基准。

当P1–P4原始文件准备好时，在已激活 `pytorch_new` 的终端中，从仓库根目录运行：

```powershell
python -X utf8 scripts/subject_holdout_baseline.py --data-root D:\biosignal-data\WAY-EEG-GAL --subjects 1 2 3 4
```

脚本逐一留出一个受试者，输出平均轨迹基线和 EEG 模型的分方向及整体 MAE。新增受试者后，把编号追加到 `--subjects` 即可。

若要复跑本次**固定P1–P4训练、只测试P5**的实验，请运行下面的独立脚本，不要把它与五人留一被试混为一谈：

```powershell
python -X utf8 scripts/fixed_subject_test.py --data-root D:\biosignal-data\WAY-EEG-GAL
```

脚本会核对先前记录的配置与源码哈希；同配置再次运行属于复跑，不是新的独立测试。详见[测试前协议](reports/fixed_train_1_2_3_4_test_P5_protocol.json)。

固定P1–P4训练、只测试P6的复跑命令：

```powershell
python -X utf8 scripts/fixed_subject_test.py --data-root D:\biosignal-data\WAY-EEG-GAL --train-subjects 1 2 3 4 --test-subject 6
```

## 下一步

先固定当前流程与已有诊断，再扩充未参与当前模型选择的被试进行验证。EEG输入100Hz首轮对照及目标输出网格密度对照已完成，后续可分别实现含滤波的目标重采样、归一化与论文模型，并持续报告平均轨迹基线。四人探索性结果与论文完整跨被试评估应分别解释。
