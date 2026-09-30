# 诊断图索引

当前共16张图，包括固定21通道岭回归的P3/P4诊断、四人EEG采样率、目标输出网格对照及固定P1–P4训练的P5/P6测试。手腕数值使用原始位置单位。

| 图 | 阅读方法 |
|---|---|
| [P3平均轨迹与逐时间点误差](diagnostics_P3_all_trials.png) / [P4](diagnostics_P4_all_trials.png) | 左列比较真实均值、平均轨迹与EEG预测；右列比较每个时间点的MAE |
| [P3最差Y试次](diagnostics_P3_worstY.png) / [P4](diagnostics_P4_worstY.png) | 都按Y方向退步选择试次；P4这张不能当作其最差Z试次 |
| [P3特征偏移](diagnostics_P3_feature_shift.png) / [P4](diagnostics_P4_feature_shift.png) | 以训练者标准差为单位的特征中位数差，以及整体特征距离分布 |
| [P3 Y通道修正](diagnostics_P3_Y_channel_impact.png) / [P4 Z](diagnostics_P4_Z_channel_impact.png) | 通道对平均预测修正的贡献，以及固定权重下的中位数替换敏感性；不是删除通道重训结果 |
| [P3 Y按段误差](diagnostics_P3_Y_runwise.png) / [P4 Z](diagnostics_P4_Z_runwise.png) | 左侧正柱代表EEG比基线差；右侧为每段真实终点位移分布 |
| [P3 Y修正方向](diagnostics_P3_Y_correction.png) / [P4 Z](diagnostics_P4_Z_correction.png) | 左侧比较真实所需与EEG实际平均修正；右侧为逐试次1秒终点修正散点 |
| [EEG输入采样率对照](sampling_rate_control_mae.png) | AA表示额外抗混叠滤波；左侧比较四人基线与三种输入模型的整体MAE，右侧比较同一抗混叠信号100Hz相对500Hz的分轴MAE变化；全部用原始500点轨迹计分 |
| [目标输出网格对照](target_sampling_rate_control_mae.png) | EEG固定原500Hz；左侧统一在0–0.990秒评分，比较密集500点输出与100点输出插值；右侧额外保留0.998秒末点，共101点，统一对完整500点评分。Dense/Sparse表示输出网格，不含新增目标滤波 |
| [固定训练P1–P4、测试P5](fixed_train_1_2_3_4_test_P5.png) | 左列为P5所有抓握均值，右列为预先固定的测试第1次；蓝色真实、橙色训练者平均轨迹、绿色EEG预测。不是五人留一被试结果，均值图不能替代逐次指标 |
| [固定训练P1–P4、测试P6](fixed_train_1_2_3_4_test_P6.png) | 与P5相同的固定训练者和画图规则；已保存报告显示P6总体及XYZ均退步，不能用均值曲线接近代替模型逐次成绩 |

总体结论见 [阶段总结](../../STAGE_SUMMARY.md)，完整数字与复跑命令见 [RESULTS.md](../../RESULTS.md)。
