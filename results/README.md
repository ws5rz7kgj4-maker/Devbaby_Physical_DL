# Result files / 结果文件

CSV tables under `tables/` are derived from bundled JSON/CSV inputs by
[`../scripts/build_results.py`](../scripts/build_results.py). Values are not
rounded in CSV. Figures round display labels only. These scripts perform no
training and start no remote jobs.

`tables/` 中的 CSV 由所附 JSON/CSV 自动生成；`raw/` 中的原始 CSV 按归档保存。生成表保留原始数值精度；图中的数字标签仅为
显示而四舍五入。脚本不进行训练、不启动远程作业。

## Files / 文件

| File / 文件 | Content / 内容 |
|---|---|
| `tables/early_development_by_seed.csv` | Earlier physical-v1 success by model seed and age / 早期 physical-v1 按模型种子和训练年龄的成功率 |
| `tables/early_development_macro.csv` | Earlier physical-v1 equal-weight IID/OOD macro success curves / 早期 physical-v1 等权 IID/OOD 宏平均成功率曲线 |
| `tables/early_sealed_final_comparison.csv` | Separate sealed-final OOD results including references / 独立封存终点评测 OOD 成功率，含参考基线 |
| `tables/pilot_conditions.csv` | 12 exploration conditions; paired-scene means / 12 个探索条件的配对场景均值 |
| `tables/pilot_reference.csv` | Privileged reference success counts / 使用特权状态的参考控制器成功次数 |
| `tables/baseline_all.csv` | 48 FP32 prediction panels / 48 个 FP32 预测评测组合 |
| `tables/baseline_200_validation.csv` | 200-update development-validation subset / 200 次更新后的开发验证集子表 |
| `tables/frozen_control_all.csv` | 4 training ages × 3 horizons; compromised-training diagnostic / 4 个训练阶段 × 3 个预测长度，仅作有缺陷训练的诊断 |
| `tables/frozen_control_final.csv` | Final age of that frozen audit / 上述冻结审计的最终训练阶段 |
| `tables/gpu_parameter_updates.csv` | All 44 parameter tensors × 3 precision/cache modes / 全部 44 个参数张量 × 3 个精度及缓存条件 |
| `tables/gpu_mode_summary.csv` | Gradient and update counts per mode / 各模式的梯度与更新计数 |
| `tables/cpu_parameter_updates.csv` | CPU FP32 single-step checks from age 0 and 65,536 / 两个起始阶段的 CPU FP32 单步检查 |
| `tables/checkpoint_parameter_changes.csv` | Exact initial-to-final checkpoint comparison for seeds 0 and 1 / 种子 0、1 初末检查点的精确逐参数比较 |
| `tables/checkpoint_group_changes.csv` | Same checkpoint comparison aggregated by module / 同一检查按模块汇总 |
| `source_sha256.json` | SHA-256 of the 10 JSON inputs used by the table builder / 表格生成器使用的 10 个 JSON 输入的 SHA-256 |
| `early_source_sha256.json` | SHA-256 of the 5 additional early physical-v1 source files / 新增 5 个早期 physical-v1 来源文件的 SHA-256 |

`source_sha256.json` checks the bytes in this package; it is not a claim that all
external checkpoints or complete sensory archives are distributed here.

`source_sha256.json` 用于检查本包中的输入字节，不能据此认为本包包含了所有外部
检查点或全部感官数据。

## Units and interpretation / 单位与解释

The **early physical-v1 system** is a body-prediction/control experiment using
explicit coordinate/contact goals and a programmed CEM cost. Its physical-success
curves must be distinguished from the later world-model prediction audits.
`early_development_macro.csv` averages domain/task cells equally within each
model seed, then averages the eight seeds equally. OOD excludes the IID domain.
Each OOD age has 8 seeds × 4 domains × 3 tasks × 6 scenes = 576 scored seed-scene
trials; these are not 576 independent scenes. The development endpoint is
427/576 = 74.13%. The separate sealed-final full-model panel is
1,680/2,304 = 72.92%; it is not an additional point on that development curve.
Random and privileged-reference OOD results each contain 288 scored trials.
No confidence intervals are shown on the early figure; the fine development
lines show the eight individual full-model seeds. Archived seed-only intervals
for the sealed-final panel must not be reused as development-curve intervals.

**早期 physical-v1 系统**研究身体预测与控制，使用显式坐标/接触目标及程序化 CEM
代价。它的物理成功率曲线必须与后期世界模型的预测审计区分。
`early_development_macro.csv` 先在每个模型种子内部对领域×任务组合等权平均，再对
8 个模型种子等权平均。OOD 排除 IID 领域。每个 OOD 年龄点含
8 种子 × 4 领域 × 3 任务 × 6 场景 = 576 次计分，不是 576 个独立场景。
开发曲线终点为 427/576 = 74.13%；独立封存 full 模型终点为
1,680/2,304 = 72.92%，不能把它接到开发曲线上。随机和特权参考 OOD 各有 288 次计分。
早期图不显示置信区间；开发图细线表示 8 个 full 模型种子。封存终点已有的 seed-only
区间不能用于开发曲线。

| Field / 字段 | Definition / 定义 |
|---|---|
| `contact_fraction`, `contact_pct` | Fraction of episode timesteps in contact; `_pct` is ×100 / 回合内发生接触的时间步占比；`_pct` 等于比例乘 100 |
| `action_effect_2cm_fraction`, `action_effect_2cm_pct` | Fraction of timesteps with at least one object's position >2 cm from the matched idle trajectory; includes persistent displacement / 至少一个物体相对同初始场景 idle 轨迹偏移超过 2 cm 的时间占比，包含持续位移 |
| `hold_steps` | Number of steps retaining a sampled waypoint, not a constant-action duration / 保持同一采样目标点的步数，不表示动作数值恒定 |
| `noise_sd_before_clipping` | Nominal translational noise before clipping / 截断前的名义平移噪声标准差 |
| `action_saturation_fraction` | Fraction of translational action components with absolute value ≥0.999 / 平移动作分量绝对值 ≥0.999 的比例 |
| `actual_object_mse` | RGB MSE in the union of initial and target object masks, using the actual action sequence / 使用实际动作序列，在起始和目标物体掩码并集内计算的 RGB 均方误差 |
| `zero_object_mse` | Same target with the candidate input action sequence zeroed / 目标不变，输入动作序列置零后的误差 |
| `reversed_object_mse` | Same target with time order of actions reversed, not sign-negated / 目标不变，动作顺序倒置后的误差；不是对动作取负 |
| `persistence_object_mse` | Initial image copied forward as the prediction / 用起始图像直接作为未来预测的误差 |
| `actual_full_image_mse` | RGB MSE across the full image, not only objects / 整幅图像的 RGB 均方误差 |
| `training_transitions` | Unique collected transitions available at a training age, not optimizer steps / 该训练阶段可用的唯一采集交互步数，不是优化器更新次数 |
| `distance_advantage_over_uniform_m` | Mean candidate final distance minus the model-selected candidate's distance; positive is better / 所有候选终点距离的均值减去模型选择候选的距离；正值更好，单位为米 |
| `ranking_seed_scene_rows` | 8 model seeds × 4 scenes = 32 rows, not 32 independent scenes / 8 模型种子 × 4 场景 = 32 行，不是 32 个独立场景 |
| `exact_equal`, `exactly_unchanged` | Bit-for-bit identical parameter tensors between the audited checkpoints / 所审计检查点之间逐元素完全一致的参数张量 |
| `grad_present` | Gradient is not `None`; this alone does not imply a nonzero gradient / 梯度不为 `None`；不代表梯度非零 |
| `grad_finite` | All gradient values finite; absent gradients are blank in CSV / 梯度数值均有限；缺失梯度在 CSV 中为空 |
| `step_max_abs_change` | Maximum absolute elementwise change in one optimizer step / 单次优化器更新前后逐元素绝对变化的最大值 |

The exploration pilot uses 8 **paired** initial scenes across 12 conditions:
96 episodes and 12,288 timesteps do not imply 96 independent scenes or 12,288
independent samples. The plotted means are descriptive. Bootstrap intervals in
`raw/pilot_analysis.json` are exploratory and not corrected for multiple comparisons.

探索 pilot 在 12 个条件中复用 8 个**配对**初始场景。96 回合、12,288 时间步不等于
96 个独立场景或 12,288 个独立样本。图中展示描述性均值；
`raw/pilot_analysis.json` 的 bootstrap 区间仅用于探索，未作多重比较校正。

The FP32 learning figure connects only three measured update counts (0, 50, 200)
at horizon 64. It is a prediction-error diagnostic on previously inspected
development data, not a physical-success curve or a fitted scaling law.

FP32 图仅连接预测长度为 64 时三个实测更新点（0、50、200）；它是已看过的开发
数据上的预测误差诊断，不是物理成功率曲线或 scaling law 拟合。

The GPU single-step diagnostic has **no GradScaler**. Cache-off restores finite
gradients for all 44 tensors and updates all 8 recurrent tensors. However,
`image_start.weight` and `image_start.bias` have present but zero gradients and
zero updates in both FP16 modes in this particular unscaled step. Thus the
artifact supports neither “all parameters updated under FP16” nor “the complete
repaired training path passed.”

GPU 单步诊断**没有使用 GradScaler**。关闭缓存后 44 个张量均有有限梯度，8 个循环
模块张量均发生更新；但本次不带缩放的 FP16 单步中，`image_start.weight` 和
`image_start.bias` 在两种缓存模式下均为零梯度、零更新。因此不能写成“FP16 下全部
参数已更新”，也不能写成“修复后的完整训练路径已经通过”。

Raw parameter JSON includes the source text of the audited model. It is retained
as provenance for the failed implementation, not advertised as repaired training code.

参数审计原始 JSON 内保留了受审计模型的源代码文本。这是有缺陷实现的溯源记录，
不能作为已经修复并完成验证的训练代码使用。
