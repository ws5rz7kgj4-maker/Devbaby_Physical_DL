# Sources and verification / 来源与复核

Documentation date / 文档日期: **2026-11-11**.

## Evidence map / 证据索引

| Study / 研究 | Included evidence / 所附证据 | Upstream location / 原位置 |
|---|---|---|
| Early formal physical control / 早期正式控制 | [Early source map](../results/raw/early/provenance.json), formal CSV/JSON and source definitions | [formal-physical-eef2ff40](https://huggingface.co/datasets/swenlio/devbaby-results/tree/main/runs/formal-physical-eef2ff40) |
| Early supplement / 早期补充 | `results/raw/early/supplement_*` | [supplement-c2b8f789d4b0](https://huggingface.co/datasets/swenlio/devbaby-supplement-results/tree/main/runs/supplement-c2b8f789d4b0) |
| Later world-model protocol / 后期世界模型协议 | [Protocol](../results/raw/formal_protocol.json), [plan](../results/raw/formal_plan.json), [fit seal](../results/raw/formal_fit_seal.json) | [worldmodel-dacc6ff327cf](https://huggingface.co/datasets/swenlio/devbaby-worldmodel-results/tree/main/runs/worldmodel-dacc6ff327cf) |
| Exploration pilot / 随机探索 | `results/raw/pilot_*.json` and source-archive checksum | Original `DevBaby_Exploration_Pilot.zip`, synthetic sensory files omitted from this compact release / 原完整包，未在此重打包感官大文件 |
| Frozen audit / 冻结审计 | [Summary](../results/raw/frozen_control_summary.json), protocol/status | [frozen-control-v1-ba8f8d332928](https://huggingface.co/datasets/swenlio/devbaby-control-audit-results/tree/main/audits/frozen-control-v1-ba8f8d332928) |
| FP32 comparison / FP32 对照 | [Summary](../results/raw/baseline_summary.json), protocol/status | [4fc555c1bed2](https://huggingface.co/datasets/swenlio/devbaby-control-audit-results/tree/main/baseline-checks/4fc555c1bed2) |
| CPU/GPU audits / 参数与缓存 | [CPU](../results/raw/parameter_cpu_result.json), [GPU](../results/raw/parameter_gpu_result.json), exact scripts | [parameter-audits](https://huggingface.co/datasets/swenlio/devbaby-control-audit-results/tree/main/parameter-audits) |

Upstream repositories are private; links can require the owner's authorization.
All headline numerical tables can be checked from the compact evidence included
here without upstream access. Complete model checkpoints, sensory trajectories,
all physical-panel recordings and every scene-level bootstrap input are not
distributed here. No access token or credential is included.

原仓库为私有，链接可能需要所有者授权。主要数值表可仅凭本包轻量证据核对，
无需访问上游。全部模型权重、感官轨迹、物理面板记录和所有场景级 bootstrap 输入
未完整分发。包内没有访问令牌或凭据。

## Historical source archives / 历史源码包

[source_archives.json](source_archives.json) records byte sizes and SHA-256 hashes.
Six compact ZIPs are included under `source_archives/`: early physical/supplement,
world-model numerical recovery, frozen audit, FP32 comparison and parameter audit.
The approximately 52 MB exploration sensory archive is identified by checksum;
only lightweight result inputs are included here.

[source_archives.json](source_archives.json) 记录大小和 SHA-256。`source_archives/`
保留六个小型历史 ZIP：早期物理/补充、世界模型数值恢复、冻结审计、FP32 对照、
参数审计。约 52 MB 的完整探索感官包记录哈希，本包只包含轻量结果输入。

Historical archive names are not current validation claims. In particular,
`DevBaby_WorldModel_Fixed.zip` still contains the later cache defect. Smoke tests
inside archives are implementation checks, not formal benchmark results. Source
ZIP hashes, runner-file hashes and protocol hashes are different objects and must
not be substituted for each other.

历史文件名不代表当前验证状态。尤其 `DevBaby_WorldModel_Fixed.zip` 仍含后来发现的
缓存缺陷。包内 smoke 测试不是正式基准结果。源码 ZIP、runner 文件和协议的哈希
指向不同对象，不能互换。

## What was recomputed / 本次复算范围

- Early OOD developmental success counts and sealed endpoints were checked against CSVs and eight per-seed endpoint JSONs.
- Supplementary primary means, exact paired sign-flip p-values and six-way Holm correction were recomputed. Archived crossed-bootstrap intervals were retained; this pass did not reconstruct them from every scene-level record.
- Pilot means were checked against all 96 episode records and their eight paired initial scenes.
- FP32/frozen summaries were turned into tables without fitting new models.
- Parameter claims were checked against all recorded tensors and cache modes, including present-but-zero FP16 gradients.
- Local links, input checksums and table derivations were validated. No new neural training, GPU job or physical rollout was performed.

- 从 CSV 和八份逐种子终点 JSON 复核早期 OOD 成功计数与封存终点。
- 复算补充主对比均值、精确配对符号翻转 p 值和六项 Holm 校正；交叉 bootstrap 区间沿用归档统计，未从全部场景逐项重算。
- 从 96 个 episode 记录及八个配对场景复核 pilot 均值。
- 将 FP32/冻结汇总转为表格，没有拟合新模型。
- 核对全部记录参数与缓存条件，包括存在但为零的 FP16 梯度。
- 检查本地链接、输入哈希与表格推导；没有新增神经训练、GPU 作业或物理 rollout。

## Integrity and status / 完整性与状态

[jobs.json](jobs.json) separates saved-result completion from historical platform
states. It is not a new live-job listing. A fitting seal confirms a workflow stage,
not complete evaluation or model competence. Historical canceled-run panel counts
are not silently promoted into current statistics.

[jobs.json](jobs.json) 区分保存结果的完成标记与历史平台状态，不是实时作业列表。
fitting seal 只记录流程阶段，不代表评估完成或模型有效。取消运行的历史面板计数
不会自动进入当前统计。

[content_sha256.json](content_sha256.json) covers this release's non-figure files
and excludes itself and generated caches. Input-specific manifests also accompany
the table builder. Checksums verify byte identity, not scientific correctness or
independent reproduction of training.

[content_sha256.json](content_sha256.json) 覆盖本版本非图像文件，排除自身与生成缓存；
表格构建器另带输入专用清单。哈希验证字节身份，不等于科学结论正确或训练已独立复现。
