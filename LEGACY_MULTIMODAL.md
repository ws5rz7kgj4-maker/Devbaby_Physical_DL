# DevBaby — 全合成的具身发展学习实验

本项目把一个约 102 万参数、随机初始化的神经网络放进程序化 3D 物理世界。
它从第一人称画面、合成声音、自己发出的动作和本体感觉中学习。
物体属性、事件、语音转录和指称关系完整保存在独立标注目录，主训练器不读取它们。

**这是可运行、可复核的研究基线。它不预设语言一定学会，不把成功率提高解释为意识出现。**
当前语言测量是声音与物体/事件的匹配；动作测量是根据身体坐标目标实际执行伸手。
“听懂完整指令后自主抓取”“自然语言生成”“人类婴儿智力/意识”都不在本版本已实现的能力范围内。

## 1. 在你的 Mac 终端提交 Hugging Face Job

解压后，把整个 `devbaby` 文件夹放在 Downloads。Mac 只需要装启动器依赖，
PyTorch、模拟器、语音引擎都在云端安装。**不需要把 HF token 发给聊天窗口。**

```bash
cd "$HOME/Downloads/devbaby"
python3 -m venv .venv-launch
source .venv-launch/bin/activate
python -m pip install -r requirements-launcher.txt
hf auth login
python hf_launch.py submit --profile smoke --flavor t4-small
```

登录 token 需要创建/写入自己仓库与启动 Jobs 的权限；HF 账户需要可用的 Jobs 余额。
`submit` 会启动一个计费 Job，并在你的账号下创建/复用两个 **private dataset repositories**：
`你的用户名/devbaby-source` 保存代码，`你的用户名/devbaby-results` 保存结果。
本包不会自动替你提交；只有你执行 `submit` 或 sweep 的 `--execute` 才提交。

先运行 smoke，确认依赖、CUDA、声音、数据记录、训练、保存都正常。
它只有 4 个训练 episode、96 个交互步、8 次梯度更新，**不能用来宣称学会语言**。

查看最近提交的 Job：

```bash
python hf_launch.py status
python hf_launch.py logs
```

日志里有 `"status": "completed"` 且 HF Job 状态为 `COMPLETED` 后下载：

```bash
python hf_launch.py download --directory outputs
```

终端会打印 `report.html` 的确切位置，用浏览器打开即可看成长曲线和有声录像。
目录为 `outputs/runs/运行编号/`。不需要在 Mac 上装 PyTorch 就能看结果。

## 2. 跑一个实质性的初步实验

```bash
python hf_launch.py submit --profile pilot --variant grounded --seed 0 --flavor t4-small
```

| 配置 | 随机训练 episode | 环境交互步 | 梯度更新 | 每种物体泛化条件试次数 | Job 默认超时 |
|---|---:|---:|---:|---:|---:|
| smoke | 4 | 96 | 8 | 4 | 30 分钟 |
| pilot | 128 | 8,192 | 1,200 | 48 | 3 小时 |
| research | 1,024 | 65,536 | 16,000 | 240 | 8 小时 |

超时是运行上限，不是实际耗时承诺。模型很小，场景生成在 CPU 上，训练在 GPU 上。
T4 的 16 GB 显存足以作为该配置的起点；本地验证是 CPU，尚未替你验证实际 HF GPU 作业。
需要更快 CPU 场景生成时可以选择官方列表中的其他资源规格。

完整研究配置的单个条件：

```bash
python hf_launch.py submit --profile research --variant grounded --seed 0 --flavor t4-small
```

## 3. 消融与“什么情况下学会”

| `--variant` | 改变的条件 |
|---|---|
| `grounded` | 随机经历；养育者说的话与事件/物体有关，q=1 |
| `q0` | 语言内容独立随机采样，q=0；允许偶然匹配 |
| `q25` / `q50` | q=0.25 / 0.5 |
| `shuffle_audio` | 同一 minibatch 中跨轨迹打乱声音，保留每段声音内部时间顺序 |
| `no_audio` | 移除声音输入、跨模态损失和声音预测损失 |
| `no_action` | 去掉当前和前一步动作作为模型输入 |
| `no_world_memory` | 世界模型 GRU 每一步重置；视觉/语音配对的短时编码器仍保留 |
| `progress` | 根据观测到的身体预测误差下降学习探索倾向；仍有随机探索 |
| `delayed` | 养育者语音延迟 12 个控制步，即 1.5 秒 |

先打印矩阵，**不提交作业**：

```bash
python hf_sweep.py --profile pilot --seeds 0 1 2 3 4
```

确认自己要运行该矩阵后，执行：

```bash
python hf_sweep.py --profile pilot --seeds 0 1 2 3 4 --execute
```

默认是 8 条件 × 5 个种子 = **40 个分别计费的 Job，顺序执行，一次最多 1 个**。
终端需保持运行以提交后续作业；关闭终端不会停止已经提交的 Job。
任一 Job 失败后停止继续提交。不要一开始就运行整个 research 矩阵。
可以缩小条件列表：

```bash
python hf_sweep.py --profile pilot --variants grounded q0 shuffle_audio --seeds 0 1 2 --execute
```

## 4. 防重复、停止、续跑

重复同一条 `submit` 命令会返回已经记录的 Job，不会自动创建新 Job。
代码哈希、配置和运行编号被保存在本地 `.launch_records/`。
不要为了“刷新”删除这个目录。若服务器接受提交后响应丢失，启动器先查已有 Job；
无法判断时会停止，不会盲目重试。请保留整个项目目录。

```bash
python hf_launch.py cancel
```

作业失败/超时后，可在**相同代码与相同配置**下加 `--resume` 续跑。
续跑会创建一个新的计费 Job，从已上传的最近完整 block 继续；前一个 Job 必须已结束。

```bash
python hf_launch.py submit --profile pilot --variant grounded --seed 0 --flavor t4-small --resume
```

每个完整 block 保存模型、优化器、随机数状态、全部经历和评测。
硬中断可能丢失尚未上传的当前 block；模型不会假装该 block 已完成。
若 `status.json` 为 `time_budget_reached`，表示提前保存退出，尚未完成计划的全部训练。
Hugging Face 的任务状态 `COMPLETED` 本身不代表研究计划完成，要同时看这个文件。

## 5. 结果里有什么

```text
report.html                 可离线打开的报告
preview.mp4                 第一人称录像，包含合成语音与物理碰撞声
learning_curves.png/.csv    每个检查点的成长曲线
metrics/eval_*.json         配对、预测、控制、独立探针的原始指标
metrics/trials_*.json       每道题的选项分数、选择和实际控制结果
checkpoints/last.pt         含优化器和随机状态的续跑检查点
checkpoints/model_*.pt      各阶段模型快照
data/shards/block_*.zip     全部训练经历及独立标注
evaluation.zip             固定的独立评测材料
training_trace.jsonl       每次梯度更新的损失、采样 episode 和时间段
exposure.jsonl             各 episode 的物体/词语暴露与交互统计
data_manifest.json         原始经历压缩包的大小和 SHA-256
config.json                实际运行参数
environment.json           依赖、设备、代码版本
status.json                是否完整完成、最后完成的 block
```

每个训练 episode 解压后包含 `observations.npz`、连续 `audio.wav`、
`annotations/scene_and_language.json` 和 `annotations/frames.jsonl`。
详见 `DATA_SCHEMA.md`。训练数据不会因过了 replay 窗口而删除；只有训练采样使用滚动 replay。

评测包括：熟悉属性的新实例、未见颜色×形状组合、未见布局、未见音色、未见物理参数、联合变化。
另外有 still/moving/falling 三选一事件诊断。
“控制能力”是模型预测身体运动后执行动作到坐标目标；不能把它写成听语音完成抓取。

下载多个独立种子后汇总：

```bash
python analyze_runs.py outputs --output aggregate.csv
```

这个脚本按独立模型种子计算均值和 bootstrap 区间，不把同一条视频的帧当独立训练重复。
单种子不输出跨种子置信区间。能力“出现时间”只是预先指定阈值的描述性统计，
不等于证明发生了突现；成长曲线是主要结果。

## 6. 在 Linux/自己的 GPU 直接运行

推荐 Python 3.11；PyBullet 官方 Linux wheel 适用于 3.10/3.11。
安装适合设备的 PyTorch 2.6 或更新版本，再装其余依赖：

```bash
python -m pip install torch
python -m pip install -r requirements-runtime.txt
python -m devbaby.run --config configs/smoke.json --output outputs/local-smoke
python -m pip install pytest
python -m pytest -q
```

GPU 无需窗口或显示服务器；物理和图像使用 PyBullet DIRECT/TinyRenderer。
eSpeak NG 通过 `espeakng-loader` 提供，无需系统 apt 安装。

查看某个模型如何预测一段经历：

```bash
python -m devbaby.inspect \
  --checkpoint outputs/local-smoke/checkpoints/last.pt \
  --episode outputs/local-smoke/data/episodes/0000000 \
  --output outputs/inspection
```

输出当前画面/真实下一帧/预测下一帧的对比，以及内部状态 `.npz`。
只加载自己生成、可信来源的 PyTorch 检查点。

## 7. 实验的科学边界

请先阅读 `SCIENTIFIC_SCOPE.md` 与实际验证记录 `VALIDATION.md`。
所有视觉素材都由几何体生成；声音来自规则式语音和解析碰撞声音合成；没有读取真实儿童数据。
主模型没有 pretrained weights、字幕、语义标签或语言知识。
养育者使用的人类设计语言规则属于**环境**先验；身体控制接口、抓取反射和学习目标也属于人为先验。
目前约 1M 参数、64×64 视觉、有限词汇和短记忆，是研究起点，不是完整人类婴儿复刻。

官方接口参考（2026-10-23 核查）：
- https://huggingface.co/docs/huggingface_hub/en/guides/jobs
- https://pypi.org/project/pybullet/3.2.7/
- https://github.com/thewh1teagle/espeakng-loader
- https://github.com/espeak-ng/espeak-ng
