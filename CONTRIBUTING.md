# Reviewing and committing this snapshot / 检查与提交

This directory can be committed as a new evidence repository or copied into an existing project. It has no configured remote and does not publish itself. No new license is selected here; confirm project licensing separately if you want an open-source release.

本目录可作为独立证据仓库提交，也可复制到现有项目。没有预设远程仓库，不会自动发布。本包没有代选许可证；如需开源发布，应另行确定项目许可。

## Local checks / 本地检查

```bash
python3 scripts/validate_results.py
python3 scripts/validate_early_stats.py
python3 scripts/validate_repository.py
```

The checks need only Python 3.10+ and do not launch training. They verify evidence arithmetic, archived-input hashes, generated tables, local Markdown links and package structure. They are not full training-path tests. Figure regeneration is described in the README.

检查仅需 Python 3.10+，不会启动训练。检查内容为证据计算、来源摘要、生成表格、Markdown 本地链接和包结构，不是完整训练路径测试。重建图表见首页。

## Suggested commit / 建议提交

For a new local repository, from the unpacked directory:

在解压目录中，新建本地仓库可执行：

```bash
git init
git add .
git diff --cached --stat
git commit -m "docs: archive bilingual DevBaby results and training audits"
```

For an existing repository, copy these files into the intended directory first, then review and commit there. The commands above deliberately do not select a GitHub account, create a remote repository or push. Publishing requires your actual destination URL and your normal Git authentication.

现有仓库应先将文件放入合适目录，再检查并提交。上述命令没有选择 GitHub 账号、创建远程仓库或推送；发布时使用你的实际仓库地址和已有 Git 身份验证。

## Suggested pull-request description / 建议 PR 描述

```text
Archive DevBaby's completed early physical-control/adaptation results and later
world-model diagnostics in Chinese and English. Distinguish the early controller
from the compromised world-model training run, preserve the AMP-cache failure
evidence and limit the repair claim to the verified single-step mechanism.

Include source-linked numerical evidence, reproducible tables/figures, protocol
and run provenance, explicit limitations and the pending validation sequence.

Validation: evidence/table checks, early contrast/sign-flip/Holm recomputation,
local-link and checksum validation. No new training or cloud jobs.
```

```text
以中英文归档 DevBaby 已完成的早期物理控制/适应结果与后期世界模型诊断。
区分早期控制器和受缺陷影响的世界模型，保留 AMP 缓存故障证据，将修复结论
限定为已验证的单步机制。

包含可追溯数值、可重建表格/图、协议和运行身份、局限与后续验证顺序。

验证：证据与表格校验、早期对比/符号翻转/Holm 复算、本地链接与哈希校验。
没有新增训练或云作业。
```

## Evidence changes / 后续修改

- Preserve old raw files and run identities. Add new results under a new run rather than silently replacing failure evidence.
- Keep English and Chinese claims aligned. Label development, sealed test, historical diagnostic and planned work distinctly.
- Never report frames, reused scenes or cached trial rows as independent training seeds.
- After changing numeric inputs, regenerate tables and figures, document why inputs changed, and deliberately update input hashes. Do not update hashes merely to make a failed check pass.
- `provenance/content_sha256.json` is a release integrity manifest. Update it only for a reviewed new evidence snapshot; figure rendering is excluded because renderer metadata can vary.

- 保留旧原始文件和运行身份；新结果使用新运行记录，不静默覆盖故障证据。
- 中英文结论同步；区分开发、封存测试、历史诊断和计划。
- 帧、复用场景和缓存计分行不能当作独立训练种子。
- 数值输入改变后重建表格和图，记录改变原因，再审阅更新来源哈希；不能只为通过检查而改哈希。
- `provenance/content_sha256.json` 是发布完整性清单，仅在审阅新快照时更新；图的渲染字节因环境可能变化而不列入该清单。
