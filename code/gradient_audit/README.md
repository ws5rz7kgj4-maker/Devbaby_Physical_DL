# Archived audit and patch / 归档审计与补丁

These three files are exact copies from `DevBaby_Parameter_Audit.zip`. They are
preserved to document the mechanism, not advertised as a completed repaired
training pipeline.

这三个文件原样复制自参数审计包，用于保存机制证据，不代表完整修复训练已验证。

| File | Purpose / 用途 |
|---|---|
| `apply_gradient_fix.py` | Exact one-time source patch with backup; no cloud submission / 带备份的一次性精确源码补丁，不提交云作业 |
| `run.py` | Historical CPU parameter audit; depends on original private HF artifacts / 历史 CPU 参数审计，依赖原私有 HF 文件 |
| `gpu_cache_check.py` | Historical GPU cache-on/cache-off comparison, without GradScaler / 历史 GPU 缓存开关对照，没有 GradScaler |

The audit scripts read/write the original Hugging Face artifact locations when
executed in their configured environment. They are not part of the offline
documentation checks. No credentials are embedded. Use the lightweight
`scripts/validate_*.py` checks at the repository root to verify this snapshot.

审计脚本在原配置环境下会读取/写入 Hugging Face 结果位置，不属于离线文档检查。
文件内不含凭据。核对本包请使用根目录 `scripts/validate_*.py`。

For the original compatible source tree only:

```bash
python3 code/gradient_audit/apply_gradient_fix.py /absolute/path/to/original/project
```

Do not apply the patch to this documentation directory. It deliberately fails
when source patterns differ or a backup already exists. The old package named
`WorldModel_Fixed` still contains the cache bug; its name refers to an earlier
overflow-recovery change. See the bilingual training-bug reports for validation
limits and required new-run policy.

不要对本说明目录应用补丁。源码模式不匹配或已有备份时脚本会失败。历史
`WorldModel_Fixed` 名称指更早的 overflow 修复，该包仍含缓存缺陷。验证边界与
新运行要求见双语训练缺陷报告。
