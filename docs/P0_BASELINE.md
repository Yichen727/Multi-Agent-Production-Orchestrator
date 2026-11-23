# P0：可回退、可测量的工程基线

日期：2026-11-23。P0 建立基线，尚未实施 P1 数据库迁移、ID 修复、服务化或 UI 重组。

本文件记录 P0 时点；之后完成的数据库迁移与两项已知缺陷修复见 [P1A_CATALOGUE.md](P1A_CATALOGUE.md)。P0 的旧库快照、版本和测试数作为历史基线保留。

后续优先级澄清：用户所指的成本控制主要针对视频分析方法和工作量，模型费用暂不作为重点。P0 已有价格/费用记录及试验入口作为历史交付保留；后续优先推进缓存、解码/抽帧复用、自适应采样与按需事件分析，不继续扩展预算管理。当前实施要求以 PLAN.md 为准。

## 版本与本地备份

- P0 前源码提交：`c5e4f0c3ec89adf643dfd9aea0af5950e7420575`。
- 首次已验证快照：`.local/p0/20261123T111707Z-da4ccd44/`。
- 包含 2 个项目、105 条素材、450 个事件、8 对 XML/JSON 导出。
- SQLite backup API 从只读连接取得包含已提交 WAL 内容的一致性快照，不直接复制活跃 `.db`。
- `manifest.json` 记录 25 个文件的 SHA-256；`schema.json` 记录表、列、外键、索引和行数。
- `source.zip` 使用源码白名单；不复制 `.env`、API 密钥、原始视频或整个用户环境。原始源码另由 Git 基线保存。
- `environment.json` 记录解释器、相关依赖、FFmpeg/ffprobe 与当时的 Git 提交。
- `.local/` 和运行日志不进入版本管理；本轮不推送本地备份。

在项目根目录创建/校验快照；自定义数据库显式传 `--database`：

```powershell
python -m scripts.baseline capture
python -m scripts.baseline verify .local/p0/20261123T111707Z-da4ccd44
```

恢复时先停 UI/worker，校验清单，在新工作目录用配套源码打开快照，检查行数、关联与导出，再切换正式目录。不能覆盖正在使用的数据库或混入旧 WAL/SHM。本轮没有还原或迁移生产库。

## 环境复现

验证环境：Windows AMD64，Python 3.12.4。

| 组件 | 版本 |
| --- | --- |
| LangGraph / langchain-openai | 1.2.5 / 1.3.2 |
| OpenAI SDK / Streamlit | 2.41.1 / 1.59.2 |
| Pydantic / SQLAlchemy / pytest | 2.13.4 / 2.0.30 / 9.1.1 |
| FFmpeg / ffprobe | 2024-09-26，git-f43916e217 |

`requirements-p0.lock.txt` 固定本机运行/测试依赖闭包，共 97 项，没有收录无关 Conda 包与本机构建 URL。它是 Windows/Python 3.12 的版本快照，不是跨平台或带分发文件哈希的通用锁。现有 requirements.txt 不变，没有升级依赖。

建议在独立环境复现，安装操作由使用者按需执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-p0.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

FFmpeg 仍需配置系统可执行文件。本轮未下载或安装依赖。当前机器的 `python` 命中不可执行的 WindowsApps 别名，本轮实际使用：

```powershell
& C:/Users/86153/anaconda3/python.exe -B -m pytest -q -p no:cacheprovider
```

测试临时文件位于 `.local/pytest/<随机目录>/`，避免不同 Windows 用户的临时目录权限冲突。

## 回归检查与缺陷

测试在导入配置前禁用 dotenv，使用虚拟 key、内存数据库、临时输出与网络阻断；不解码或改写真实素材。真实 LangChain 回调路径使用内存 HTTP transport 验证。

覆盖：项目隔离、ID/路径歧义、事件归属、整片长度与顺序、事件边界/保护与压缩下限、非法范围、整数 FPS 时间映射、音轨链接、缺失媒体、模式约束、工具计划提取、未变文件跳过探测/模型、WAL 备份、日志和预算。

本轮最终自动回归：35 项通过、5 项严格预期失败；JUnit 报告为 `.local/p0-test-results.xml`。生产数据库逐行逻辑校验与初始快照一致。

五项严格预期失败不是已修复功能：

| 缺陷 | 后续阶段 |
| --- | --- |
| 同项目未勾选素材可进入规划 | P1b 候选白名单 |
| 恰好 200 文件错误超限 | P1a 扫描 |
| 重复写入项目重分配素材/事件 ID | P1a 增量入库 |
| 任意小数 FPS 被标为 NTSC | P5 导出 |
| NTSC 秒数换帧使用 nominal FPS | P5 导出 |

配置 `xfail_strict=true` 且只允许 AssertionError；修复后 XPASS 会使测试失败，提醒移除标记，其他环境/权限异常不能算预期失败。保留现有 langchain-community 弃用警告，P0 不替换依赖。

## 固定样本与质量标注

- 快照 `samples.json` 固定 26 条素材，种子 20261123，覆盖两项目的时长、场景数、事件数、方向和 FPS 差异。
- 清单包含路径、旧 ID 与指纹；迁移后需结合来源验证，不只按数字匹配。
- `benchmarks/quality_cases.json` 固定 18 条检索输入、8 个剪辑意图。它们是任务输入，不声称素材一定含有这些内容。
- `relevant_paths: null` 表示待人工标注，不等于没有相关素材；`review: null` 也不是质量评分。
- 编辑师观看后填写相关素材路径；剪辑方案记录模型/版本、候选、意图符合度、遗漏、动作错误、人工修正分钟数、是否采用及评论。标签与素材版本共同固定，不随模型结果改写。
- 人工质量、对白剪辑、完整入库与模型规划尚未评测。

## 离线性能基准

```powershell
python -m scripts.benchmark .local/p0/20261123T111707Z-da4ccd44 --repeats 5
```

结果为 `.local/p0-offline-results.json`。先校验备份，在临时数据库副本上运行，禁用 dotenv、网络、LangSmith 和 embedding。检索直接使用固定结构化条件；分别记录进程内首次调用和五次后续调用，不代表 OS 冷缓存或自然语言端到端延迟。

首次记录：18 条词法/事件检索，各用例热运行 p50 约 0.19–29.99 ms，p95 约 0.20–30.45 ms；项目 1 的 12 条整片/40 个事件时间线加 XML 编译 p50 约 8.37/16.78 ms；项目 2 约 6.03/13.89 ms。8 对历史导出可以解析。模型请求 0 次、费用 0 USD。

五次重复只用于同机对照，不作为生产 p95 保证。不包含模型、embedding、FFmpeg、媒体验证或 Premiere 导入。人工标签未填时 Recall@20 为 null；能解析历史导出不表示帧率与最终画面正确。

## 实际运行观测

`.env` 配置 `METRICS_ENABLED`，默认 true；`METRICS_OUTPUT_DIR` 默认 `app/data/output/metrics/`。JSONL 按进程拆文件，线程写入加锁，写失败不改变业务返回。

- 阶段：run_id、project_id、耗时、完成/异常状态与可获得的业务 outcome。
- 操作：ffprobe、场景检测、抽帧、代理、整片/事件视觉分析耗时。
- 模型：调用 ID、型号、输入/缓存/输出/可获取的推理 token、耗时、状态与版本化估算费用。
- Embedding：耗时、输入条数；embed_documents 不暴露的真实 token/费用保留 null。
- SDK 内部重试次数不可观测，保留 null；一个模型回调不能视为一个确定的 HTTP 请求。
- 不写 key、提示词、输出正文、帧、完整异常文本。

价格代码只登记已核对的 GPT-5.5 与 GPT-5.4 mini 标准文本费率和已知快照，依据：[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5)、[GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)。未知型号或 usage 不全时费用为 null。推理 token 已包含在输出 token，不二次计费；缓存明细缺失时保守按未缓存计。特殊处理、长上下文、税费不在简化价格表内，估算不等于账单。

## 有界的付费评测入口

P0 提供文本查询解析试验，不上传视频或全库重算。默认 dry-run，不读取真实 key、不创建 API 客户端：

```powershell
python -m scripts.evaluate --budget-usd 0.05 --max-requests 2
```

需要付费时由使用者明确添加 `--live`：

```powershell
python -m scripts.evaluate --live --model gpt-5.4-mini --budget-usd 0.05 --max-requests 2
```

默认最多一次请求、输出上限 512 token，最大可配置 2048；SDK 重试关闭。每次按文本字节/schema 开销保守预留，返回后按真实 usage 结算；余额不足不提交下一次。未知收费或网络错误保留预留并停止。应用预算不等于供应商硬封顶，已提交请求仍可能收费。

结果 `.local/query-trial.json` 记录预算、逐例状态、结算/预留。本轮仅做 dry-run：两条请求预留示例约 0.01212 USD，实际请求 0 次、费用 0 USD。视觉与规划 A/B 留到相应阶段，在限定样本和预算下开展。

P0 完成的是恢复材料、环境、自动回归、缺陷清单、任务/标注方法和受控评测入口。真实模型性能、人工质量、解码吞吐和 Premiere 导入仍待后续验证。
