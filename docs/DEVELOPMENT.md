# Atomized Post Production Helper 开发约定

当前阶段：P1c。实施范围和后续顺序以 [PLAN.md](../PLAN.md) 为准。

公开品牌统一从 app/branding.py 读取；强调色为 Atomized Pink（#FF0069），
Streamlit 原生控件主题位于 .streamlit/config.toml。Preview 使用带状态的 popover，
只在窗口打开时渲染视频；不能把全部预览预先加载或重新放到页面底部。

## 职责

- 保留四个显式入口：`run_ingest/run_search/run_selection/run_delivery`；UI 经 orchestrator 调用。
- 只有 Selection 使用推理 Agent。Ingest 返回 `IngestResult`；Search 一次 `SearchQuery` 解析后直接检索；Delivery 返回 `DeliveryResult`，确定性编译。
- 不恢复已移除的 ingest/search/delivery Agent 壳、全局最近结果、LLM supervisor。
- 不因减少调用而改变检索单元：Search 返回完整素材，event 是命中依据；Selection 才决定整片或时间片段编排。
- 剪辑意图决定结构和顺序，没有固定叙事模板。Clip Assembly 保留整片；Moment Assembly 按真实事件边界压缩，不能由编译器自行删片、重排或创作。
- 输出画幅只做等比 fit，保留完整画面；不自动裁切、拉伸或重新构图。

## 契约与身份

- 完整素材统一使用 `clips/clip_id/ClipMetadata/clip_metadata/selected_clips`；`shot_type` 是镜头类型，应保留。
- 旧字段只允许在数据库迁移、`plan_contract.read_edit_plan` 和基线历史格式适配中出现。新保存的计划、JSON、样本清单使用 `clip_id`。
- 不通过删除重插、重置序列或重新编号解决重复入库。相同项目、规范路径与未变化来源复用 clip/event 的 ID 和分析。
- 更新同一路径的内容保留 clip_id，递增 source_revision；新有效事件集合不能使旧计划继续引用过期窗口。
- 计划与导出 JSON 使用 schema_version。读入复制转换，原件不变；ID、项目、原路径、来源版本和磁盘指纹均需验证。
- 历史文件缺少来源版本时，只能检查结构和身份，不能把当前版本写进去冒充已验证。要求重新规划。

## 范围与状态

- project_id 必须是正整数；无效输入报错，绝不回退项目 1。模型不能提供或覆盖注入的项目范围。
- 工具从状态读取候选白名单、模式、画幅与时长。相同项目的未勾选素材同样不能被访问或加入计划。
- Selection 开始时固定候选来源快照；运行途中来源版本或有效事件集合变更应拒绝该计划。
- UI 恢复已有在线目录，不依赖本次会话先运行 Ingest。项目/编辑者切换恢复各自会话工作区，不能串用候选、输入草稿、素材目录或输出。
- 导入后只淘汰失效引用；无关素材追加不应清空有效计划。
- 调用结果通过返回值传递，不能依赖进程级最近结果；run_id 与测量 ContextVar 对齐。

## 入库边界

- 默认最多 200 个文件，恰好 200 可处理；超过上限在写入前拒绝。
- 显式分页从 batch_offset=0 开始，依次跟随 next_offset，每页最多 200。目录在分页期间应保持稳定；中断可从头重试，幂等复用保证不重复分析。
- 分页不得同步标记缺失素材离线。同步只接受完整、成功扫描，默认始终追加/更新。
- 失败步骤可补齐，已有有效标签/向量不因失败而清空。数据库写入使用短事务与来源版本检查。
- 后台 worker、抽帧/解码缓存及分析粒度调整属于 P2；模型选择与视频分析工作量分开评估。

## 验证与配置

- 运行 `python -B -m pytest -q`；测试禁用真实 dotenv、数据库与网络。
- 至少覆盖：跨项目/候选拒绝、历史 ID 漂移、版本/指纹/事件范围、三次重复入库、分页、并发结果与 UI 项目切换。
- 集成测试可使用真实 Selection 工具图，但模型、视觉、embedding、媒体探测均用 mock；不能把它说成真实模型质量或 Premiere 导入验收。
- 不提交密钥、原始视频、数据库、备份和本地验收报告。配置统一位于 app/config.py，日志统一 get_logger。
- VisionTags/EventTags 的字段描述是模型输出协议；保留事实描述，不能编造时间码、对象、ID 或精度。



## P1c UI 与状态

- 四个同级页面按条件渲染：Media Library、Find Footage、Edit Plan、Export。不要用会同时执行所有页面内容的 tabs 实现隔离。
- 导入仅在侧栏 Footage → Import / Update 显式触发。不得在 rerun 中递归扫描视频目录。
- 候选唯一来源为 candidate_clip_ids；页面 checkbox 是视图，其清理、隐藏、分页不得清空候选。按项目和编辑者隔离，浏览器会话之间不共享。
- 页面 widget key 使用 library_/find_/plan_/export_ 命名；持久输入草稿保存在 ui_values，不能依赖隐藏页面仍保有 widget key。
- P1b 路径 checkbox、旧状态和表单字段仅在 workspace_state 的版本化迁移边界读取；新写入统一使用 clip_metadata/edit_plan/export_result 等名称。
- 素材库一次读取不含 embedding 的轻量元数据，默认每页 20 条。缩略图只按用户请求在当前页生成，视频只在显式 Preview 后加载。
- 方案版本保存已校验计划的深拷贝，不能由后续模型结果覆盖。Export 必须明确选择版本，写文件前再次验证来源；切页不得触发编译。
- 会话方案支持 JSON 下载和显式打开以跨会话保存，不声称已实现持久多用户方案库或后台任务。人工微调属于 P3。
