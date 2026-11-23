# P0 命名与引用清单

这是 P0 时点记录的 P1 迁移范围。P1a 已完成物理表/字段及运行 SQL 的迁移，公开字段别名与 UI 命名仍待 P1b/P1c；见 [P1A_CATALOGUE.md](P1A_CATALOGUE.md)。原逐行引用见 `.local/p0/20261123T111707Z-da4ccd44/naming_references.json`；表/列/索引/外键见同目录 `schema.json`。后续实施应重新扫描，避免依赖过期行号。

## 数据库

| 位置 | 当前关系 | P1 动作 |
| --- | --- | --- |
| shots | shot_id 主键，project_id + file_path 唯一 | clips/clip_id，保持 ID 数值，增加规范路径和来源版本 |
| clip_events | shot_id → shots.shot_id，DELETE CASCADE | 父字段/FK 迁移，保留 event_id，增加事件集合 |
| 素材索引 | idx_shots_project/orientation/shot_type/duration | 更新名称/目标，保持查询覆盖 |
| 事件索引 | idx_events_shot 等 | 更新父字段索引，其余按新结构调整 |
| projects.clip_count | 全项目替换时维护 | 增量真实计数，不能使用 MAX(id) |
| 写入与迁移 | _SHOT_COLUMNS、_migrate_shot_columns、replace_project_shots | 版本化迁移与真正 UPSERT |

## 模型、状态与接口

- ShotMetadata → ClipMetadata；同步 `app/models/__init__.py` 和导入方。
- SearchCandidate、TimelineSegment、ClipEvent、TimelineEntry、ClipRecommendation 和导出中间表示的 shot_id → clip_id。
- ProductionState 的 shot_metadata → clip_metadata、selected_shots → selected_clips；同步编排器 `_base_state`。
- TransitionSuggestion.from_shot_id/to_shot_id、ProductionReport.total_shots 等旧契约先核对是否仍使用；有调用则迁移，无调用则另行清理。
- database_service 的 DDL/SQL/join/计数/索引，catalogue_resolver 的列与 ID 解析，retrieval_service 的素材/事件 SELECT 和结果键同步更新。
- get_catalogued_shots、replace_project_shots、get_shots_by_type、classify_shot_attributes、list_all_shots 及提示词改为对应 clip 语义；不为改名保留已取消的 ReAct 壳。
- Selection 的候选/事件工具、timeline planner、白名单与摘要；Delivery/编译器的 ID 解析、来源核验和 segment 字段同步。

## 表单、UI 与会话

| 当前入口 | 目标 |
| --- | --- |
| Run Ingest Analysis | 侧栏 Footage 下“导入 / 更新素材” |
| Media Pool / bin_shots | 素材库页面；工作区候选按稳定 clip_id 保存 |
| Search / Selection / Delivery | 搜索素材 / 剪辑方案 / 导出 |
| Clip Assembly / Moment Assembly | 整段编排 / 片段剪辑 |
| Editing mode/intent、Target duration、Sequence name | 标签、说明、空状态使用一致语言 |

同步核对 `bin_{path}`、`srch_tgl_{path}`、`select_all_bin`、`search_query`、`editing_mode_label`、`aspect_ratio`、`target_seconds`、`intent`、`seq_name`、`reveal_export` 与 callback。需要迁移 key 和数据读取，不能只替换标签。四页共享同一工作区选择，按项目与编辑会话隔离。

## 历史引用与保留术语

- 8 组历史导出 JSON 含 shot_id；早期重复导入使部分旧 ID 与当前库不一致。字段别名只能处理命名，不能修复漂移 ID；必须核对项目、路径与来源。
- 新写入统一版本化新契约，旧读取兼容集中在边界，历史 XML/JSON 原件保留，转换另存。
- 测试 fixture、断言、README、PLAN、工具示例与开发约定同步迁移；旧名允许出现在明确的兼容/迁移测试与历史说明中。
- **保留 shot_type**：它代表真实镜头类型/景别。场景/镜头边界术语不全局替换；clip_events 和 event_id 的名称也保留。
