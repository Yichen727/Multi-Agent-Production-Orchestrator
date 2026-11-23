# P1b：模型、服务与引用契约

实施日期：2026-11-23。P1c 页面重组与剩余可见命名尚未实施。

## 已完成的行为

- 完整素材统一为 ClipMetadata、clip_id、clip_metadata、selected_clips；模型、状态、SQL 返回、解析、工具、提示词和新导出同步迁移。保留镜头分类 shot_type。
- 删除 Ingest/Search/Delivery 的 Agent 模块；只有 Selection 编译工具推理图。四个 orchestrator 入口及 UI 调用方式保留。
- Ingest 直接返回本次 IngestResult；Delivery 直接返回本次 DeliveryResult。返回结果带 run_id，与 orchestrator 的测量上下文一致，不再使用全局“最近结果”。
- Search 最多一次结构化 SearchQuery 解析，直接 hybrid_search；没有结果复述调用。空查询直接列素材；解析失败仅回退字面检索，不再请求模型。核心词、近义扩展、事件辅助召回和过滤仍由原检索算法处理。
- 项目必须为正整数，错误输入不会指向项目 1。完整素材和事件查询按项目过滤。
- Selection 的详情、事件、两种规划与备选材料解析均限制在勾选候选内；候选来源版本或有效事件集合在运行途中变化则拒绝。
- 注入的编辑模式、画幅与目标时长是权威输入；未设时长也具有约束力，模型不能自行设置。
- 已入库的在线目录启动即用。项目或编辑者切换清理旧选择和输出；重新入库/刷新只淘汰失效计划，无关素材追加保留有效计划。

## 计划与历史文件

新计划及导出中间 JSON 使用 schema_version=1 与 project_id；段落/素材条目携带 clip_id、source_revision，以及适用的 event_id/event_set_id。

历史读取只经 plan_contract：

```python
from pathlib import Path
from app.services.plan_contract import read_edit_plan, validate_edit_plan

original = Path("historical.json").read_text(encoding="utf-8")
plan = read_edit_plan(original, project_id=1)  # 兼容旧 shot_id 或 sequence/clips
verified = validate_edit_plan(plan, project_id=1)  # 无法证明来源时明确拒绝
```

读取会复制转换，核对项目、原 ID、真实路径和可获得的源时长；绝不把漂移的 ID 悄悄换成按路径找到的另一个素材。无来源版本的历史计划可以读取结构及身份，但不能假定当前来源就是旧来源，需要重新规划。

导出前进一步检查来源版本、磁盘指纹、在线状态、有效事件集合、有限且合法的裁切范围。Clip Assembly 必须为整片，Moment Assembly 必须在数据库里的真实事件范围内。校验失败在媒体探测和写文件前返回失败结果。

已经保存的新版导出 JSON 可重新读入并验证，保留顺序、裁切及来源/事件引用。历史文件从不原地覆盖。

## 200 文件边界与分页

默认仍是最多 200 个文件；恰好 200 可处理，超出上限先拒绝、后续目录不受影响。需要大目录时可显式分页：

```python
from app.services.ingest_service import ingest_footage

offset = 0
while True:
    result = ingest_footage("footage", project_id=1, batch_offset=offset)
    if result.status == "failure":
        raise RuntimeError(result.message)
    if result.next_offset is None:
        break
    offset = result.next_offset
```

每页最多 200 个文件，目录/文件排序确定；分页期间保持目录稳定。分页只能追加/更新，不能 synchronize。中断后可从头重试，未变文件复用现有 clip/event ID 和已完成分析。默认扫描中断、目录为空、超过上限都不会清空已有目录。

## 验收证据

- 完整离线回归：101 passed、2 xfailed；两项仍是 P5 的非 NTSC 小数帧率判定和 NTSC 秒到帧映射。
- 两种剪辑模式通过真实 Selection 工具图、白名单、规格、确定性导出及 JSON 回读；视觉、模型、embedding 与媒体探测使用 mock。
- Streamlit AppTest 验证已有目录启动解锁、切换项目和错误项目输入；并发服务结果使用独立 run_id。
- 201 个文件默认拒绝，显式分页为 200+1，重复分页不增加 ID/数量或重复分析。
- 实际数据库与 P1a 迁移后快照相比，projects、clips、clip_events、event_sets、analysis_steps、schema_migrations 的全部行数据摘要一致；仍为 105 clips、450 events、105 event_sets、420 analysis_steps。integrity_check=ok，外键错误为 0。
- 历史 8 组 JSON 均检出 ID/路径漂移，8 个原件的 SHA256 与内容不变；没有伪造可用的迁移结果。
- 18 个固定检索案例、4 个本地编排/编译案例和 8 组历史 XML/JSON 完成离线重放，网络调用为 0。

本地报告：`.local/p1b/test-results.xml`、`offline-results.json`、`catalogue-acceptance.json`、`historical-references.json`。这些本地数据不提交 Git。

未运行真实模型质量评测、真实视频全库分析或 Premiere 导入验收，不能将本轮测试当作这些指标。后台执行和视频分析方法优化属于 P2，剩余表单/四页导航属于 P1c，帧率与 NLE 验证属于 P5。
