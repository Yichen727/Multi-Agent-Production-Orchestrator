# P1c：工作页面、表单与候选状态

实施日期：2026-11-23。

## 页面与命名

主区域采用四个同级页面：Media Library、Find Footage、Edit Plan、Export；每次只执行当前页面的读取与渲染。侧栏只放项目、工作区设置、Footage、导入结果和候选计数。

- Import / Update 位于 Footage 下，报告 Reused / New / Changed / Failed 与分析更新。高级选项保留视觉/时间片段分析、显式重分析与内容校验，日常使用不需要选择模型。
- Media Library 分开显示完整素材目录与当前候选，提供内容/方向/分析状态筛选、每页 20 或 50 条、当前页批量选择、清除候选和跳转剪辑方案。
- Find Footage 仅在提交后搜索，结果分页复用，不因勾选、预览或切页重复请求。搜索与素材库共用候选。
- Edit Plan 的模式改为 Full Clips / Selected Moments，显示时长、画幅、时间线和解释。目标时长只在 Selected Moments 生效。
- Export 明确选择已保存的方案版本，再命名和编译。支持 XML/JSON 下载，以及打开经过来源校验的计划 JSON。
- 项目名作为普通入口；Project ID 与 Editor ID 放在 Workspace settings。移除 MSc 项目说明、阶段编号与 Locked 等旧提示。

## 状态与历史恢复

候选唯一来源为 candidate_clip_ids，不再以路径或页面 widget 的存在作为是否选中的依据。分页、筛选、隐藏页面触发 Streamlit widget 清理时，候选仍保留。清除当前页不会清除其他页的候选。

workspace_state 的 schema v2 负责一次性读取 P1b 的路径 checkbox 与旧字段，按当前项目的真实完整路径映射为 clip_id；旧计划与编辑输入也按明确字段映射恢复。新状态使用 clip_metadata、edit_plan、edit_report、export_result、export_report 和 ui_values。

同一浏览器会话按 (project_id, editor_id) 保存工作区快照，切换后恢复各自候选、草稿、素材目录与方案；不同浏览器会话完全独立。候选离线会移出候选集合，过期计划不能导出；历史方案版本保留以便识别和重新规划。Editor ID 不构成身份认证。

每次成功生成保存一个校验后的计划深拷贝，导出可选其中一个版本。方案版本保存在会话中，通过 Download plan JSON / Open plan 显式跨会话保存与打开；不声称已建持久化的团队方案库。

## 响应速度与边界

- UI 目录读取不加载 embedding 向量；不在每次 rerun 遍历媒体文件夹。
- 默认不抽帧。Show thumbnails 只加载当前页；Preview 只打开明确选择的源文件，其他页面不提前渲染全部预览。
- 页面切换不调用模型、重新导入、重新搜索或重新导出。
- UI 显式使用 P1b 分页接口，每批最多 200 个文件，追加/更新并复用原 ID。一次导入的多页结果汇总显示。
- 导入仍为同步操作，后台任务/取消/恢复属于 P2；人工调整顺序和裁切属于 P3。
- 模型、视频分析算法、数据库 schema 及已有素材/event 数值均不在本轮改变。

## 验证

Streamlit AppTest 与离线回归覆盖：

- 四页只渲染自身控件，切页零分析/模型/编译调用，默认零抽帧。
- 素材库 25 条分页、筛选、勾选与当前页全选/清除，候选跨页不丢失。
- 搜索与素材库双向同步候选，返回搜索页不重复搜索。
- 侧栏连续更新同一素材，clip/event ID 与探测/视觉调用数不增加。
- 生成两个方案版本，显式选择导出，返回导出页不再次编译。
- P1b 状态迁移、旧方案恢复、项目/编辑者往返切换和浏览器会话隔离。
- 按项目隔离素材文件夹，显式创建命名项目，无向量的素材目录查询。

完整离线测试结果：111 passed、2 xfailed；两项已知帧率缺陷仍按 P5 处理。真实数据库与 P1b 快照逐表全行摘要一致：105 clips、450 events、105 event_sets、420 analysis_steps，integrity_check=ok、外键错误为 0。

本地数据库核对报告位于 `.local/p1c/catalogue-acceptance.json`；本地测试报告位于 `.local/p1c/test-results.xml`。使用模拟模型与媒体分析；不代表真实模型质量或 Premiere 实测导入验收。
