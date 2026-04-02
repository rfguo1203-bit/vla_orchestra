# VLM 任务状态上下文记忆方案

## Summary
目标是在当前单帧 `terminate/reason` 判断链路上，升级出一个稳定的“任务状态记忆层”，让 VLM 在关键帧上基于 `主视角图像 + 任务描述 + 结构化历史记忆` 做三件事：

1. 提取当前帧的任务相关状态
2. 更新一份压缩且可累计的任务状态记忆
3. 输出当前任务是否完成

整体设计优先“稳定准确”，采用“结构化 JSON 记忆 + 关键帧调用 + 单次调用内完成观察/更新/判断”。输入保持为你当前设想的 `主图 + task text`，不把 wrist/state 作为首版依赖。

## Key Changes
### 1. 把当前单帧终止判断，改成“状态机式记忆更新”
在 `simple_eval_libero10_pi05.py` 里，把现有 `_query_vlm_termination()` 扩成一个更通用的 `query_vlm_task_state()` 流程，返回结构从：

- `{"terminate": bool, "reason": str}`

升级为稳定 schema，例如：

```json
{
  "frame_state": {
    "scene_observation": "当前可见事实",
    "robot_subtask": "机器人当前正在做什么",
    "progress_signals": ["已完成/未完成的关键迹象"],
    "uncertainties": ["看不清/无法确认的点"]
  },
  "task_memory": {
    "goal": "任务目标的规范化描述",
    "completed_subgoals": ["已确认完成"],
    "active_subgoal": "当前进行中子目标",
    "pending_subgoals": ["剩余子目标"],
    "state_summary": "面向下一帧的压缩记忆",
    "completion_confidence": 0.0
  },
  "decision": {
    "terminate": false,
    "status": "in_progress|completed|uncertain",
    "reason": "简短依据"
  }
}
```

要求：

- 所有字段语义固定，不允许自由改 key
- `frame_state` 只描述当前帧可见事实
- `task_memory` 是跨帧累计记忆
- `decision` 只基于前两者得出，避免直接跳结论

### 2. 引入“短记忆 + 长记忆”两层上下文，而不是把所有历史摘要直接拼接
不要把“之前每一帧的任务状态总结”全量塞给模型；这会快速漂移并污染判断。改为两层：

- `recent_history`
  保留最近 `K` 次关键帧的结构化摘要，建议 `K=3~5`
- `running_memory`
  保留一份累计压缩后的全局记忆，只含任务完成判断真正需要的信息

更新规则：

- 每次关键帧 VLM 输出后，先写入 `recent_history`
- 再由程序侧按固定规则更新 `running_memory`
- 当 `recent_history` 超长时，丢弃最旧项，不回灌全文历史

这样做的核心是让“长期信息”由程序维护结构，而不是让模型反复重写整段自然语言历史。

### 3. 把 prompt 设计成“角色说明 + 工作流 + 严格判定准则 + few-shot”
系统提示不要只说“判断任务是否完成”，而要明确四部分：

- 身份
  你是机器人任务进度裁判，不是控制器
- 输入定义
  你会看到任务描述、当前主视角图像、历史结构化记忆
- 工作流
  先观察当前帧，再更新记忆，再判断是否完成
- 判定准则
  只有当任务目标在图像中被明确满足时才能输出 `terminate=true`
  看不清、被遮挡、只接近成功、短暂接触、可能已完成都不能判完成

few-shot 示例建议至少覆盖：

- 明显未完成
- 接近完成但不能判完成
- 被遮挡或证据不足
- 明显已完成
- 中途发生状态回退，记忆需要修正

重点不是示例多，而是示例的字段格式、证据标准、保守策略要高度一致。

### 4. 用“任务模板化拆解”提升跨任务稳定性
对 `task_name` 不要直接原样塞进 prompt 后让模型自由理解。建议在程序侧先规范化成一个 `task_spec`，至少包含：

- `task_goal`
- `success_criteria`
- `common_false_positive_patterns`

首版可以基于 LIBERO task language 做轻量规则生成，不需要复杂手写本体。目的不是精确规划，而是把“成功条件”显式化，减少模型每帧重新解释任务。

例如：

- `put mug on plate`
  success_criteria: mug stably on plate
  false_positive_patterns: mug hovering above plate, mug touching plate edge, gripper still holding mug over plate

这个 `task_spec` 应进入 system/user prompt，成为记忆更新的参照系。

### 5. 增加程序侧防漂移约束，而不是完全相信模型写回的记忆
记忆更新要有程序约束：

- 只接受 schema 内字段
- 对 `completed_subgoals` 做去重
- 若新输出与旧记忆冲突，允许标记为 `uncertain`，不要盲目覆盖
- `terminate=true` 需要同时满足：
  - `decision.status == "completed"`
  - `completion_confidence` 高于阈值
  - `reason` 非空
- 可选加入二次确认：
  连续 `N` 次关键帧都判完成才真正 early stop，建议首版 `N=2`

这一步对稳定性非常关键，因为“单次误判完成”比“多跑几步”风险更大。

### 6. 日志与产物从“终止记录”升级为“记忆轨迹”
当前每个视频旁只存 `vlm_checks`。建议改成完整 `memory_trace`，每条记录包含：

- `step`
- `task_spec`
- `prompt_version`
- `recent_history_snapshot`
- `running_memory_before`
- `frame_state`
- `running_memory_after`
- `decision`
- `raw_text`
- `parse_ok`

这样后续能直接回看“为什么误判”“哪种任务描述漂移”“哪个 few-shot 有帮助”。

## Public Interfaces / CLI Changes
建议在现有 CLI 基础上新增或替换这些参数：

- `--vlm-memory-mode`
  `single_frame | contextual`
  默认切到 `contextual`
- `--vlm-keyframe-interval`
  取代或别名当前 `--vlm-check-interval`
- `--vlm-history-size`
  recent history 保留条数
- `--vlm-completion-threshold`
  完成置信度阈值
- `--vlm-completion-confirmations`
  连续多少次 completed 才 early stop
- `--vlm-prompt-version`
  方便实验不同 prompt/few-shot
- `--vlm-task-spec-mode`
  `raw | normalized`

JSON 输出接口要从 `vlm_checks` 升级为 `memory_trace`，同时 episode summary 里保留最终：

- `termination_source`
- `termination_reason`
- `final_task_memory`

## Test Plan
### 单元测试
- 响应解析
  能正确解析严格 JSON、带包裹文本的 JSON、缺字段响应、非法 JSON
- 记忆更新
  recent history 截断正确
  completed_subgoals 去重正确
  冲突状态不会无条件覆盖
- 终止门控
  单次 completed 但未达确认次数时不终止
  低置信度 completed 不终止

### 提示与状态机测试
- 同一任务的连续几帧中，记忆应保持术语一致，不应每帧改写目标表述
- “接近成功”场景应稳定输出 `in_progress`
- “证据不足/被遮挡”场景应输出 `uncertain` 而不是 completed
- 真完成后连续两次关键帧应成功 early stop

### 回放评估
基于现有视频输出，人工挑若干 LIBERO 任务片段做离线回放：

- 未完成片段误判完成率
- 已完成片段漏判率
- 记忆字段稳定性
- 不同 prompt version 的对比

首批重点覆盖：

- 需要精确放置的任务
- 容易被遮挡的任务
- “接近但未完成”高频任务

## Assumptions
- 首版只使用 `主视角图像 + task text + 程序维护的结构化记忆`
- 首版采用单次 VLM 调用完成“观察、记忆更新、完成判断”，不做两阶段链式调用
- 关键帧触发沿用当前固定步长机制，但语义升级为“关键帧状态更新”
- 当前仓库核心实现点就是 `simple_eval_libero10_pi05.py`，落地时主要改这一处
- 运行环境暂时缺 `torch`，所以这轮只给出决策完整的实现方案；真正编码前需要先确认本地依赖可运行
