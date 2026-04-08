# Generalized VLM Context Architecture TODO

目标：把当前“单帧完成判定 + 自由文本记忆”的方案，升级为“首帧任务建模 + 关键帧结构化观察 + 差异更新 progress state”的通用方案，使同一套链路可以适配任意 task，而不是绑定抓取放置类任务。

## Summary

新的 context 架构分为两个阶段：

1. `Bootstrap`
   基于 `task_description + first_frame_image` 生成一份冻结的 `task_model`，并初始化首帧状态。

2. `Keyframe Update`
   每个关键帧基于 `task_model + previous_frame_state + previous_progress_state + current_image`：
   - 生成当前帧的结构化观察 `current_frame_state`
   - 生成与上一关键帧的差异 `frame_delta`
   - 更新跨帧任务进展 `progress_state`
   - 输出当前完成判断 `decision`

核心原则：

- 不预设 task 一定是抓取、放置、开关或按钮任务
- 不预设目标对象名、目标容器名或固定字段
- 不再使用自由文本 `running_summary` 作为主记忆
- 每次关键帧必须先分析“变化”，再判断“是否完成”
- 长期状态由程序维护，模型只输出结构化观察和增量判断

## 必做

### 1. 定义通用 `task_model` schema
把当前直接使用 `task_name` 的方式，升级为首帧生成并冻结的 `task_model`。

建议结构：

```json
{
  "task_summary": "normalized short description",
  "task_type": "placement|rearrangement|opening|closing|stacking|pressing|turning|unknown",
  "entities": [
    {
      "id": "entity_1",
      "role": "primary_object|reference_object|target_region|tool|unknown",
      "name": "task-relevant entity",
      "visual_cues": ["shape/color/text/relative position if known"]
    }
  ],
  "success_conditions": [
    {
      "id": "cond_1",
      "description": "condition that must be true in the image",
      "evidence_type": "spatial_relation|contact_relation|containment|state_change|visibility"
    }
  ],
  "failure_prone_patterns": [
    "near but not completed",
    "held by gripper but not released",
    "critical occlusion"
  ],
  "required_observation_slots": [
    "entity_presence",
    "entity_location",
    "entity_relation",
    "robot_interaction",
    "task_relevant_state_change"
  ]
}
```

要求：

- `task_model` 只在 bootstrap 阶段生成一次
- episode 内默认冻结，不反复重写
- 允许实体名不完全精确，但必须保持稳定的 `id`
- 不允许在 schema 里写死 `target_container`、`in_basket` 等 task-specific 字段

### 2. 定义通用 `frame_state` schema
用任务无关的结构化图像描述，替代当前 `frame_state.summary` 自由文本。

建议结构：

```json
{
  "entities": [
    {
      "id": "entity_1",
      "visible": true,
      "location": "left|center|right|foreground|background|unknown",
      "pose_or_state": ["upright", "lying", "open", "closed", "pressed", "rotated", "unknown"],
      "held_by_robot": false,
      "spatial_relations": [
        {
          "other_entity_id": "entity_2",
          "relation": "inside|on_top_of|touching|aligned_with|left_of|right_of|near|far|unknown"
        }
      ]
    }
  ],
  "robot_state": {
    "gripper_status": "open|closed|holding|unknown",
    "held_entity_id": "entity_1|none|unknown",
    "end_effector_region": "left|center|right|upper|lower|unknown",
    "interaction_mode": "idle|approaching|grasping|moving|placing|releasing|manipulating|unknown"
  },
  "scene_state": {
    "task_relevant_changes_visible": true,
    "critical_occlusion": false,
    "visibility_quality": "good|partial|poor"
  }
}
```

要求：

- `frame_state` 只描述当前帧可见事实
- 不直接输出“任务已完成”这类结论
- 字段尽量枚举化，避免长自然语言
- 可扩展支持非抓取任务，如开关、按压、旋转、对齐

### 3. 定义通用 `frame_delta` schema
新增关键字段 `frame_delta`，强制模型回答上一关键帧到当前关键帧发生了什么任务相关变化。

建议结构：

```json
{
  "entity_changes": [
    {
      "entity_id": "entity_1",
      "changed_fields": ["location", "held_by_robot", "spatial_relations"],
      "summary": "entity moved from one relevant region/state to another"
    }
  ],
  "robot_changes": [
    {
      "field": "interaction_mode",
      "previous": "grasping",
      "current": "releasing"
    }
  ],
  "scene_change_level": "none|small|meaningful|major",
  "task_relevant_change": "short summary"
}
```

要求：

- 每个关键帧必须输出 `frame_delta`
- 即使没有明显变化，也必须显式输出 `scene_change_level="none"`
- 不允许跳过差异分析直接做完成判断

### 4. 定义通用 `progress_state` schema
把当前自然语言 `running_summary`，替换成任务无关的跨帧进展状态。

建议结构：

```json
{
  "condition_status": [
    {
      "condition_id": "cond_1",
      "status": "not_met|partially_met|likely_met|confirmed_met|uncertain",
      "evidence": "short evidence",
      "confidence": 0.72
    }
  ],
  "entity_tracking": [
    {
      "entity_id": "entity_1",
      "tracking_status": "stable|intermittent|lost",
      "latest_known_state": "short structured summary"
    }
  ],
  "overall_progress": {
    "stage": "not_started|in_progress|near_completion|completed|uncertain",
    "completion_confidence": 0.41,
    "blocking_factor": "occlusion|missing_entity|insufficient_evidence|none"
  }
}
```

要求：

- `progress_state` 表示“成功条件满足到什么程度”
- 不再绑定特定 task 语义
- `condition_status` 是完成判断的主依据
- `progress_state` 由程序维护，不允许模型自由整段重写

### 5. 升级 VLM 输出 schema
把当前最小 schema：

- `frame_state.summary`
- `task_memory.state_summary`
- `decision`

升级为：

```json
{
  "frame_state": {},
  "frame_delta": {},
  "progress_state_patch": {},
  "decision": {
    "terminate": false,
    "status": "in_progress|completed|uncertain",
    "reason": "short evidence-based reason"
  }
}
```

要求：

- bootstrap 阶段单独支持输出 `task_model`
- keyframe update 阶段输出 `frame_state + frame_delta + progress_state_patch + decision`
- 保持严格 JSON 和固定 key

### 6. 改造 prompt 为“两阶段工作流”
当前 prompt 只要求“总结当前状态 + 更新记忆 + 判断完成”。新的 prompt 需要分为两类：

#### 6.1 Bootstrap Prompt
输入：

- `task_description`
- `first_frame_image`

要求模型：

1. 规范化任务描述
2. 提取 task-relevant entities
3. 生成 `task_model`
4. 输出首帧 `frame_state`
5. 初始化 `progress_state`

#### 6.2 Keyframe Update Prompt
输入：

- `task_model`
- `previous_frame_state`
- `previous_progress_state`
- `current_image`

要求模型固定按顺序完成：

1. 识别当前帧相关实体与机器人状态，输出 `current_frame_state`
2. 对比上一关键帧，输出 `frame_delta`
3. 按 `task_model.success_conditions` 更新 `progress_state_patch`
4. 输出 `decision`

明确约束：

- 不允许跳过差异分析
- 不允许只复述历史
- 不允许自由改写任务目标
- 证据不足时必须输出 `uncertain` 或低置信度

### 7. 程序侧接管长期状态维护
当前实现里，`update_episode_memory()` 直接覆盖自然语言 `running_summary`。新方案中，程序需要维护：

- `task_model`
- `previous_frame_state`
- `progress_state`
- `consecutive_completed_count`

更新规则：

- `task_model` 仅 bootstrap 生成一次
- `previous_frame_state` 每关键帧更新为当前帧
- `progress_state` 基于旧状态和 `progress_state_patch` 合并
- `confirmed_met` 不因单帧看不见立即回退
- 新旧证据冲突时降级为 `uncertain`
- 缺乏明确视觉证据时，不升级到 `confirmed_met`

### 8. 替换现有 episode memory 结构
当前 memory:

- `recent_history: list[str]`
- `running_summary: str`

替换为：

```json
{
  "task_model": {},
  "previous_frame_state": {},
  "progress_state": {},
  "consecutive_completed_count": 0
}
```

要求：

- 不再把自然语言摘要作为主状态
- 如需调试，可额外保留轻量 textual trace，但不参与核心状态更新

### 9. 升级 early stop 规则
当前 early stop 只看一次 `decision.terminate == true`。新方案改为程序侧门控：

- 所有 `success_conditions` 均达到 `confirmed_met`
- `overall_progress.completion_confidence >= threshold`
- `decision.status == "completed"`
- 连续 `N=2` 个关键帧满足才真正 early stop

要求：

- 避免单帧误判完成
- 支持遮挡恢复后的稳健确认

### 10. 升级 trace / 日志结构
把当前 `memory_trace` 升级为结构化轨迹。

每条关键帧记录至少包含：

- `step`
- `task_model_snapshot`
- `previous_frame_state`
- `current_frame_state`
- `frame_delta`
- `progress_state_before`
- `progress_state_patch`
- `progress_state_after`
- `decision`
- `parse_ok`
- `raw_text`
- `error`

目标：

- 能直接回看“为什么没有检测到进度变化”
- 能分析模型是否在差异分析阶段失效
- 能分析 progress 合并规则是否合理

## 可选

### 11. 增加 task_model 轻量修正机制
如果 bootstrap 阶段没有识别清楚实体，后续关键帧可允许有限修正：

- 只允许补充 `visual_cues`
- 不允许更改 `success_conditions`
- 不允许变更实体 `id`

### 12. 支持多关键帧图像输入
如果单帧差异仍不足，可扩展 keyframe prompt 输入：

- `previous_keyframe_image`
- `current_keyframe_image`

优先级：

- 先实现结构化 `previous_frame_state`
- 如果仍不够，再引入双图输入

### 13. 增加 prompt version 管理
加入：

- `bootstrap_prompt_version`
- `keyframe_prompt_version`

方便做 prompt ablation 和回归分析。

## 暂不做

- 多轮链式调用
- 多模型 voting
- 复杂规则引擎
- 自动任务本体库
- 多视角联合输入
- 低维状态融合
- 离线回放评测工具
- 自动 few-shot 检索

## 验收标准

- 首帧能稳定生成结构化 `task_model`
- 后续关键帧不再依赖自由文本 `running_summary`
- 每个关键帧都能输出 `frame_state` 和 `frame_delta`
- `progress_state` 能体现条件满足度的变化，而不是重复静态描述
- 相邻关键帧状态无明显变化时，会显式输出 `scene_change_level="none"`
- 真完成时，连续关键帧能稳定收敛到 `completed`
- trace 中能清楚看到“观察 -> 差异 -> 进度 -> 决策”的完整链路

## 推荐实现顺序

1. 定义新的通用 schema：`task_model / frame_state / frame_delta / progress_state / decision`
2. 实现 bootstrap prompt 和解析逻辑
3. 实现 keyframe update prompt 和解析逻辑
4. 替换 episode memory 结构
5. 实现程序侧 `progress_state` merge 规则
6. 改 early stop 门控
7. 升级 `memory_trace`
8. 基于真实 trace 做 prompt 和 merge 规则迭代

## 三个实现阶段

### 阶段一：打通通用结构化链路
目标：先把当前“自由文本记忆”链路切到“结构化输入/输出”链路，保证系统能稳定产出可解析、可追踪的通用状态表示。

本阶段范围：

- 定义并落地新的通用 schema：
  - `task_model`
  - `frame_state`
  - `frame_delta`
  - `progress_state`
  - `decision`
- 实现 bootstrap prompt 和对应解析逻辑
- 实现 keyframe update prompt 和对应解析逻辑
- 替换现有 episode memory 结构：
  - 去掉 `recent_history`
  - 去掉 `running_summary`
  - 引入 `task_model`
  - 引入 `previous_frame_state`
  - 引入 `progress_state`
- 保持现有 `vlm_check_interval` 机制不变
- 保持现有单次 VLM 调用模式，不引入多轮链式调用

阶段验收：

- 首帧能生成稳定的 `task_model`
- 每个关键帧都能输出 `frame_state` 和 `frame_delta`
- 现有主循环可继续运行，不因 schema 升级而中断
- 不再依赖自由文本 `running_summary` 作为核心状态

### 阶段二：接管跨帧进度更新与完成门控
目标：让程序真正维护任务进展，而不是让模型每轮自由改写状态；同时把完成判定从“单帧判断”升级为“结构化条件门控”。

本阶段范围：

- 实现程序侧 `progress_state` merge 规则
- 引入 `condition_status` 驱动的进展维护机制
- 增加冲突降级逻辑：
  - 新旧状态冲突时降为 `uncertain`
  - 已确认条件不因单帧缺失直接回退
- 改造 early stop 规则：
  - 所有成功条件达到 `confirmed_met`
  - `completion_confidence` 达到阈值（不做）
- 升级 trace：
  - 记录 `progress_state_before`
  - 记录 `progress_state_patch`
  - 记录 `progress_state_after`
  - 记录完整的 `decision` 和 error/raw_text

阶段验收：

- `progress_state` 能真实体现任务推进，而不是重复静态描述
- 相邻关键帧无变化时，会稳定输出 `scene_change_level="none"`
- 出现中间推进、回退、遮挡时，状态更新符合预期
- 完成门控明显比当前单次 completed 更稳

### 阶段三：增强泛化性与可调试性
目标：在基础链路稳定后，提升跨任务泛化能力、调试效率和后续实验效率。

本阶段范围：

- 增加 `task_model` 的轻量修正机制：
  - 允许补充 `visual_cues`
  - 不允许改写 `success_conditions`
- 评估是否需要引入双图输入：
  - `previous_keyframe_image`
  - `current_keyframe_image`
- 增加 prompt version 管理：
  - `bootstrap_prompt_version`
  - `keyframe_prompt_version`
- 基于真实 trace 回放，持续迭代：
  - schema 细节
  - prompt 约束
  - progress merge 规则
  - completion threshold
- 视效果决定是否扩展：
  - 更多通用 `pose_or_state`
  - 更细粒度 `relation`
  - 更强的 task type 归纳

阶段验收：

- 同一套 context 方案能覆盖更多 task 类型，而不是只对少数任务有效
- trace 足够支持快速定位“观察失败 / 差异失败 / merge 失败 / 门控失败”
- prompt 和状态机参数能以版本化方式稳定迭代
