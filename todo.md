# VLM Context Memory MVP TODO

目标：以尽可能小的实现，在当前单帧 `terminate/reason` 判断链路上加入一个基础版上下文记忆能力，让 VLM 能基于当前图像和少量历史，更稳定地判断任务是否完成。

## 必做

### 1. 最小化返回 schema [已完成]
把 VLM 输出限制为固定 JSON，先只保留最必要字段：

```json
{
  "frame_state": {
    "summary": "当前帧与任务相关的简短状态总结"
  },
  "task_memory": {
    "state_summary": "面向下一次判断的压缩任务状态"
  },
  "decision": {
    "terminate": false,
    "status": "in_progress|completed|uncertain",
    "reason": "简短依据"
  }
}
```

要求：

- 不引入更多层级或复杂字段
- 程序只接受固定 key
- 缺字段或非法 JSON 时按失败处理，不更新记忆

### 2. 升级 VLM 调用接口
在 `simple_eval_libero10_pi05.py` 中，把当前单帧终止判断改成基础版上下文判断。

输入包含：

- 当前主视角图片
- task 描述
- `running_summary`
- `recent_history`

输出：

- 上面的最小 JSON schema

要求：

- 保持单次调用完成“看图 + 更新状态 + 判断完成”
- 继续使用当前 OpenAI-compatible 接口

### 3. 实现最小记忆容器 [已完成]
每个 episode 内维护：

- `recent_history: list[str]`
- `running_summary: str`

更新规则：

- 每次关键帧调用前，把当前记忆传给 VLM
- 调用成功后：
  - 将 `frame_state.summary` 追加到 `recent_history`
  - 将 `task_memory.state_summary` 覆盖到 `running_summary`
- `recent_history` 只保留最近 `K=3`

### 4. 使用轻量 prompt
prompt 只保留基础规则：

- 你是任务完成判定器，不是控制器
- 输入包括任务描述、当前图像、历史记忆
- 先总结当前状态，再更新任务状态，再判断是否完成
- 只有图像中能明确看出任务已经完成，才输出 `terminate=true`
- 看不清、遮挡、接近完成、疑似完成，都不能判 completed
- 只输出 strict JSON

不实现：

- few-shot
- 多轮链式推理
- 复杂规则模板

### 5. 保持现有关键帧机制
继续沿用当前 `vlm_check_interval`。

要求：

- 不改整体控制流
- 只在现有关键帧检查位置接入上下文记忆逻辑

### 6. 最小终止规则
基础版只要满足以下条件就 early stop：

- `decision.terminate == true`
- `decision.status == "completed"`
- `decision.reason` 非空

不实现：

- 连续两次 completed 才终止
- 额外投票机制
- completion confidence 阈值

### 7. 保存最小 memory trace
把当前 `vlm_checks` 升级为基础版 `memory_trace`，每条记录至少包含：

- `step`
- `running_summary_before`
- `recent_history_before`
- `frame_state`
- `task_memory`
- `decision`
- `raw_text`

目标：

- 出问题时能回看
- 不追求一步到位的完整实验日志体系

### 8. 加基础容错
如果 VLM 返回异常：

- 本次视为不终止
- 记录 `raw_text` 或错误信息
- 不更新 `running_summary`
- 不让整个 episode 因解析失败中断

## 可选

### 9. 轻量 task 文本规范化
不要做复杂 `task_spec`，只做很轻的包装，例如：

- `Task: ...`
- `Goal: determine whether this task is already completed in the current image`

目的：

- 减少模型误解原始 task 文本
- 不引入新的规则引擎

### 10. 增加一个 history size 参数
如有必要，再补一个简单参数：

- `--vlm-history-size`

默认值可直接设为 `3`。

如果不想增加参数，先写死也可以。

## 暂不做

- few-shot 示例
- 连续 N 次 completed 才终止
- completion confidence
- `completed_subgoals / pending_subgoals / active_subgoal`
- 自动生成复杂 `task_spec`
- 多视角输入
- 使用低维状态辅助判断
- 复杂冲突修正逻辑
- prompt version 管理
- 离线回放评测工具

## 验收标准

- 代码仍能按当前方式运行 episode
- 关键帧时会把历史记忆带入 VLM
- VLM 返回合法 JSON 时，记忆会持续更新
- VLM 返回异常时，流程不会崩
- 任务完成时仍可 early stop
- 输出结果中能看到基础版 `memory_trace`

## 推荐实现顺序

1. 定义最小 schema 和解析逻辑 [已完成]
2. 加 episode 内记忆容器 [已完成]
3. 改 prompt 和 VLM 请求体
4. 在关键帧接入记忆更新
5. 改日志落盘为 `memory_trace`
6. 补异常处理
