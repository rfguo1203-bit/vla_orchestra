# CaP-Agent0 Framework 阅读整理与本工程融合规划

本文档基于论文 **CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation** 全文与 appendix 阅读整理，目标不是复现 `CaP-Gym` / `CaP-Bench`，而是以 **工程复现 CaP-Agent0 思路** 为中心，总结其框架，并明确它与当前本工程 `VLM/VLA` 路线的结合方式。

当前工程的核心方向，与论文的差异可以先一句话讲清楚：

- 论文主线：让 code-agent 直接写机器人控制代码，代码调用 perception/control primitives 完成任务。
- 我们当前主线：让 VLM 做观察/推理，让 VLA 直接输出动作执行。
- 最终融合目标：做成一个统一 agent，在同一套任务循环里既能调用 VLA，也能在必要时切到 code-agent 写控制代码或调用 code-skill。

## 1. 先给结论

CaP-Agent0 不是一个单纯“换个 prompt 的 coding agent”，而是一套 **围绕 robot control 的 agent runtime**。它的核心不在单次代码生成，而在下面四件事同时成立：

1. 机器人环境被包装成一个可多轮交互的代码执行环境。
2. agent 不是直接吃原始图像做 codegen，而是引入了一个把视觉变化转成文本的 `VDM`。
3. agent 会把成功 rollout 里反复出现的低层工具逻辑沉淀成持久 skill library。
4. agent 不信任单次生成，而是并行采样多个候选，再综合成最终执行方案。

换句话说，CaP-Agent0 的本质是：

**一个带视觉文本化、执行反馈闭环、可持续 skill 沉淀、支持 test-time search 的 embodied coding runtime。**

这也是它最值得迁移到我们工程里的地方。

## 2. CaP-Agent0 的工程目标与定位

论文里 CaP-Agent0 是从 CaP-Bench 的失败模式倒推出来的，主要回应三个问题：

1. 低层 primitive 直接暴露给模型后，单次 code generation 很脆弱。
2. 原始 RGB 直接喂给 code model，对代码生成帮助不稳定，甚至会退化。
3. 很多失败不是“模型不会”，而是“单次 test-time exploration 不够”。

因此 CaP-Agent0 选择的路线不是继续堆训练数据，而是：

- 维持 low-level primitive 的表达力；
- 用多轮交互和执行反馈补足单次生成的不稳定；
- 用 `VDM` 把视觉 grounding 转成 text；
- 用自动 skill synthesis 恢复人类会手写的中层结构；
- 用并行候选与综合器增加 test-time compute。

这点对我们很重要：  
论文已经明确给出一个结论，**“高层抽象能提高成功率，但也会掩盖能力边界；低层接口更难，但更通用”**。  
这与我们后面想做的融合框架高度一致，因为 VLA 适合承担连续低层执行，而 code-agent 适合承担结构恢复、fallback、验证与策略重构。

## 3. CaP-Agent0 的运行时框架

从工程角度看，CaP-Agent0 可以拆成 8 个运行时部件。

### 3.1 Task Interface

输入是自然语言任务，例如：

- lift the red cube
- place the blue cube on top of the yellow cube
- stack these as high as you can

输出不是直接的 action token，而是 **一段可执行 Python 控制程序**，程序中调用机器人工具接口。

### 3.2 Primitive / Tool Layer

CaP-Agent0 的代码不是凭空控制机器人，而是写代码调用一组底层 API。论文中 S3 级别 primitive 主要包括：

- 观测：`get_observation`
- 分割：`segment_sam3_text_prompt`、`segment_sam3_point_prompt`
- 视觉指点：`point_prompt_molmo`
- 抓取规划：`plan_grasp`
- 3D 几何：`get_oriented_bounding_box_from_3d_points`
- 单臂控制：`solve_ik`、`move_to_joints`、`open_gripper`、`close_gripper`
- 双臂版本：`solve_ik_arm0/1`、`move_to_joints_both` 等

这里有一个非常关键的工程思想：

- 论文不是让大模型直接输出 joint trajectory。
- 而是让模型写代码，去 **组合 perception、geometry、IK、grasp planner、gripper control**。

这意味着 CaP-Agent0 的 agent 层，本质上是一个 **tool-composition planner + code synthesizer**。

### 3.3 Python Sandbox / Code Executor

每一轮 agent 生成的代码，会被放进 Python sandbox 执行。  
执行后返回的信息包括：

- `stdout`
- `stderr`
- 环境状态变化
- 下一轮观测

这使得 agent 可以做出典型 coding-agent 行为：

- 打印中间变量
- 检查 perception 结果
- 根据错误重写代码
- 在执行失败后换策略

论文里多轮设置 M1/M3/M4 的提升，本质上就是靠这个 **执行反馈闭环**。

### 3.4 Multi-turn Controller

CaP-Agent0 不是一轮生成完就结束，而是一个多轮循环：

1. 读取任务与当前上下文。
2. 生成候选程序。
3. 选出最终程序并执行。
4. 读取执行输出与环境变化。
5. 判断 `FINISH` 还是 `REGENERATE`。
6. 若失败，带着前序代码和反馈进入下一轮。

论文里还尝试了一个更强的“debug/verifier”式 prompt，要求模型逐行检查已执行代码、交叉验证 stdout/stderr、把缺失证据视作失败。这个版本没有稳定优于原版，但它说明了一件重要的事情：

**CaP-Agent0 的核心不是“再生成一次”，而是“把多轮视为验证-修复循环”。**

### 3.5 VDM: Visual Differencing Module

这是论文最值得迁移到我们工程里的部分之一。

论文发现：

- 直接把原始 RGB 图像插到多轮 codegen 里，效果不稳定，甚至比纯文本执行反馈更差。
- 更好的做法是单独引入一个 VLM，把视觉观测翻译成任务相关文本，再交给 coding agent。

VDM 在第一轮做：

- 初始场景描述
- 提取与任务相关的视觉属性

VDM 在后续轮次做：

- 描述上一轮到当前轮的视觉差异
- 指出任务是否看起来已完成
- 提供与策略修正相关的场景变化

因此 coding agent 实际消费的不是“图像本身”，而是：

- 任务描述
- VDM 的 scene summary / visual diff
- sandbox 执行反馈
- 已执行代码历史

这和我们当前工程已经在做的“关键帧 + memory summary + completion judgment”非常接近。差别只是：

- 我们现在的 VLM 主要做任务完成判定和轻量记忆；
- 论文里的 VDM 更明确地承担了 **策略相关观察描述器** 的角色，而不只是 stop 判定器。

### 3.6 Auto-Synthesized Persistent Skill Library

这是 CaP-Agent0 的第二个核心。

论文观察到：  
即使不给人类手写高层 API，强模型在成功代码里也会反复自己写出一些辅助函数，例如：

- 坐标/姿态变换
- mask 转点云
- 深度转 3D
- 选择 top-down grasp
- 轨迹插值
- 向量归一化

于是他们把 skill library 的形成流程做成一个自动 pipeline：

1. 收集成功 rollout，特别是 S3 成功代码。
2. 从成功代码中抽取函数定义。
3. 用 LLM 分析哪些函数是高频复用、任务无关、值得沉淀的。
4. 将这些函数提升为持久 skill。
5. 下一轮代码生成时，把 skill library 暴露给 agent 使用。

论文当前实现是单次 synthesis pass，但明确强调这个过程天然可以迭代：

- 新成功样本持续加入；
- skill 可继续 refine / merge / prune；
- 可调节筛选标准，让 skill 更 task-agnostic，或允许更任务定制的 skill。

论文列出的 9 个合成 skill 是：

1. `rotation_matrix_to_quaternion`
2. `decompose_transform`
3. `depth_to_point_cloud`
4. `mask_to_world_points`
5. `pixel_to_world_point`
6. `transform_points`
7. `interpolate_segment`
8. `normalize_vector`
9. `select_top_down_grasp`

这 9 个 skill 有一个非常鲜明的特点：

- 它们不是完整任务级 skill；
- 也不是人类预置的大而全宏动作；
- 而是 **高频、稳定、任务无关的几何/感知/操作 glue code**。

这对我们后面的“公共组件层”设计非常关键。

### 3.7 Parallel Reasoning / Candidate Synthesis

CaP-Agent0 不信任单次输出，而是每轮并行采样多个候选。

论文中的两种配置：

- 单模型：同一个模型采样 9 次，温度从 `0.1` 到 `0.9`
- 多模型：`Gemini-3-Pro`、`Claude Opus 4.5`、`GPT-5.2` 各采样 3 次，温度为 `0.1 / 0.5 / 0.9`

然后再有一个 central coding agent 负责：

- 比较这些候选
- 组合优点
- 当候选根本分歧时选更 robust 的方案
- 在后续轮次里，还需要综合 “REGENERATE” 和 “FINISH” 投票

这个 central synthesizer 的角色，本质上就是一个 **candidate judge + merger**。

论文的结论不是“ensemble 能让模型更聪明”，而是：

- 单模型多轮更像“事后修 bug”
- ensemble 更容易“预先想到 fallback”

这点很适合迁移到我们未来的 planner 层。

### 3.8 Human-in-the-loop Feedback

appendix A 里还展示了 real-world chat UI：

- 用户下发任务
- agent 显示 scene description、生成代码、执行细节
- 用户可以在轮与轮之间给反馈

例如抓苹果时，用户说 “grasped the apple too high”，agent 下一轮改代码重抓。

这说明 CaP-Agent0 的交互协议不是封闭的，它支持：

- model self-feedback
- visual feedback
- execution feedback
- human corrective feedback

## 4. 以工程复现为目标，CaP-Agent0 最小复现应包含什么

如果我们只复现 `CaP-Agent0` 的框架思想，而不复现 CaP-Gym/CaP-Bench，本质上至少需要下面 7 个模块。

### 4.1 Tool Runtime

需要一个统一 tool registry，至少能注册：

- perception tools
- geometry tools
- planner / solver tools
- execution tools
- utility skills

它要支持：

- docstring / schema 展示给 agent
- 执行权限控制
- 调用日志
- tool versioning

### 4.2 Code Execution Sandbox

需要一个受控 Python 执行器，支持：

- 执行 agent 生成代码
- 注入工具函数命名空间
- 采集 stdout/stderr
- 保存已执行代码历史
- 返回异常堆栈

### 4.3 Observation Grounding Layer

要有一个独立于 executor 的观察层，负责把环境观测转成 agent 可消费上下文。  
按论文经验，不推荐让 code model 直接面向原始像素做主要推理，因此这里最好分成：

- 原始图像 / 传感器采集层
- VDM 文本化层
- 可选结构化状态抽取层

### 4.4 Turn State / Memory Store

每一轮必须保存：

- task instruction
- VDM 输出
- executed code
- stdout/stderr
- decision state
- 成功/失败证据
- human feedback

这其实就是 CaP-Agent0 的多轮记忆骨架。

### 4.5 Skill Library Pipeline

至少要支持：

- 从成功代码里抽取函数
- 聚类/筛选候选 skill
- skill 去重与命名
- skill 元数据记录
- skill 注入下轮 prompt/runtime

### 4.6 Candidate Generation + Synthesis

需要支持：

- 多候选生成
- 候选打分或合成
- `FINISH / REGENERATE` 级别的继续/停止判定

### 4.7 Trace / Evaluation Infrastructure

没有 trace，就没有 skill synthesis，也无法知道多轮为何有效。至少要保存：

- 每轮输入上下文
- 候选代码
- 最终执行代码
- 工具调用结果
- 观测变化
- 任务结论

## 5. 哪些点可以直接复用进我们当前工程

这里重点回答问题 2.a。

虽然论文是“code-agent 写机器人代码”，而我们现在是“VLM 观察 + VLA 执行”，但下面这些点可以直接迁移，而且会对当前工程提供明确指引。

### 5.1 `VDM` 思路可以直接复用，而且和当前 memory 架构天然兼容

我们当前已经有：

- 关键帧触发
- `recent_history`
- `running_summary`
- 结构化 JSON 解析
- 基于 VLM 的任务完成判断

这已经非常像一个轻量版 `VDM memory loop`。  
可直接升级的方向是：

- 把当前 VLM 从“完成判定器”扩成“任务相关视觉变化描述器”
- 除了 `completed / in_progress / uncertain`，再输出更面向执行的观察字段
- 让它显式描述 “上一次到这一次发生了什么变化”
- 让它产出供 planner 使用的 textual diff，而不仅是 stop decision

对本工程的指引：

- 当前 `vlm_eval/vlm_memory.py` 已经是非常好的 VDM 雏形。
- 下一步不是盲目改 VLA，而是先把 observation summarization 做成更像论文里的 `VDM`。

### 5.2 Multi-turn 闭环可以直接迁移到 “VLA 失败恢复”

论文的多轮不是 code-agent 独有。它背后的通用思想是：

- 执行
- 观察
- 验证
- 修正
- 再执行

这个循环同样适用于 VLA。

在我们工程里，可以直接做成：

1. VLA 执行一段 action chunk。
2. VLM/VDM 在关键帧读取状态变化。
3. planner 判断：
   - 继续当前 VLA rollout
   - 重新发 instruction 给 VLA
   - 切换到恢复 skill
   - 进入 code-agent 模式

这意味着 CaP-Agent0 的最大可迁移资产之一，不是 code generation 本身，而是 **test-time recovery loop**。

### 5.3 Skill library 的“形成方式”可以直接复用

我们当前虽然不是写机器人控制代码，但依然会有 skill 的形成需求，只是 skill 形态会更多样：

- VLA callable skill
- perception skill
- verification skill
- recovery skill
- code utility skill

论文最值得复用的不是那 9 个函数，而是这个抽象：

- **不要一开始就人肉设计完整 skill ontology**
- 先从成功轨迹里找高频可复用模式
- 再把它们沉淀成 persistent skill

对本工程的直接指导是：

- skill 的来源应优先是成功 rollout，而不是拍脑袋定义；
- skill 管理要支持逐步沉淀，而不是一次性设计定型；
- skill 可以分层，不必都长成宏动作。

### 5.4 显式 verification 比“盲信执行器”更关键

论文一再强调：

- 强 agent 会主动验证目标是否真的完成；
- 没有视觉 grounding 时，它甚至会自己插入状态检查代码；
- 多轮提升很大一部分来自 verification，而不是单纯重试。

这对 VLA 系统尤其重要，因为 VLA 很容易：

- 动作看起来像对了
- 但实际状态没有完成
- 或者发生接触失败、对象滑落、遮挡等情况

对本工程的直接指导是：

- 不要把 VLA rollout 当成黑箱成功；
- 每个关键子目标都应有独立 verification；
- verification 最好和 planner 解耦，不依赖 VLA 自己“感觉完成”。

### 5.5 Parallel candidate planning 可以直接迁移到 planner 层

我们当前不一定需要“9 份控制代码候选”，但很适合做：

- 多个文本计划候选
- 多个恢复策略候选
- 多个 skill 组合候选
- 多模型 / 多 prompt 的任务拆解候选

也就是说，论文里的并行 reasoning 可以迁移成：

- 对 **高层 plan** 采样，而不是只对代码采样。

### 5.6 Human feedback channel 可以直接兼容现有方向

论文已经证明人类反馈在 embodied multi-turn loop 里是自然的。  
对我们而言同样成立：

- 用户能指出 “抓高了”“方向不对”“应该先移开遮挡物”
- planner 可以把这类反馈当成下一轮约束

这个能力对于早期系统非常重要，因为它能在没有完整自动恢复策略时，提供廉价 supervision。

## 6. 哪些点当前工程暂时不支持，但之后值得引入

这里回答问题 2.b。

### 6.1 直接写机器人控制代码的 code-execution runtime

这是论文与我们当前实现差异最大的地方。  
我们目前主要是：

- VLA 直接出动作
- VLM 负责判定/记忆

而不是：

- agent 写 Python 控制程序
- 程序调用 low-level robotics API

因此下面这些目前都还不在本工程里：

- 可执行机器人代码 sandbox
- 统一 primitive 注入命名空间
- 代码级 stdout/stderr 调试循环
- `FINISH / REGENERATE` 协议

但它们恰恰是未来要把两条路线融合时必须补齐的。

### 6.2 低层 primitive 作为一等接口

当前工程更接近：

- `task text + image -> VLA action`

而 CaP-Agent0 依赖的是：

- `task text + tool docs + observation -> Python code over primitives`

所以像下面这些接口层，目前工程里是缺的：

- `segment_*`
- `plan_grasp`
- `solve_ik`
- `move_to_joints`
- geometry utility API

这不是说现在就要补齐，而是说未来若想支持 code-agent 分支，必须建设一个统一 primitive layer。

### 6.3 从成功 rollout 自动抽取 code-skill 的 pipeline

我们当前还没有：

- 成功代码库
- 函数抽取
- skill 聚类
- skill 版本化
- skill 质量门控

这块短期内没法直接启用，因为还没有 code-agent runtime。但它是之后非常值得引入的能力。

### 6.4 外部 Python 生态直接进入机器人控制流程

论文 real-world appendix 展示了一个非常强的能力：  
agent 在任务里直接调用 SciPy、几何计算、RANSAC 等通用软件库，来解决机器人任务里的局部子问题。

这和 VLA 的强项完全不同。  
VLA 擅长直接动作生成，但不擅长显式调用外部算法工具。

因此：

- 这类“显式工具链能力”是当前工程暂不支持但未来应该接入的。

### 6.5 真正的 hybrid routing

当前工程还没有“什么时候用 VLA，什么时候改用 code-agent”的路由器。  
未来如果要融合，两条执行通道必须由统一 planner 管理。

## 7. 核心公共组件有哪些

这里回答问题 2.c。  
如果要把论文框架和我们工程统一起来，真正的公共组件不是“VLA”或“code-agent”本身，而是下面这些跨范式共享的层。

### 7.1 Observation-to-Context Layer

这是最核心的公共层。

职责：

- 接环境观测
- 做任务相关信息抽取
- 形成 textual / structured context
- 提供给 planner、VLA、code-agent、verifier

它应该统一承载：

- 当前工程的 contextual VLM memory
- 论文里的 VDM
- 未来的多视角/结构化状态抽取

### 7.2 Task State / Memory Store

应统一保存：

- task goal
- 当前阶段
- 最近观测变化
- 已确认完成的子目标
- 当前失败信号
- 执行模式
- 人类反馈

现在仓库里已经有最小版本：

- `recent_history`
- `running_summary`

未来可以自然扩成更完整的 task state。

### 7.3 Skill Registry

这是融合框架的中枢之一。

它不应该只管理一种 skill，而应该统一管理三类：

1. `perception skills`
2. `action skills`
3. `code skills`

每个 skill 至少要有：

- `name`
- `type`
- `inputs/outputs`
- `preconditions`
- `postconditions`
- `safety constraints`
- `implementation backend`
- `version`
- `source`

其中 `source` 很重要，最好区分：

- hand-written
- mined-from-rollout
- synthesized-by-agent
- learned policy wrapper

### 7.4 Skill Formation / Iteration Pipeline

这是 skill engineering 的核心，不是单个 registry 就够了。  
它至少要覆盖：

1. 发现：从成功轨迹里发现候选 skill
2. 抽象：把一次性操作提炼成可复用接口
3. 验证：离线或在线验证 skill 是否稳定
4. 收录：写入 registry
5. 观察：监控 skill 使用率、成功率、失败模式
6. 迭代：合并、替换、淘汰、重命名

论文给我们的最重要启发是：

- skill 工程应该是一个持续循环，不是文档式配置表。

### 7.5 Planner / Router

这是未来统一 agent 的真正“大脑”。  
它负责决定本轮应该：

- 继续调用 VLA
- 改写 VLA instruction
- 调用已有 skill
- 切换到 code-agent
- 让 code-agent 写新 skill
- 请求人类反馈

没有这一层，就只是两套系统并排摆着，不算融合。

### 7.6 Verifier / Recovery Layer

它是 planner 的搭档，负责：

- 判断是否完成
- 判断是否偏航
- 判断是否需要重试
- 决定 recovery 类型

这层可以共享当前工程里已有的 VLM completion logic，并逐步升级成任务级 verifier。

### 7.7 Unified Trace Store

如果没有统一 trace，后面几乎所有高级能力都没法做：

- skill synthesis
- error taxonomy
- planner training / tuning
- human feedback replay
- routing policy analysis

至少要能统一记录：

- observation trace
- planner decision trace
- VLA rollout trace
- code execution trace
- skill invocation trace
- verification trace

## 8. 我们与论文融合后的目标框架

最终目标不是“把 VLA 塞进 CaP-Agent0”，也不是“给 VLA 外面包个 code-agent”，而是做一个 **双执行通道统一 agent**。

建议把最终框架理解成 5 层。

### 8.1 Layer 1: World Interface

负责统一环境输入输出：

- RGB / depth / proprioception / state
- robot execution API
- simulator / real robot 兼容

### 8.2 Layer 2: Context Builder

负责把原始观测变成 agent 可消费上下文：

- VDM summary
- visual diff
- running memory
- task state

### 8.3 Layer 3: Planner / Router

根据任务阶段和当前状态，选择执行模式：

- `mode = vla`
- `mode = skill`
- `mode = code`
- `mode = hybrid`

这里的决策依据可以包括：

- 任务是否短闭环、接触密集
- 是否已有稳定 skill
- 当前是否出现失败恢复需求
- 当前观测是否需要显式几何推理或工具调用

### 8.4 Layer 4: Executors

这里并列两个主执行器。

#### A. VLA Executor

适合：

- 短时连续控制
- 视觉伺服
- 接触丰富但模板清晰的动作段

输入：

- instruction
- 当前观测
- 可选子目标/约束

输出：

- 动作 chunk
- rollout trace

#### B. Code Executor

适合：

- 需要显式感知工具链
- 需要几何/逻辑推理
- 需要 fallback 和验证
- 需要现场写辅助逻辑
- 需要把新的成功模式沉淀成 skill

输入：

- task
- tool docs
- VDM context
- history
- skill library

输出：

- Python code
- execution trace
- 新 skill 候选

### 8.5 Layer 5: Verifier and Skill Engine

统一负责：

- 判断任务/子任务是否完成
- 决定是否重试或切模式
- 从成功经验中提炼 skill

## 9. 一个更具体的融合运行循环

可以把未来系统的单轮循环写成下面这样：

1. 用户给出任务。
2. `Context Builder` 基于当前观测生成 task-relevant summary。
3. `Planner/Router` 选择执行模式。
4. 若适合连续控制，则调用 `VLA Executor`。
5. 若需要显式工具推理或恢复，则调用 `Code Executor`。
6. `Verifier` 检查任务是否推进、是否完成、是否失败。
7. `Memory Store` 记录本轮状态。
8. 若出现成功新模式，则交给 `Skill Engine` 形成候选 skill。
9. 下一轮 planner 再决定是否继续原模式或切换模式。

这套循环里，VLA 和 code-agent 不是竞争关系，而是：

- VLA 提供连续动作能力
- code-agent 提供结构恢复、显式验证、工具编排和策略重构能力

## 10. 对当前工程的具体落地建议

结合仓库现状，建议按三阶段推进。

### 阶段一：先把当前 VLM memory 升级成 VDM-style context layer

这一步最贴合当前代码基础，风险也最低。

建议做的事：

1. 让 `vlm_eval/vlm_memory.py` 不只输出完成判断，还输出更明确的 `scene/task diff`。
2. 把当前 `running_summary` 从“完成态摘要”扩成“策略相关世界状态摘要”。
3. 让关键帧 VLM 额外标记：
   - 当前子目标候选
   - 明显失败信号
   - 需要恢复的原因
4. 把 `memory_trace` 扩成 planner 可消费的状态轨迹。

这一步做完后，本工程就已经具备论文里 VDM 的雏形。

### 阶段二：引入 planner / verifier，而不是直接上 code-agent

在还没有 code sandbox 之前，可以先补统一 planner：

1. 定义 planner 输出：
   - `continue_vla`
   - `reissue_instruction`
   - `call_recovery_skill`
   - `escalate_to_code`
2. 明确 verifier 输出：
   - `completed`
   - `stalled`
   - `failed`
   - `uncertain`
3. 先把 skill 做成手写 registry，不急着自动 synthesis。

这样可以先把“统一 agent 的骨架”搭起来。

### 阶段三：再引入 code-agent 支路与 skill synthesis

这一步才真正接近论文框架融合。

建议新增：

1. 一个最小 Python sandbox
2. 一套受控 low-level / mid-level tool API
3. 从成功代码里抽函数的 pipeline
4. code-skill registry
5. planner 中的 `mode switch`

这时系统才能变成真正的 hybrid：

- 默认 VLA
- 失败时切 code-agent
- code-agent 成功后将经验沉淀成 skill
- 后续任务优先复用 skill 或重新交还 VLA

## 11. 对论文框架的最终判断

从我们工程目标出发，论文里最有价值的不是“让大模型直接写机器人控制代码”这个表面形式，而是下面三个更普适的结论：

1. **视觉信息不应未经处理地直接塞给控制/编码主 agent，而应先变成任务相关文本或结构化上下文。**
2. **执行反馈、多轮验证和恢复机制，本质上比单次 pass@1 生成更重要。**
3. **skill 不是先验写死的宏动作集合，而应该从成功经验里逐步沉淀出来。**

这三点都可以直接指导我们当前工程。

## 12. 一句话版融合方案

如果把本文档压成一句工程定义，我会这样描述未来目标：

**构建一个统一 embodied agent：以 VDM-style context layer 维护任务状态，以 planner/router 决定调用 VLA、已有 skill 或 code-agent，在多轮验证与恢复中逐步沉淀 persistent skill library，最终实现“既能直接控制机器人，也能按需写代码控制机器人”的混合系统。**

## 13. 对本仓库的直接启发

结合当前仓库已有内容，可以认为我们已经有了融合框架的第一个基础件：

- `simple_eval_libero10_pi05.py` 提供了实际 rollout loop
- `vlm_eval/vlm_memory.py` 提供了轻量任务记忆与结构化解析
- `memory_architecture.md` / `context_plan.md` 已经在往 “VLM 观察 + 记忆 + 判定” 方向推进

因此当前最合理的路线不是立刻模仿论文去写整套 code-agent runtime，而是：

1. 先把当前 VLM memory 升级为更完整的 `VDM-style task state layer`
2. 再加入 `planner/verifier`
3. 最后再引入 `code-agent + skill synthesis`

这条路线与论文结论一致，也与当前仓库状态最匹配。
