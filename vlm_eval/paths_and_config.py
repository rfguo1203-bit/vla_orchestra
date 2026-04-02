from __future__ import annotations

"""Paths, defaults, and Hydra config loading for the standalone eval script."""

import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "simple_eval_libero10_pi05.py"
WORKSPACE_ROOT = SCRIPT_PATH.parent


def resolve_rlinf_repo_root() -> Path:
    """Locate the RLinf repository for this standalone wrapper script."""
    candidate_roots = [
        Path(os.environ["RLINF_REPO_PATH"]).expanduser().resolve()
        for _ in [None]
        if os.environ.get("RLINF_REPO_PATH")
    ]
    candidate_roots.extend(
        [
            WORKSPACE_ROOT,
            WORKSPACE_ROOT.parent / "RLinf",
            WORKSPACE_ROOT / "RLinf",
        ]
    )

    for candidate in candidate_roots:
        if (candidate / "rlinf").is_dir() and (candidate / "examples" / "embodiment").is_dir():
            return candidate

    searched = ", ".join(str(path) for path in candidate_roots)
    raise FileNotFoundError(
        "Could not locate the RLinf repository. Set RLINF_REPO_PATH to the RLinf repo root. "
        f"Searched: {searched}"
    )


REPO_ROOT = resolve_rlinf_repo_root()
EMBODIED_PATH = REPO_ROOT / "examples" / "embodiment"
DEFAULT_CONFIG_NAME = "libero_10_ppo_openpi_pi05"
DEFAULT_VLM_PROMPT = """
你是一个机器人任务完成状态判定器，不是控制器。
你会收到：
- 任务描述，
- 当前主视角图像，
- 之前检查得到的一条简短运行状态摘要，
- 最近几次状态摘要列表。

你的任务：
1. 总结当前图像里与任务相关的状态，
2. 更新一条供下一次判断使用的简短任务记忆摘要，
3. 判断任务现在是否已经完成。

规则：
- 只能根据提供的图像和文本上下文进行判断。
- 只有当任务已经在当前图像中被明确完成时，才能输出 terminate=true。
- 如果图像存在歧义、被遮挡、只是部分完成、或者只是接近成功，都不能判定为 completed。
- 只能输出严格 JSON，且必须使用下面这个固定结构：
{
  "frame_state": {"summary": "..."},
  "task_memory": {"state_summary": "..."},
  "decision": {
    "terminate": true,
    "status": "completed",
    "reason": "..."
  }
}
- decision.status 只能是以下三者之一："in_progress"、"completed"、"uncertain"。
""".strip()
DEFAULT_VLM_HISTORY_SIZE = 3


def set_runtime_env() -> None:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    os.environ.setdefault("RLINF_REPO_PATH", str(REPO_ROOT))
    os.environ.setdefault("EMBODIED_PATH", str(EMBODIED_PATH))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def load_eval_cfg(config_name: str, overrides: list[str] | None = None):
    """Compose the embodied eval config through Hydra."""
    set_runtime_env()

    import hydra
    from omegaconf import OmegaConf, open_dict

    config_dir = str(EMBODIED_PATH / "config")
    with hydra.initialize_config_dir(
        config_dir=config_dir,
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

    with open_dict(cfg):
        cfg.runner.only_eval = True
    OmegaConf.resolve(cfg)
    return cfg
