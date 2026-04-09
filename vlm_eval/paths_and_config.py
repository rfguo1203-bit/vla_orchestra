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
你是机器人任务状态分析器，不是控制器。
你需要输出严格 JSON，但图像观察与任务进展字段必须使用自然语言摘要。
请严格遵循调用方给出的阶段说明（bootstrap 或 keyframe_update）和字段要求。
通用规则：
- 只能依据当前图像和输入上下文判断。
- 证据不足、存在遮挡、仅接近完成时，不得判定 completed。
- 只有当图像证据明确支持任务完成时，才允许 terminate=true。
- decision.status 只能是 in_progress/completed/uncertain。
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
