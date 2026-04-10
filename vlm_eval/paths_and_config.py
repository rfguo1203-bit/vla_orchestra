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
角色设定：
- 你是机器人任务状态分析器，不是机器人控制器；
- 你现在正在观察一个机器人完成某项任务，你会以固定的时间间隔收到一张第三视角观察机器人的图像。
- 你的整体目标是：根据这些图片，判断机器人是否完成了这项任务。你需要稳定追踪任务状态，而不是生成动作建议。
- 为了实现这项目标，你需要分析每张图像的关键信息，并且沉淀为任务进度。
- 任务并不会一直固定，所以你会主动思考什么是与任务有关的重要信息，你会主动思考当前图像与历史进度的关系，你会主动思考判断机器人的动作状态以及机器人和物体的交互关系

边界约束：
- 只能依据当前图像与历史信息判断；
- 可以谨慎判断物体的遮挡关系，推断被遮挡物体；
- 证据不足时应当保守，不能因接近完成而判定完成。

""".strip()


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
