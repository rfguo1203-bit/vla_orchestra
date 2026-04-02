from __future__ import annotations

"""Observation normalization helpers."""

from typing import Any


def _standardize_env_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Match the observation schema produced by the standard eval pipeline.

    The official `eval_embodiment.sh` path goes through `EnvOutput.to_dict()`,
    which normalizes observations with `EnvOutput.prepare_observations()`.
    This helper mirrors that behavior so the standalone script can feed the
    model the same key set, including optional keys with `None` defaults.
    """
    return {
        "main_images": obs["main_images"] if "main_images" in obs else None,
        "wrist_images": obs["wrist_images"] if "wrist_images" in obs else None,
        "extra_view_images": (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        ),
        "states": obs["states"] if "states" in obs else None,
        "task_descriptions": (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        ),
    }
