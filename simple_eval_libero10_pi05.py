#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run Pi0.5 inference on a single LIBERO-10 task and save local videos."""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import random
import re
import secrets
import ssl
import socket
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from itertools import accumulate
from pathlib import Path
import sys
from typing import Any

from tqdm.auto import tqdm


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parent


def _resolve_rlinf_repo_root() -> Path:
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


REPO_ROOT = _resolve_rlinf_repo_root()
EMBODIED_PATH = REPO_ROOT / "examples" / "embodiment"
DEFAULT_CONFIG_NAME = "libero_10_ppo_openpi_pi05"
DEFAULT_VLM_PROMPT = """
You are a robot task completion judge, not a controller.
You will receive:
- the task description,
- the current base camera image,
- a short running summary from earlier checks,
- a short list of recent history summaries.

Your job:
1. summarize the current image state relevant to the task,
2. update the task memory into one short state summary for the next check,
3. decide whether the task is already completed.

Rules:
- Judge only from the provided image and text context.
- Output terminate=true only when the task is clearly completed in the current image.
- If the image is ambiguous, partially complete, occluded, or only near success, do not mark completed.
- Reply with strict JSON only using this exact schema:
{
  "frame_state": {"summary": "..."},
  "task_memory": {"state_summary": "..."},
  "decision": {
    "terminate": true,
    "status": "completed",
    "reason": "..."
  }
}
- decision.status must be one of: "in_progress", "completed", "uncertain".
""".strip()
DEFAULT_VLM_HISTORY_SIZE = 3


def compute_num_save_videos(total_episodes: int, save_fraction: float) -> int:
    """Compute how many episodes should be exported as videos."""
    if total_episodes < 0:
        raise ValueError("total_episodes must be non-negative")
    clamped_fraction = min(max(save_fraction, 0.0), 1.0)
    return int(total_episodes * clamped_fraction)


def select_video_indices(total_episodes: int, num_save_videos: int) -> list[int]:
    """Choose evenly spaced episode indices for video export."""
    if total_episodes <= 0 or num_save_videos <= 0:
        return []
    if num_save_videos >= total_episodes:
        return list(range(total_episodes))
    if num_save_videos == 1:
        return [0]

    indices = [
        round(i * (total_episodes - 1) / (num_save_videos - 1))
        for i in range(num_save_videos)
    ]
    deduped: list[int] = []
    for idx in indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped


def build_task_reset_state_ids(
    cumsum_trial_id_bins: list[int], task_id: int
) -> list[int]:
    """Map a LIBERO task id to its contiguous global reset-state id range."""
    if task_id < 0 or task_id >= len(cumsum_trial_id_bins):
        raise ValueError(
            f"task_id must be in [0, {len(cumsum_trial_id_bins) - 1}], got {task_id}"
        )
    start = 0 if task_id == 0 else cumsum_trial_id_bins[task_id - 1]
    end = cumsum_trial_id_bins[task_id]
    return list(range(start, end))


def choose_reset_state_ids(
    task_reset_state_ids: list[int],
    num_episodes: int | None,
    shuffle: bool,
    seed: int,
) -> list[int]:
    """Pick which reset states to evaluate for the selected task."""
    selected_ids = list(task_reset_state_ids)
    if shuffle:
        random.Random(seed).shuffle(selected_ids)
    if num_episodes is not None:
        if num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")
        selected_ids = selected_ids[:num_episodes]
    return selected_ids


def _set_runtime_env() -> None:
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
    _set_runtime_env()

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


def load_libero10_metadata(task_suite_name: str = "libero_10") -> dict[str, Any]:
    """Load task descriptions and reset-state ranges from the LIBERO benchmark."""
    _set_runtime_env()

    from rlinf.envs.libero.utils import get_benchmark_overridden

    task_suite = get_benchmark_overridden(task_suite_name)()
    descriptions = [
        str(task_suite.get_task(task_id).language)
        for task_id in range(task_suite.get_num_tasks())
    ]
    trial_counts = [
        len(task_suite.get_task_init_states(task_id))
        for task_id in range(task_suite.get_num_tasks())
    ]
    cumsum_trial_id_bins = list(accumulate(trial_counts))
    return {
        "task_suite": task_suite,
        "task_descriptions": descriptions,
        "trial_counts": trial_counts,
        "cumsum_trial_id_bins": cumsum_trial_id_bins,
    }


def normalize_task_name(task_name: str) -> str:
    """Normalize common separators and whitespace for task matching."""
    normalized = re.sub(r"[_\-]+", " ", task_name.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def resolve_task_id(
    task_descriptions: list[str],
    task_id: int | None = None,
    task_name: str | None = None,
) -> int:
    """Resolve a LIBERO-10 task from either id or language description."""
    if task_id is not None:
        if task_id < 0 or task_id >= len(task_descriptions):
            raise ValueError(
                f"task_id must be in [0, {len(task_descriptions) - 1}], got {task_id}"
            )
        return task_id

    if not task_name:
        raise ValueError("Either task_id or task_name must be provided.")

    normalized_query = normalize_task_name(task_name)
    normalized_descriptions = [
        normalize_task_name(description) for description in task_descriptions
    ]

    for idx, normalized_description in enumerate(normalized_descriptions):
        if normalized_description == normalized_query:
            return idx

    substring_matches = [
        idx
        for idx, normalized_description in enumerate(normalized_descriptions)
        if normalized_query in normalized_description
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        raise ValueError(
            "Matched multiple tasks by substring. Please use a more specific task_name."
        )

    raise ValueError(f"Could not find a LIBERO-10 task matching: {task_name}")


def _to_bool(value: Any) -> bool:
    try:
        return bool(value.item())
    except AttributeError:
        return bool(value)


def _slugify_task_name(task_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in task_name).strip("_")


def _build_output_session_dir(video_base_dir: Path, task_id: int) -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    return video_base_dir / f"{date_str}_{task_id}"


def _predict_video_path(output_session_dir: Path, video_idx: int) -> Path:
    return output_session_dir / f"{video_idx}.mp4"


def _resolve_seed(seed: int | None) -> int:
    """Use the provided seed or generate a random one when absent."""
    return seed if seed is not None else secrets.randbelow(2**31)


def _get_next_video_index(output_session_dir: Path) -> int:
    """Return the next non-overlapping MP4 index for the target output directory."""
    if not output_session_dir.exists():
        return 0

    existing_indices: list[int] = []
    for mp4_path in output_session_dir.glob("*.mp4"):
        try:
            existing_indices.append(int(mp4_path.stem))
        except ValueError:
            continue
    return max(existing_indices, default=-1) + 1


def _finalize_output_layout(
    video_base_dir: Path,
    output_session_dir: Path,
    seed: int,
    task_dir: str,
) -> None:
    """Move videos from RLinf's seed-based layout into the requested flat layout."""
    legacy_task_dir = video_base_dir / f"seed_{seed}" / task_dir
    if not legacy_task_dir.exists():
        return

    output_session_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(legacy_task_dir.iterdir()):
        target_path = output_session_dir / source_path.name
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(source_path), str(target_path))

    shutil.rmtree(legacy_task_dir, ignore_errors=True)
    legacy_seed_dir = video_base_dir / f"seed_{seed}"
    if legacy_seed_dir.exists():
        try:
            legacy_seed_dir.rmdir()
        except OSError:
            pass


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


def _extract_base_image(obs: dict[str, Any]) -> Any:
    """Return the first environment's main camera image."""
    main_images = obs.get("main_images")
    if main_images is None:
        return None
    try:
        return main_images[0]
    except (IndexError, TypeError):
        return main_images


def _encode_image_to_data_url(image: Any) -> str:
    """Encode an observation image into a PNG data URL."""
    from PIL import Image

    image_array = image
    if hasattr(image_array, "detach"):
        image_array = image_array.detach().cpu().numpy()
    elif hasattr(image_array, "cpu") and hasattr(image_array, "numpy"):
        image_array = image_array.cpu().numpy()

    image_pil = Image.fromarray(image_array)
    image_buffer = io.BytesIO()
    image_pil.save(image_buffer, format="PNG")
    image_b64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def _extract_vlm_text_content(response_payload: dict[str, Any]) -> str:
    """Extract text content from an OpenAI-compatible VLM response."""
    content: Any = None

    if isinstance(response_payload.get("output"), list):
        text_parts: list[str] = []
        for output_item in response_payload["output"]:
            for content_item in output_item.get("content", []):
                if content_item.get("type") in {"output_text", "text"}:
                    text_parts.append(content_item.get("text", ""))
        if text_parts:
            content = "\n".join(part for part in text_parts if part)

    if content is None:
        choices = response_payload.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(part for part in text_parts if part)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("VLM response did not contain parsable text content.")

    return content.strip()


def _empty_vlm_task_state(
    raw_text: str = "",
    parse_ok: bool = False,
) -> dict[str, Any]:
    """Return the minimal contextual task-state schema with safe defaults."""
    return {
        "frame_state": {"summary": ""},
        "task_memory": {"state_summary": ""},
        "decision": {
            "terminate": False,
            "status": "uncertain",
            "reason": "",
        },
        "raw_text": raw_text,
        "parse_ok": parse_ok,
    }


def _parse_vlm_task_state(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Parse an OpenAI-compatible VLM response into the minimal task-state schema."""
    content = _extract_vlm_text_content(response_payload)
    json_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if json_match is None:
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)

    json_text = json_match.group(0)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)

    frame_state = parsed.get("frame_state")
    task_memory = parsed.get("task_memory")
    decision = parsed.get("decision")
    if not isinstance(frame_state, dict):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(task_memory, dict):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(decision, dict):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)

    frame_summary = frame_state.get("summary")
    memory_summary = task_memory.get("state_summary")
    terminate = decision.get("terminate")
    status = decision.get("status")
    reason = decision.get("reason")
    if not isinstance(frame_summary, str):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(memory_summary, str):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(terminate, bool):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)
    if status not in {"in_progress", "completed", "uncertain"}:
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(reason, str):
        return _empty_vlm_task_state(raw_text=content, parse_ok=False)

    task_state = _empty_vlm_task_state(raw_text=content, parse_ok=True)
    task_state["frame_state"]["summary"] = frame_summary.strip()
    task_state["task_memory"]["state_summary"] = memory_summary.strip()
    task_state["decision"]["terminate"] = terminate
    task_state["decision"]["status"] = status
    task_state["decision"]["reason"] = reason.strip()
    return task_state


def _init_episode_memory(history_size: int = DEFAULT_VLM_HISTORY_SIZE) -> dict[str, Any]:
    """Create the minimal per-episode memory container for contextual VLM state."""
    return {
        "recent_history": [],
        "running_summary": "",
        "history_size": max(1, int(history_size)),
    }


def _snapshot_episode_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the current episode memory state for logging/debugging."""
    return {
        "recent_history": list(memory.get("recent_history", [])),
        "running_summary": str(memory.get("running_summary", "")),
        "history_size": int(memory.get("history_size", DEFAULT_VLM_HISTORY_SIZE)),
    }


def _should_terminate_from_task_state(task_state: dict[str, Any]) -> bool:
    """Return whether a parsed task state should trigger early termination."""
    decision = task_state["decision"]
    return (
        decision["terminate"]
        and decision["status"] == "completed"
        and bool(decision["reason"])
    )


def _update_episode_memory(memory: dict[str, Any], task_state: dict[str, Any]) -> None:
    """Update episode memory from a parsed task-state response."""
    if not task_state.get("parse_ok", False):
        return

    frame_summary = task_state["frame_state"]["summary"]
    if frame_summary:
        recent_history = list(memory.get("recent_history", []))
        recent_history.append(frame_summary)
        history_size = max(1, int(memory.get("history_size", DEFAULT_VLM_HISTORY_SIZE)))
        memory["recent_history"] = recent_history[-history_size:]
    memory["running_summary"] = task_state["task_memory"]["state_summary"]


def _build_failed_task_state(error_message: str) -> dict[str, Any]:
    """Return a non-terminating task-state record for VLM call failures."""
    task_state = _empty_vlm_task_state(raw_text="", parse_ok=False)
    task_state["decision"]["reason"] = error_message.strip()
    return task_state


def _build_contextual_vlm_prompt(
    base_prompt: str,
    task_name: str,
    memory: dict[str, Any],
) -> str:
    """Build the lightweight contextual prompt for a keyframe VLM check."""
    recent_history = memory.get("recent_history", [])
    history_text = "\n".join(
        f"- {item}" for item in recent_history if isinstance(item, str) and item.strip()
    )
    if not history_text:
        history_text = "- none"

    running_summary = str(memory.get("running_summary", "")).strip() or "none"
    return (
        f"{base_prompt}\n\n"
        f"Task: {task_name}\n"
        "Goal: determine whether this task is already completed in the current image.\n"
        f"Running summary: {running_summary}\n"
        "Recent history:\n"
        f"{history_text}\n"
        "Judge based only on the provided base camera image."
    )


def _query_vlm_task_state(
    api_url: str,
    api_key: str | None,
    x_auth_token: str | None,
    model_name: str,
    prompt: str,
    image: Any,
    timeout: float,
) -> dict[str, Any]:
    """Call the local OpenAI-compatible VLM endpoint and return the parsed task state."""
    image_data_url = _encode_image_to_data_url(image)
    request_body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_bytes = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=request_bytes,
        headers=headers,
        method="POST",
    )
    if x_auth_token:
        request.add_header("x-auth-token", x_auth_token)
    last_timeout_error: TimeoutError | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
                context=ssl.create_default_context(),
            ) as response:
                response_bytes = response.read()
                response_text = response_bytes.decode("utf-8", errors="replace").strip()
                if not response_text:
                    raise RuntimeError(
                        "VLM response body is empty. "
                        f"content_type={response.headers.get('Content-Type')!r}"
                    )
                try:
                    response_payload = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    response_preview = response_text[:500]
                    raise RuntimeError(
                        "VLM response is not valid JSON. "
                        f"content_type={response.headers.get('Content-Type')!r}, "
                        f"body_preview={response_preview!r}"
                    ) from exc
                return _parse_vlm_task_state(response_payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"VLM request failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            is_timeout = isinstance(exc.reason, TimeoutError | socket.timeout)
            if is_timeout:
                last_timeout_error = TimeoutError(
                    f"VLM request timed out after {timeout}s on attempt {attempt}/3."
                )
                if attempt < 3:
                    continue
                raise last_timeout_error from exc
            raise RuntimeError(f"VLM request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            last_timeout_error = TimeoutError(
                f"VLM request timed out after {timeout}s on attempt {attempt}/3."
            )
            if attempt < 3:
                continue
            raise last_timeout_error from exc
        except socket.timeout as exc:
            last_timeout_error = TimeoutError(
                f"VLM request timed out after {timeout}s on attempt {attempt}/3."
            )
            if attempt < 3:
                continue
            raise last_timeout_error from exc

    if last_timeout_error is not None:
        raise last_timeout_error
    raise RuntimeError("VLM request failed for an unknown reason.")


def run_single_task_eval(
    task_id: int,
    config_name: str = DEFAULT_CONFIG_NAME,
    model_path: str | None = None,
    output_dir: str | None = None,
    num_episodes: int | None = 1,
    shuffle: bool = False,
    seed: int | None = None,
    save_fraction: float = 1.0,
    vlm_check_interval: int = 0,
    vlm_api_url: str | None = None,
    vlm_api_key: str | None = None,
    vlm_x_auth_token: str | None = None,
    vlm_model: str | None = None,
    vlm_prompt: str = DEFAULT_VLM_PROMPT,
    vlm_timeout: float = 30.0,
) -> dict[str, Any]:
    """Run a single-task LIBERO-10 evaluation loop without Ray workers."""
    from omegaconf import open_dict

    from rlinf.envs import get_env_cls
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.wrappers import RecordVideo
    from rlinf.models import get_model

    cfg = load_eval_cfg(config_name=config_name)
    metadata = load_libero10_metadata(task_suite_name=cfg.env.eval.task_suite_name)
    task_descriptions = metadata["task_descriptions"]
    if task_id < 0 or task_id >= len(task_descriptions):
        raise ValueError(
            f"task_id must be in [0, {len(task_descriptions) - 1}], got {task_id}"
        )

    resolved_seed = _resolve_seed(seed)
    task_name = task_descriptions[task_id]
    task_slug = _slugify_task_name(task_name)
    task_reset_state_ids = build_task_reset_state_ids(
        metadata["cumsum_trial_id_bins"],
        task_id=task_id,
    )
    chosen_reset_state_ids = choose_reset_state_ids(
        task_reset_state_ids=task_reset_state_ids,
        num_episodes=num_episodes,
        shuffle=shuffle,
        seed=resolved_seed,
    )
    if not chosen_reset_state_ids:
        raise ValueError("No reset states selected for evaluation.")

    num_save_videos = compute_num_save_videos(
        total_episodes=len(chosen_reset_state_ids),
        save_fraction=save_fraction,
    )
    save_video_indices = set(
        select_video_indices(
            total_episodes=len(chosen_reset_state_ids),
            num_save_videos=num_save_videos,
        )
    )

    video_base_dir = Path(output_dir or (REPO_ROOT / "results" / "libero10_pi05_single_task"))
    output_session_dir = _build_output_session_dir(video_base_dir, task_id)
    with open_dict(cfg):
        if model_path is not None:
            cfg.actor.model.model_path = model_path
            cfg.rollout.model.model_path = model_path
        cfg.env.eval.total_num_envs = 1
        cfg.env.eval.auto_reset = False
        cfg.env.eval.ignore_terminations = False
        cfg.env.eval.use_fixed_reset_state_ids = False
        cfg.env.eval.seed = resolved_seed
        cfg.env.eval.video_cfg.save_video = bool(save_video_indices)
        cfg.env.eval.video_cfg.video_base_dir = str(video_base_dir)

    env_cfg = copy.deepcopy(cfg.env.eval)
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    if env_cfg.video_cfg.save_video:
        env = RecordVideo(env, env_cfg.video_cfg)

    model_cfg = copy.deepcopy(cfg.actor.model)
    model = get_model(model_cfg)
    model.eval()

    episode_results: list[dict[str, Any]] = []
    saved_video_paths: list[str] = []
    next_video_index = _get_next_video_index(output_session_dir)
    vlm_enabled = vlm_check_interval > 0
    if vlm_enabled and (not vlm_api_url or not vlm_model):
        raise ValueError(
            "When vlm_check_interval > 0, vlm_api_url and vlm_model are required."
        )

    try:
        for episode_idx, reset_state_id in enumerate(chosen_reset_state_ids):
            env.is_start = False
            obs, _ = env.reset(reset_state_ids=[reset_state_id])
            obs = _standardize_env_obs(obs)

            episode_memory = _init_episode_memory()
            done = False
            success = False
            episode_steps = 0
            termination_source = "env"
            termination_reason = ""
            memory_trace: list[dict[str, Any]] = []
            progress_bar = tqdm(
                total=env_cfg.max_episode_steps,
                desc=f"Episode {episode_idx}",
                leave=True,
            )

            try:
                while not done and episode_steps < env_cfg.max_episode_steps:
                    raw_chunk_actions, _ = model.predict_action_batch(
                        env_obs=obs,
                        mode="eval",
                        compute_values=False,
                    )
                    chunk_actions = prepare_actions(
                        raw_chunk_actions=raw_chunk_actions,
                        env_type=env_cfg.env_type,
                        model_type=model_cfg.model_type,
                        num_action_chunks=model_cfg.num_action_chunks,
                        action_dim=model_cfg.action_dim,
                        policy=model_cfg.get("policy_setup", None),
                        wm_env_type=env_cfg.get("wm_env_type", None),
                    )

                    for chunk_step in range(chunk_actions.shape[1]):
                        action = chunk_actions[:, chunk_step]
                        obs, _reward, terminations, truncations, _infos = env.step(action)
                        obs = _standardize_env_obs(obs)
                        episode_steps += 1
                        progress_bar.update(1)

                        terminated = _to_bool(terminations[0])
                        truncated = _to_bool(truncations[0])
                        done = terminated or truncated
                        success = success or terminated
                        if terminated:
                            termination_source = "env"
                        elif truncated:
                            termination_source = "truncation"

                        should_check_vlm = (
                            vlm_enabled
                            and not done
                            and episode_steps > 0
                            and episode_steps % vlm_check_interval == 0
                        )
                        if should_check_vlm:
                            base_image = _extract_base_image(obs)
                            if base_image is None:
                                raise ValueError(
                                    "Contextual VLM check requires obs['main_images'] to exist."
                                )
                            memory_before = _snapshot_episode_memory(episode_memory)
                            task_prompt = _build_contextual_vlm_prompt(
                                base_prompt=vlm_prompt,
                                task_name=task_name,
                                memory=memory_before,
                            )
                            error_message = ""
                            try:
                                vlm_task_state = _query_vlm_task_state(
                                    api_url=vlm_api_url,
                                    api_key=vlm_api_key,
                                    x_auth_token=vlm_x_auth_token,
                                    model_name=vlm_model,
                                    prompt=task_prompt,
                                    image=base_image,
                                    timeout=vlm_timeout,
                                )
                            except Exception as exc:
                                error_message = str(exc)
                                vlm_task_state = _build_failed_task_state(error_message)

                            _update_episode_memory(episode_memory, vlm_task_state)
                            trace_record = {
                                "step": episode_steps,
                                "running_summary_before": memory_before["running_summary"],
                                "recent_history_before": memory_before["recent_history"],
                                "frame_state": vlm_task_state["frame_state"],
                                "task_memory": vlm_task_state["task_memory"],
                                "decision": vlm_task_state["decision"],
                                "parse_ok": vlm_task_state["parse_ok"],
                                "raw_text": vlm_task_state["raw_text"],
                                "error": error_message,
                            }
                            memory_trace.append(trace_record)
                            if _should_terminate_from_task_state(vlm_task_state):
                                done = True
                                success = True
                                termination_source = "vlm"
                                termination_reason = vlm_task_state["decision"]["reason"]
                        if done:
                            break
            finally:
                progress_bar.close()

            video_path = None
            if episode_idx in save_video_indices and isinstance(env, RecordVideo):
                env.video_cnt = next_video_index
                video_path = _predict_video_path(
                    output_session_dir=output_session_dir,
                    video_idx=env.video_cnt,
                )
                env.flush_video(video_sub_dir=task_slug)
                next_video_index = env.video_cnt
                saved_video_paths.append(str(video_path))

            episode_results.append(
                {
                    "episode_idx": episode_idx,
                    "task_id": task_id,
                    "task_name": task_name,
                    "reset_state_id": reset_state_id,
                    "success": success,
                    "steps": episode_steps,
                    "termination_source": termination_source,
                    "termination_reason": termination_reason,
                    "memory_trace": memory_trace,
                    "episode_memory": _snapshot_episode_memory(episode_memory),
                    "video_path": str(video_path) if video_path is not None else None,
                }
            )
            if video_path is not None:
                output_session_dir.mkdir(parents=True, exist_ok=True)
                json_path = video_path.with_suffix(".json")
                with open(json_path, "w") as fp:
                    json.dump(memory_trace, fp, indent=2)
    finally:
        env.close()
        _finalize_output_layout(
            video_base_dir=video_base_dir,
            output_session_dir=output_session_dir,
            seed=resolved_seed,
            task_dir=task_slug,
        )

    return {
        "task_id": task_id,
        "task_name": task_name,
        "seed": resolved_seed,
        "episodes": episode_results,
        "saved_video_paths": saved_video_paths,
        "video_base_dir": str(output_session_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0.5 on a single LIBERO-10 task and save videos."
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="LIBERO-10 task id.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="LIBERO-10 task language description or unique substring.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all LIBERO-10 task ids and descriptions, then exit.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for the local pi0.5/OpenPI checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory used as video_base_dir.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="How many reset states to evaluate for the selected task.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle reset states before truncating to num_episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed used for env selection and shuffle. Defaults to a random seed.",
    )
    parser.add_argument(
        "--save-fraction",
        type=float,
        default=1.0,
        help="Fraction of evaluated episodes to export as videos.",
    )
    parser.add_argument(
        "--vlm-check-interval",
        type=int,
        default=0,
        help="Run one contextual VLM check every k env steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--vlm-api-url",
        type=str,
        default=None,
        help="OpenAI-compatible VLM endpoint URL, for example /v1/chat/completions.",
    )
    parser.add_argument(
        "--vlm-api-key",
        type=str,
        default=None,
        help="Optional API key used for the VLM endpoint.",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        help="Model name sent to the VLM endpoint.",
    )
    parser.add_argument(
        "--vlm-x-auth-token",
        type=str,
        default=None,
        help="Optional x-auth-token header used by the VLM endpoint.",
    )
    parser.add_argument(
        "--vlm-prompt",
        type=str,
        default=DEFAULT_VLM_PROMPT,
        help="Prompt template used for contextual VLM checks.",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each VLM request.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metadata = load_libero10_metadata()

    if args.list_tasks:
        for idx, description in enumerate(metadata["task_descriptions"]):
            print(f"{idx}: {description}")
        return

    selected_task_id = resolve_task_id(
        task_descriptions=metadata["task_descriptions"],
        task_id=args.task_id,
        task_name=args.task_name,
    )
    results = run_single_task_eval(
        task_id=selected_task_id,
        config_name=args.config_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        shuffle=args.shuffle,
        seed=args.seed,
        save_fraction=args.save_fraction,
        vlm_check_interval=args.vlm_check_interval,
        vlm_api_url=args.vlm_api_url,
        vlm_api_key=args.vlm_api_key,
        vlm_x_auth_token=args.vlm_x_auth_token,
        vlm_model=args.vlm_model,
        vlm_prompt=args.vlm_prompt,
        vlm_timeout=args.vlm_timeout,
    )

    print(f"Task {results['task_id']}: {results['task_name']} (seed={results['seed']})")
    for episode in results["episodes"]:
        print(
            "Episode "
            f"{episode['episode_idx']}: reset_state_id={episode['reset_state_id']}, "
            f"success={episode['success']}, steps={episode['steps']}, "
            f"termination_source={episode['termination_source']}, "
            f"termination_reason={episode['termination_reason']!r}, "
            f"video={episode['video_path']}"
        )


if __name__ == "__main__":
    exit_code = 1
    try:
        main()
        exit_code = 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code == 0:
            # Some simulator / rendering native extensions can crash during
            # interpreter teardown after a successful run. Exit immediately once
            # outputs are flushed to avoid post-run destructor crashes.
            os._exit(0)
