from __future__ import annotations

"""Execution flow for single-task LIBERO evaluation."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_and_video import (
    build_output_session_dir,
    compute_num_save_videos,
    extract_base_image,
    extract_wrist_image,
    finalize_output_layout,
    get_next_video_index,
    predict_video_path,
    resolve_seed,
    select_video_indices,
    slugify_task_name,
    to_bool,
)
from .libero_tasking import (
    build_task_reset_state_ids,
    choose_reset_state_ids,
    load_libero10_metadata,
)
from .obs_utils import standardize_env_obs
from .paths_and_config import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_VLM_PROMPT,
    REPO_ROOT,
    load_eval_cfg,
)
from .vlm_memory import (
    PARSE_MODE_BOOTSTRAP,
    PARSE_MODE_KEYFRAME,
    build_bootstrap_vlm_prompt,
    build_failed_task_state,
    build_keyframe_vlm_prompt,
    init_episode_memory,
    init_vlm_conversation,
    query_vlm_task_state,
    reset_vlm_caches,
    should_terminate_from_task_state,
    snapshot_episode_memory,
    update_episode_memory,
)


@dataclass
class EvalRuntime:
    env_cfg: Any
    model_cfg: Any
    env: Any
    model: Any
    record_video_cls: type
    env_fps: float
    task_name: str
    task_slug: str
    resolved_seed: int
    chosen_reset_state_ids: list[int]
    save_video_indices: set[int]
    video_base_dir: Path
    output_session_dir: Path
    next_video_index: int
    vlm_enabled: bool


@dataclass
class EpisodePaths:
    video_path: Path | None
    vlm_video_path: Path
    json_path: Path


@dataclass
class EpisodeState:
    memory: dict[str, Any]
    conversation: list[dict[str, Any]]
    done: bool
    success: bool
    steps: int
    termination_source: str
    termination_reason: str
    memory_trace: list[dict[str, Any]]
    previous_keyframe_image: Any | None
    vlm_inference_frames: list[Any]
    keyframe_vlm_count: int


def _build_runtime(
    task_id: int,
    config_name: str,
    model_path: str | None,
    output_dir: str | None,
    num_episodes: int | None,
    max_episode_steps: int | None,
    shuffle: bool,
    seed: int | None,
    save_fraction: float,
    vlm_check_interval: int,
) -> EvalRuntime:
    from omegaconf import open_dict

    from rlinf.envs import get_env_cls
    from rlinf.envs.wrappers import RecordVideo
    from rlinf.models import get_model

    cfg = load_eval_cfg(config_name=config_name)
    metadata = load_libero10_metadata(task_suite_name=cfg.env.eval.task_suite_name)
    task_descriptions = metadata["task_descriptions"]
    if task_id < 0 or task_id >= len(task_descriptions):
        raise ValueError(
            f"task_id must be in [0, {len(task_descriptions) - 1}], got {task_id}"
        )

    resolved_seed = resolve_seed(seed)
    task_name = task_descriptions[task_id]
    task_slug = slugify_task_name(task_name)
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

    video_base_dir = Path(
        output_dir or (REPO_ROOT / "results" / "libero10_pi05_single_task")
    )
    output_session_dir = build_output_session_dir(video_base_dir, task_id)
    with open_dict(cfg):
        if model_path is not None:
            cfg.actor.model.model_path = model_path
            cfg.rollout.model.model_path = model_path
        cfg.env.eval.total_num_envs = 1
        cfg.env.eval.auto_reset = False
        cfg.env.eval.ignore_terminations = True
        cfg.env.eval.use_fixed_reset_state_ids = False
        cfg.env.eval.seed = resolved_seed
        if max_episode_steps is not None:
            cfg.env.eval.max_episode_steps = int(max_episode_steps)
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
    env_fps = float(getattr(env, "_fps", 30))

    model_cfg = copy.deepcopy(cfg.actor.model)
    model = get_model(model_cfg)
    model.eval()

    return EvalRuntime(
        env_cfg=env_cfg,
        model_cfg=model_cfg,
        env=env,
        model=model,
        record_video_cls=RecordVideo,
        env_fps=env_fps,
        task_name=task_name,
        task_slug=task_slug,
        resolved_seed=resolved_seed,
        chosen_reset_state_ids=chosen_reset_state_ids,
        save_video_indices=save_video_indices,
        video_base_dir=video_base_dir,
        output_session_dir=output_session_dir,
        next_video_index=get_next_video_index(output_session_dir),
        vlm_enabled=vlm_check_interval > 0,
    )


def _build_episode_state() -> EpisodeState:
    return EpisodeState(
        memory=init_episode_memory(),
        conversation=init_vlm_conversation(),
        done=False,
        success=False,
        steps=0,
        termination_source="env",
        termination_reason="",
        memory_trace=[],
        previous_keyframe_image=None,
        vlm_inference_frames=[],
        keyframe_vlm_count=0,
    )


def _build_episode_paths(runtime: EvalRuntime, episode_idx: int) -> EpisodePaths:
    video_path: Path | None = None
    if episode_idx in runtime.save_video_indices and isinstance(
        runtime.env, runtime.record_video_cls
    ):
        video_path = predict_video_path(
            output_session_dir=runtime.output_session_dir,
            video_idx=runtime.next_video_index,
        )
        runtime.next_video_index += 1

    json_path = (
        video_path.with_suffix(".json")
        if video_path is not None
        else runtime.output_session_dir / f"episode_{episode_idx}.json"
    )
    vlm_video_path = (
        video_path.with_name(f"{video_path.stem}_vlm{video_path.suffix}")
        if video_path is not None
        else runtime.output_session_dir / f"episode_{episode_idx}_vlm.mp4"
    )
    return EpisodePaths(
        video_path=video_path,
        vlm_video_path=vlm_video_path,
        json_path=json_path,
    )


def _save_episode_checkpoint(
    runtime: EvalRuntime,
    paths: EpisodePaths,
    memory_trace: list[dict[str, Any]],
    vlm_inference_frames: list[Any],
) -> None:
    import imageio
    import numpy as np

    runtime.output_session_dir.mkdir(parents=True, exist_ok=True)
    if paths.video_path is not None and isinstance(runtime.env, runtime.record_video_cls):
        frames = list(runtime.env.render_images)
        if frames:
            writer = imageio.get_writer(
                str(paths.video_path),
                fps=runtime.env_fps,
            )
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()
    if vlm_inference_frames:
        writer = imageio.get_writer(
            str(paths.vlm_video_path),
            fps=1.0,
        )
        try:
            for frame in vlm_inference_frames:
                frame_array = frame
                if hasattr(frame_array, "detach"):
                    frame_array = frame_array.detach().cpu().numpy()
                elif hasattr(frame_array, "cpu") and hasattr(frame_array, "numpy"):
                    frame_array = frame_array.cpu().numpy()
                else:
                    frame_array = np.asarray(frame_array)

                if frame_array.dtype.kind == "f":
                    # Handle both [0, 1] and [0, 255] float images.
                    max_value = float(np.max(frame_array)) if frame_array.size else 0.0
                    scale = 255.0 if max_value <= 1.0 else 1.0
                    frame_array = np.clip(frame_array * scale, 0, 255).astype(np.uint8)
                elif frame_array.dtype != np.uint8:
                    frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)

                writer.append_data(frame_array)
        finally:
            writer.close()
    with open(paths.json_path, "w") as fp:
        json.dump(memory_trace, fp, indent=2, ensure_ascii=False)


def _run_bootstrap_vlm_check(
    *,
    episode_state: EpisodeState,
    obs: dict[str, Any],
    task_name: str,
    vlm_api_url: str | None,
    vlm_api_key: str | None,
    vlm_x_auth_token: str | None,
    vlm_model: str | None,
    vlm_prompt: str,
    vlm_prompt_lang: str,
    vlm_timeout: float,
    vlm_bootstrap_prompt_version: str,
    vlm_include_wrist_image: bool,
) -> None:
    base_image = extract_base_image(obs)
    if base_image is None:
        raise ValueError("Bootstrap VLM check requires obs['main_images'] to exist.")
    episode_state.vlm_inference_frames.append(copy.deepcopy(base_image))
    wrist_image = extract_wrist_image(obs) if vlm_include_wrist_image else None

    bootstrap_prompt = build_bootstrap_vlm_prompt(
        base_prompt=vlm_prompt,
        task_name=task_name,
        prompt_version=vlm_bootstrap_prompt_version,
        prompt_language=vlm_prompt_lang,
    )
    bootstrap_error = ""
    try:
        bootstrap_state = query_vlm_task_state(
            api_url=vlm_api_url,
            api_key=vlm_api_key,
            x_auth_token=vlm_x_auth_token,
            model_name=vlm_model,
            prompt=bootstrap_prompt,
            image=base_image,
            wrist_image=wrist_image,
            timeout=vlm_timeout,
            parse_mode=PARSE_MODE_BOOTSTRAP,
            conversation_messages=episode_state.conversation,
        )
    except Exception as exc:
        bootstrap_error = str(exc)
        bootstrap_state = build_failed_task_state(
            bootstrap_error,
            parse_mode=PARSE_MODE_BOOTSTRAP,
        )

    update_episode_memory(episode_state.memory, bootstrap_state)
    memory_after_bootstrap = snapshot_episode_memory(episode_state.memory)
    bootstrap_trace = {
        "phase": PARSE_MODE_BOOTSTRAP,
        "step": 0,
        "prompt_version": vlm_bootstrap_prompt_version,
        "include_wrist_image": vlm_include_wrist_image,
        "episode_memory": memory_after_bootstrap,
        "parse_ok": bootstrap_state["parse_ok"],
        "raw_text": bootstrap_state["raw_text"],
        "error": bootstrap_error,
    }
    if bootstrap_state["parse_ok"]:
        bootstrap_trace["parsed_content"] = {
            "task_profile": bootstrap_state["task_profile"],
            "frame_summary": bootstrap_state["frame_summary"],
            "progress_summary": bootstrap_state["progress_summary"],
            "decision": bootstrap_state["decision"],
        }
    episode_state.memory_trace.append(bootstrap_trace)
    episode_state.previous_keyframe_image = base_image


def _run_keyframe_vlm_check(
    *,
    episode_state: EpisodeState,
    obs: dict[str, Any],
    task_name: str,
    vlm_check_interval: int,
    env_fps: float,
    vlm_api_url: str | None,
    vlm_api_key: str | None,
    vlm_x_auth_token: str | None,
    vlm_model: str | None,
    vlm_prompt: str,
    vlm_prompt_lang: str,
    vlm_timeout: float,
    vlm_keyframe_prompt_version: str,
    vlm_keyframe_include_prev_image: bool,
    vlm_frame_interval_seconds: float,
    vlm_include_wrist_image: bool,
) -> None:
    base_image = extract_base_image(obs)
    if base_image is None:
        raise ValueError("Contextual VLM check requires obs['main_images'] to exist.")
    episode_state.vlm_inference_frames.append(copy.deepcopy(base_image))
    wrist_image = extract_wrist_image(obs) if vlm_include_wrist_image else None
    episode_state.keyframe_vlm_count += 1

    estimated_frame_interval_seconds = float(vlm_check_interval / max(1.0, env_fps))
    effective_frame_interval_seconds = (
        float(vlm_frame_interval_seconds)
        if vlm_frame_interval_seconds > 0
        else estimated_frame_interval_seconds
    )
    task_prompt = build_keyframe_vlm_prompt(
        base_prompt=vlm_prompt,
        task_name=task_name,
        frame_interval_seconds=effective_frame_interval_seconds,
        prompt_version=vlm_keyframe_prompt_version,
        prompt_language=vlm_prompt_lang,
        keyframe_index=episode_state.keyframe_vlm_count,
    )
    error_message = ""
    try:
        previous_image = (
            episode_state.previous_keyframe_image
            if vlm_keyframe_include_prev_image
            else None
        )
        vlm_task_state = query_vlm_task_state(
            api_url=vlm_api_url,
            api_key=vlm_api_key,
            x_auth_token=vlm_x_auth_token,
            model_name=vlm_model,
            prompt=task_prompt,
            image=base_image,
            wrist_image=wrist_image,
            timeout=vlm_timeout,
            parse_mode=PARSE_MODE_KEYFRAME,
            previous_image=previous_image,
            conversation_messages=episode_state.conversation,
        )
    except Exception as exc:
        error_message = str(exc)
        vlm_task_state = build_failed_task_state(
            error_message,
            parse_mode=PARSE_MODE_KEYFRAME,
        )

    update_episode_memory(episode_state.memory, vlm_task_state)
    memory_after = snapshot_episode_memory(episode_state.memory)
    trace_record = {
        "phase": PARSE_MODE_KEYFRAME,
        "step": episode_state.steps,
        "prompt_version": vlm_keyframe_prompt_version,
        "include_previous_image": vlm_keyframe_include_prev_image,
        "include_wrist_image": vlm_include_wrist_image,
        "frame_interval_seconds": effective_frame_interval_seconds,
        "episode_memory_after": memory_after,
        "parse_ok": vlm_task_state["parse_ok"],
        "raw_text": vlm_task_state["raw_text"],
        "error": error_message,
    }
    if vlm_task_state["parse_ok"]:
        trace_record["parsed_content"] = {
            "frame_summary": vlm_task_state["frame_summary"],
            "change_summary": vlm_task_state["change_summary"],
            "progress_summary": vlm_task_state["progress_summary"],
            "decision": vlm_task_state["decision"],
        }
        # trace_record["running_summary_after"] = memory_after.get("running_summary", "")
    episode_state.memory_trace.append(trace_record)
    if should_terminate_from_task_state(vlm_task_state, episode_state.memory):
        episode_state.done = True
        episode_state.success = True
        episode_state.termination_source = "vlm"
        episode_state.termination_reason = vlm_task_state["decision"]["reason"]
    episode_state.previous_keyframe_image = base_image


def _maybe_run_vlm_check(
    *,
    episode_state: EpisodeState,
    obs: dict[str, Any],
    task_name: str,
    vlm_enabled: bool,
    vlm_check_interval: int,
    env_fps: float,
    vlm_api_url: str | None,
    vlm_api_key: str | None,
    vlm_x_auth_token: str | None,
    vlm_model: str | None,
    vlm_prompt: str,
    vlm_prompt_lang: str,
    vlm_timeout: float,
    vlm_keyframe_prompt_version: str,
    vlm_keyframe_include_prev_image: bool,
    vlm_frame_interval_seconds: float,
    vlm_include_wrist_image: bool,
) -> None:
    should_check_vlm = (
        vlm_enabled
        and not episode_state.done
        and episode_state.steps > 0
        and episode_state.steps % vlm_check_interval == 0
    )
    if not should_check_vlm:
        return

    _run_keyframe_vlm_check(
        episode_state=episode_state,
        obs=obs,
        task_name=task_name,
        vlm_check_interval=vlm_check_interval,
        env_fps=env_fps,
        vlm_api_url=vlm_api_url,
        vlm_api_key=vlm_api_key,
        vlm_x_auth_token=vlm_x_auth_token,
        vlm_model=vlm_model,
        vlm_prompt=vlm_prompt,
        vlm_prompt_lang=vlm_prompt_lang,
        vlm_timeout=vlm_timeout,
        vlm_keyframe_prompt_version=vlm_keyframe_prompt_version,
        vlm_keyframe_include_prev_image=vlm_keyframe_include_prev_image,
        vlm_frame_interval_seconds=vlm_frame_interval_seconds,
        vlm_include_wrist_image=vlm_include_wrist_image,
    )


def _step_episode_loop(
    *,
    runtime: EvalRuntime,
    episode_idx: int,
    episode_state: EpisodeState,
    obs: dict[str, Any],
    episode_paths: EpisodePaths,
    save_every_steps: int,
    vlm_check_interval: int,
    vlm_api_url: str | None,
    vlm_api_key: str | None,
    vlm_x_auth_token: str | None,
    vlm_model: str | None,
    vlm_prompt: str,
    vlm_prompt_lang: str,
    vlm_timeout: float,
    vlm_keyframe_prompt_version: str,
    vlm_keyframe_include_prev_image: bool,
    vlm_frame_interval_seconds: float,
    vlm_include_wrist_image: bool,
) -> None:
    from rlinf.envs.action_utils import prepare_actions
    from tqdm.auto import tqdm

    progress_bar = tqdm(
        total=runtime.env_cfg.max_episode_steps,
        desc=f"Episode {episode_idx}",
        leave=True,
    )
    try:
        while not episode_state.done and episode_state.steps < runtime.env_cfg.max_episode_steps:
            raw_chunk_actions, _ = runtime.model.predict_action_batch(
                env_obs=obs,
                mode="eval",
                compute_values=False,
            )
            chunk_actions = prepare_actions(
                raw_chunk_actions=raw_chunk_actions,
                env_type=runtime.env_cfg.env_type,
                model_type=runtime.model_cfg.model_type,
                num_action_chunks=runtime.model_cfg.num_action_chunks,
                action_dim=runtime.model_cfg.action_dim,
                policy=runtime.model_cfg.get("policy_setup", None),
                wm_env_type=runtime.env_cfg.get("wm_env_type", None),
            )

            for chunk_step in range(chunk_actions.shape[1]):
                action = chunk_actions[:, chunk_step]
                obs, _reward, terminations, truncations, _infos = runtime.env.step(action)
                obs = standardize_env_obs(obs)
                episode_state.steps += 1
                progress_bar.update(1)

                truncated = to_bool(truncations[0])
                episode_state.done = truncated
                if truncated:
                    episode_state.termination_source = "truncation"

                _maybe_run_vlm_check(
                    episode_state=episode_state,
                    obs=obs,
                    task_name=runtime.task_name,
                    vlm_enabled=runtime.vlm_enabled,
                    vlm_check_interval=vlm_check_interval,
                    env_fps=runtime.env_fps,
                    vlm_api_url=vlm_api_url,
                    vlm_api_key=vlm_api_key,
                    vlm_x_auth_token=vlm_x_auth_token,
                    vlm_model=vlm_model,
                    vlm_prompt=vlm_prompt,
                    vlm_prompt_lang=vlm_prompt_lang,
                    vlm_timeout=vlm_timeout,
                    vlm_keyframe_prompt_version=vlm_keyframe_prompt_version,
                    vlm_keyframe_include_prev_image=vlm_keyframe_include_prev_image,
                    vlm_frame_interval_seconds=vlm_frame_interval_seconds,
                    vlm_include_wrist_image=vlm_include_wrist_image,
                )

                should_save_checkpoint = (
                    save_every_steps > 0
                    and episode_state.steps > 0
                    and episode_state.steps % save_every_steps == 0
                )
                if should_save_checkpoint:
                    _save_episode_checkpoint(
                        runtime,
                        episode_paths,
                        episode_state.memory_trace,
                        episode_state.vlm_inference_frames,
                    )
                if episode_state.done:
                    break
    finally:
        progress_bar.close()


def _run_single_episode(
    *,
    runtime: EvalRuntime,
    task_id: int,
    episode_idx: int,
    reset_state_id: int,
    saved_video_paths: list[str],
    save_every_steps: int,
    vlm_check_interval: int,
    vlm_api_url: str | None,
    vlm_api_key: str | None,
    vlm_x_auth_token: str | None,
    vlm_model: str | None,
    vlm_prompt: str,
    vlm_prompt_lang: str,
    vlm_timeout: float,
    vlm_bootstrap_prompt_version: str,
    vlm_keyframe_prompt_version: str,
    vlm_keyframe_include_prev_image: bool,
    vlm_frame_interval_seconds: float,
    vlm_include_wrist_image: bool,
) -> dict[str, Any]:
    if runtime.vlm_enabled:
        reset_vlm_caches(
            api_url=vlm_api_url,
            api_key=vlm_api_key,
            x_auth_token=vlm_x_auth_token,
            timeout=vlm_timeout,
            reset_prefix_cache=True,
            reset_mm_cache=True,
            reset_running_requests=False,
            request_id=f"episode-{episode_idx}-reset-caches",
        )

    runtime.env.is_start = False
    obs, _ = runtime.env.reset(reset_state_ids=[reset_state_id])
    obs = standardize_env_obs(obs)

    episode_state = _build_episode_state()
    episode_paths = _build_episode_paths(runtime, episode_idx)

    if runtime.vlm_enabled:
        _run_bootstrap_vlm_check(
            episode_state=episode_state,
            obs=obs,
            task_name=runtime.task_name,
            vlm_api_url=vlm_api_url,
            vlm_api_key=vlm_api_key,
            vlm_x_auth_token=vlm_x_auth_token,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt,
            vlm_prompt_lang=vlm_prompt_lang,
            vlm_timeout=vlm_timeout,
            vlm_bootstrap_prompt_version=vlm_bootstrap_prompt_version,
            vlm_include_wrist_image=vlm_include_wrist_image,
        )

    _step_episode_loop(
        runtime=runtime,
        episode_idx=episode_idx,
        episode_state=episode_state,
        obs=obs,
        episode_paths=episode_paths,
        save_every_steps=save_every_steps,
        vlm_check_interval=vlm_check_interval,
        vlm_api_url=vlm_api_url,
        vlm_api_key=vlm_api_key,
        vlm_x_auth_token=vlm_x_auth_token,
        vlm_model=vlm_model,
        vlm_prompt=vlm_prompt,
        vlm_prompt_lang=vlm_prompt_lang,
        vlm_timeout=vlm_timeout,
        vlm_keyframe_prompt_version=vlm_keyframe_prompt_version,
        vlm_keyframe_include_prev_image=vlm_keyframe_include_prev_image,
        vlm_frame_interval_seconds=vlm_frame_interval_seconds,
        vlm_include_wrist_image=vlm_include_wrist_image,
    )

    video_path = None
    did_save_checkpoint = False
    if episode_idx in runtime.save_video_indices and isinstance(
        runtime.env, runtime.record_video_cls
    ):
        _save_episode_checkpoint(
            runtime,
            episode_paths,
            episode_state.memory_trace,
            episode_state.vlm_inference_frames,
        )
        did_save_checkpoint = True
        video_path = episode_paths.video_path
        if video_path is not None:
            saved_video_paths.append(str(video_path))

    episode_result = {
        "episode_idx": episode_idx,
        "task_id": task_id,
        "task_name": runtime.task_name,
        "reset_state_id": reset_state_id,
        "success": episode_state.success,
        "steps": episode_state.steps,
        "termination_source": episode_state.termination_source,
        "termination_reason": episode_state.termination_reason,
        "memory_trace": episode_state.memory_trace,
        "episode_memory": snapshot_episode_memory(episode_state.memory),
        "video_path": str(video_path) if video_path is not None else None,
        "vlm_video_path": (
            str(episode_paths.vlm_video_path)
            if episode_state.vlm_inference_frames
            else None
        ),
    }
    should_save_final_vlm_video = bool(episode_state.vlm_inference_frames)
    if not did_save_checkpoint and (save_every_steps > 0 or should_save_final_vlm_video):
        _save_episode_checkpoint(
            runtime,
            episode_paths,
            episode_state.memory_trace,
            episode_state.vlm_inference_frames,
        )
    return episode_result


def run_single_task_eval(
    task_id: int,
    config_name: str = DEFAULT_CONFIG_NAME,
    model_path: str | None = None,
    output_dir: str | None = None,
    num_episodes: int | None = 1,
    max_episode_steps: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    save_fraction: float = 1.0,
    save_every_steps: int = 0,
    vlm_check_interval: int = 0,
    vlm_api_url: str | None = None,
    vlm_api_key: str | None = None,
    vlm_x_auth_token: str | None = None,
    vlm_model: str | None = None,
    vlm_prompt: str = DEFAULT_VLM_PROMPT,
    vlm_prompt_lang: str = "zh",
    vlm_timeout: float = 30.0,
    vlm_bootstrap_prompt_version: str = "v1",
    vlm_keyframe_prompt_version: str = "v1",
    vlm_keyframe_include_prev_image: bool = False,
    vlm_frame_interval_seconds: float = 0.0,
    vlm_include_wrist_image: bool = False,
) -> dict[str, Any]:
    """Run a single-task LIBERO-10 evaluation loop without Ray workers."""
    runtime = _build_runtime(
        task_id=task_id,
        config_name=config_name,
        model_path=model_path,
        output_dir=output_dir,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        shuffle=shuffle,
        seed=seed,
        save_fraction=save_fraction,
        vlm_check_interval=vlm_check_interval,
    )
    if runtime.vlm_enabled and (not vlm_api_url or not vlm_model):
        raise ValueError(
            "When vlm_check_interval > 0, vlm_api_url and vlm_model are required."
        )

    episode_results: list[dict[str, Any]] = []
    saved_video_paths: list[str] = []

    try:
        for episode_idx, reset_state_id in enumerate(runtime.chosen_reset_state_ids):
            episode_result = _run_single_episode(
                runtime=runtime,
                task_id=task_id,
                episode_idx=episode_idx,
                reset_state_id=reset_state_id,
                saved_video_paths=saved_video_paths,
                save_every_steps=save_every_steps,
                vlm_check_interval=vlm_check_interval,
                vlm_api_url=vlm_api_url,
                vlm_api_key=vlm_api_key,
                vlm_x_auth_token=vlm_x_auth_token,
                vlm_model=vlm_model,
                vlm_prompt=vlm_prompt,
                vlm_prompt_lang=vlm_prompt_lang,
                vlm_timeout=vlm_timeout,
                vlm_bootstrap_prompt_version=vlm_bootstrap_prompt_version,
                vlm_keyframe_prompt_version=vlm_keyframe_prompt_version,
                vlm_keyframe_include_prev_image=vlm_keyframe_include_prev_image,
                vlm_frame_interval_seconds=vlm_frame_interval_seconds,
                vlm_include_wrist_image=vlm_include_wrist_image,
            )
            episode_results.append(episode_result)
    finally:
        runtime.env.close()
        finalize_output_layout(
            video_base_dir=runtime.video_base_dir,
            output_session_dir=runtime.output_session_dir,
            seed=runtime.resolved_seed,
            task_dir=runtime.task_slug,
        )

    return {
        "task_id": task_id,
        "task_name": runtime.task_name,
        "seed": runtime.resolved_seed,
        "episodes": episode_results,
        "saved_video_paths": saved_video_paths,
        "video_base_dir": str(runtime.output_session_dir),
    }
