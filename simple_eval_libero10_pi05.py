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
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

from tqdm.auto import tqdm
from vlm_eval.io_and_video import (
    build_output_session_dir,
    extract_base_image,
    finalize_output_layout,
    get_next_video_index,
    predict_video_path,
    resolve_seed,
    slugify_task_name,
    to_bool,
    compute_num_save_videos,
    select_video_indices,
)
from vlm_eval.libero_tasking import (
    build_task_reset_state_ids,
    choose_reset_state_ids,
    load_libero10_metadata,
    resolve_task_id,
)
from vlm_eval.obs_utils import standardize_env_obs
from vlm_eval.paths_and_config import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_VLM_PROMPT,
    REPO_ROOT,
    load_eval_cfg,
)
from vlm_eval.vlm_memory import (
    build_contextual_vlm_prompt,
    build_failed_task_state,
    init_episode_memory,
    query_vlm_task_state,
    should_terminate_from_task_state,
    snapshot_episode_memory,
    update_episode_memory,
)


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

    video_base_dir = Path(output_dir or (REPO_ROOT / "results" / "libero10_pi05_single_task"))
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
    next_video_index = get_next_video_index(output_session_dir)
    vlm_enabled = vlm_check_interval > 0
    if vlm_enabled and (not vlm_api_url or not vlm_model):
        raise ValueError(
            "When vlm_check_interval > 0, vlm_api_url and vlm_model are required."
        )

    try:
        for episode_idx, reset_state_id in enumerate(chosen_reset_state_ids):
            env.is_start = False
            obs, _ = env.reset(reset_state_ids=[reset_state_id])
            obs = standardize_env_obs(obs)

            episode_memory = init_episode_memory()
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
                        obs = standardize_env_obs(obs)
                        episode_steps += 1
                        progress_bar.update(1)

                        truncated = to_bool(truncations[0])
                        done = truncated
                        if truncated:
                            termination_source = "truncation"

                        should_check_vlm = (
                            vlm_enabled
                            and not done
                            and episode_steps > 0
                            and episode_steps % vlm_check_interval == 0
                        )
                        if should_check_vlm:
                            base_image = extract_base_image(obs)
                            if base_image is None:
                                raise ValueError(
                                    "Contextual VLM check requires obs['main_images'] to exist."
                                )
                            memory_before = snapshot_episode_memory(episode_memory)
                            task_prompt = build_contextual_vlm_prompt(
                                base_prompt=vlm_prompt,
                                task_name=task_name,
                                memory=memory_before,
                            )
                            error_message = ""
                            try:
                                vlm_task_state = query_vlm_task_state(
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
                                vlm_task_state = build_failed_task_state(error_message)

                            update_episode_memory(episode_memory, vlm_task_state)
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
                            if should_terminate_from_task_state(vlm_task_state):
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
                video_path = predict_video_path(
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
                    "episode_memory": snapshot_episode_memory(episode_memory),
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
        finalize_output_layout(
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


def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args()
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
