from __future__ import annotations

"""CLI helpers for the standalone single-task evaluation script."""

import argparse

from .eval_runner import run_single_task_eval
from .libero_tasking import load_libero10_metadata, resolve_task_id
from .paths_and_config import DEFAULT_CONFIG_NAME, get_default_vlm_prompt


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
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional override for cfg.env.eval.max_episode_steps.",
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
        "--save-every-steps",
        type=int,
        default=0,
        help="Save incremental video/json checkpoints every k env steps. Set 0 to disable.",
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
        "--vlm-prompt-lang",
        type=str,
        choices=("zh", "en"),
        default="zh",
        help="Language for the default VLM prompt template (used when --vlm-prompt is not set).",
    )
    parser.add_argument(
        "--vlm-prompt",
        type=str,
        default=None,
        help="Prompt template used for contextual VLM checks. If omitted, uses --vlm-prompt-lang.",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each VLM request.",
    )
    parser.add_argument(
        "--vlm-bootstrap-prompt-version",
        type=str,
        default="v1",
        help="Version tag for bootstrap prompt logic.",
    )
    parser.add_argument(
        "--vlm-keyframe-prompt-version",
        type=str,
        default="v1",
        help="Version tag for keyframe prompt logic.",
    )
    parser.add_argument(
        "--vlm-keyframe-include-prev-image",
        action="store_true",
        help="Include previous keyframe image as reference in keyframe VLM requests.",
    )
    parser.add_argument(
        "--vlm-include-wrist-image",
        action="store_true",
        help="Include current wrist camera image in VLM requests.",
    )
    parser.add_argument(
        "--vlm-frame-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Frame interval in seconds used in keyframe prompt context. "
            "Set <=0 to auto-estimate by vlm_check_interval / env fps."
        ),
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
    resolved_vlm_prompt = args.vlm_prompt or get_default_vlm_prompt(args.vlm_prompt_lang)
    results = run_single_task_eval(
        task_id=selected_task_id,
        config_name=args.config_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        shuffle=args.shuffle,
        seed=args.seed,
        save_fraction=args.save_fraction,
        save_every_steps=args.save_every_steps,
        vlm_check_interval=args.vlm_check_interval,
        vlm_api_url=args.vlm_api_url,
        vlm_api_key=args.vlm_api_key,
        vlm_x_auth_token=args.vlm_x_auth_token,
        vlm_model=args.vlm_model,
        vlm_prompt=resolved_vlm_prompt,
        vlm_prompt_lang=args.vlm_prompt_lang,
        vlm_timeout=args.vlm_timeout,
        vlm_bootstrap_prompt_version=args.vlm_bootstrap_prompt_version,
        vlm_keyframe_prompt_version=args.vlm_keyframe_prompt_version,
        vlm_keyframe_include_prev_image=args.vlm_keyframe_include_prev_image,
        vlm_frame_interval_seconds=args.vlm_frame_interval_seconds,
        vlm_include_wrist_image=args.vlm_include_wrist_image,
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
