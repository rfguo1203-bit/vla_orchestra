#!/usr/bin/env python3
"""Run one VLA-only LIBERO-10 episode and save video to outputs/debug."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_eval.eval_runner import run_single_task_eval
from vlm_eval.libero_tasking import load_libero10_metadata
from vlm_eval.paths_and_config import DEFAULT_CONFIG_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLA-only single-task rollout on LIBERO-10 (no VLM)."
    )
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO-10 task id.")
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
        help="Optional local checkpoint override.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional max steps override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_libero10_metadata()
    task_descriptions = metadata["task_descriptions"]
    if args.task_id < 0 or args.task_id >= len(task_descriptions):
        raise ValueError(
            f"task_id must be in [0, {len(task_descriptions) - 1}], got {args.task_id}"
        )

    output_dir = REPO_ROOT / "outputs" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_single_task_eval(
        task_id=args.task_id,
        config_name=args.config_name,
        model_path=args.model_path,
        output_dir=str(output_dir),
        num_episodes=1,
        max_episode_steps=args.max_episode_steps,
        shuffle=False,
        seed=args.seed,
        save_fraction=1.0,
        save_every_steps=0,
        vlm_check_interval=0,  # Disable VLM: pure VLA rollout.
    )

    episode = result["episodes"][0]
    print(f"Task {result['task_id']}: {result['task_name']} (seed={result['seed']})")
    print(
        "Episode 0: "
        f"reset_state_id={episode['reset_state_id']}, "
        f"success={episode['success']}, "
        f"steps={episode['steps']}, "
        f"termination_source={episode['termination_source']}, "
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
            # Avoid occasional native-extension teardown crash after successful run.
            os._exit(0)
