from __future__ import annotations

"""LIBERO task metadata loading and task/reset-state selection helpers."""

import random
import re
from itertools import accumulate
from typing import Any

from .paths_and_config import _set_runtime_env


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
