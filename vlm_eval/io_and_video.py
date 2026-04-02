from __future__ import annotations

"""Video/output helpers and small image/seed utilities."""

import base64
import io
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


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


def to_bool(value: Any) -> bool:
    try:
        return bool(value.item())
    except AttributeError:
        return bool(value)


def slugify_task_name(task_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in task_name).strip("_")


def build_output_session_dir(video_base_dir: Path, task_id: int) -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    return video_base_dir / f"{date_str}_{task_id}"


def predict_video_path(output_session_dir: Path, video_idx: int) -> Path:
    return output_session_dir / f"{video_idx}.mp4"


def resolve_seed(seed: int | None) -> int:
    """Use the provided seed or generate a random one when absent."""
    return seed if seed is not None else secrets.randbelow(2**31)


def get_next_video_index(output_session_dir: Path) -> int:
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


def finalize_output_layout(
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


def extract_base_image(obs: dict[str, Any]) -> Any:
    """Return the first environment's main camera image."""
    main_images = obs.get("main_images")
    if main_images is None:
        return None
    try:
        return main_images[0]
    except (IndexError, TypeError):
        return main_images


def encode_image_to_data_url(image: Any) -> str:
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
