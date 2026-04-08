from __future__ import annotations

"""Generalized VLM contextual prompts, parsing, and episode-state helpers."""

import copy
import json
import re
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any

from .io_and_video import encode_image_to_data_url

VALID_DECISION_STATUS = {"in_progress", "completed", "uncertain"}
PARSE_MODE_BOOTSTRAP = "bootstrap"
PARSE_MODE_KEYFRAME = "keyframe"
VALID_CONDITION_STATUS = {
    "not_met",
    "partially_met",
    "likely_met",
    "confirmed_met",
    "uncertain",
}


def extract_vlm_text_content(response_payload: dict[str, Any]) -> str:
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


def empty_decision() -> dict[str, Any]:
    return {
        "terminate": False,
        "status": "uncertain",
        "reason": "",
    }


def empty_bootstrap_task_state(
    raw_text: str = "",
    parse_ok: bool = False,
) -> dict[str, Any]:
    """Return bootstrap phase defaults."""
    return {
        "parse_mode": PARSE_MODE_BOOTSTRAP,
        "task_model": {},
        "frame_state": {},
        "progress_state": {},
        "decision": empty_decision(),
        "raw_text": raw_text,
        "parse_ok": parse_ok,
    }


def empty_keyframe_task_state(
    raw_text: str = "",
    parse_ok: bool = False,
) -> dict[str, Any]:
    """Return keyframe phase defaults."""
    return {
        "parse_mode": PARSE_MODE_KEYFRAME,
        "frame_state": {},
        "frame_delta": {},
        "task_model_patch": {},
        "progress_state_patch": {},
        "decision": empty_decision(),
        "raw_text": raw_text,
        "parse_ok": parse_ok,
    }


def strip_vlm_thinking(text: str) -> str:
    """Drop hidden reasoning content when the model emits a closing think tag."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_json_from_fenced_block(text: str) -> dict[str, Any] | None:
    """Extract the last valid JSON object from a fenced ```json block."""
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_top_level_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract valid top-level JSON objects without matching nested dicts."""
    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    objects.append(parsed)
                start = None

    return objects


def extract_last_top_level_json_object(text: str) -> dict[str, Any] | None:
    """Extract the last valid top-level JSON object from free-form model output."""
    objects = extract_top_level_json_objects(text)
    return objects[-1] if objects else None


def parse_decision(decision: Any) -> dict[str, Any] | None:
    """Validate and normalize a decision object."""
    if not isinstance(decision, dict):
        return None
    terminate = decision.get("terminate")
    status = decision.get("status")
    reason = decision.get("reason")
    if not isinstance(terminate, bool):
        return None
    if status not in VALID_DECISION_STATUS:
        return None
    if not isinstance(reason, str):
        return None
    return {
        "terminate": terminate,
        "status": status,
        "reason": reason.strip(),
    }


def parse_vlm_bootstrap_state(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Parse bootstrap-phase response into generalized task state."""
    content = extract_vlm_text_content(response_payload)
    cleaned_content = strip_vlm_thinking(content)
    parsed = extract_json_from_fenced_block(cleaned_content)
    if parsed is None:
        parsed = extract_last_top_level_json_object(cleaned_content)
    if parsed is None:
        return empty_bootstrap_task_state(raw_text=content, parse_ok=False)

    task_model = parsed.get("task_model")
    frame_state = parsed.get("frame_state")
    progress_state = parsed.get("progress_state")
    if not isinstance(task_model, dict):
        return empty_bootstrap_task_state(raw_text=content, parse_ok=False)
    if not isinstance(frame_state, dict):
        return empty_bootstrap_task_state(raw_text=content, parse_ok=False)
    if not isinstance(progress_state, dict):
        return empty_bootstrap_task_state(raw_text=content, parse_ok=False)

    normalized_decision = parse_decision(parsed.get("decision"))
    if normalized_decision is None:
        normalized_decision = empty_decision()

    task_state = empty_bootstrap_task_state(raw_text=content, parse_ok=True)
    task_state["task_model"] = task_model
    task_state["frame_state"] = frame_state
    task_state["progress_state"] = progress_state
    task_state["decision"] = normalized_decision
    return task_state


def parse_vlm_keyframe_state(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Parse keyframe-phase response into generalized task state."""
    content = extract_vlm_text_content(response_payload)
    cleaned_content = strip_vlm_thinking(content)
    parsed = extract_json_from_fenced_block(cleaned_content)
    if parsed is None:
        parsed = extract_last_top_level_json_object(cleaned_content)
    if parsed is None:
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)

    frame_state = parsed.get("frame_state")
    frame_delta = parsed.get("frame_delta")
    task_model_patch = parsed.get("task_model_patch")
    progress_state_patch = parsed.get("progress_state_patch")
    decision = parsed.get("decision")
    if not isinstance(frame_state, dict):
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)
    if not isinstance(frame_delta, dict):
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)
    if not isinstance(progress_state_patch, dict):
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)
    if task_model_patch is None:
        task_model_patch = {}
    if not isinstance(task_model_patch, dict):
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)
    normalized_decision = parse_decision(decision)
    if normalized_decision is None:
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)

    task_state = empty_keyframe_task_state(raw_text=content, parse_ok=True)
    task_state["frame_state"] = frame_state
    task_state["frame_delta"] = frame_delta
    task_state["task_model_patch"] = task_model_patch
    task_state["progress_state_patch"] = progress_state_patch
    task_state["decision"] = normalized_decision
    return task_state


def init_episode_memory() -> dict[str, Any]:
    """Create the per-episode generalized memory container."""
    return {
        "task_model": {},
        "previous_frame_state": {},
        "progress_state": {},
    }


def snapshot_episode_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the current episode memory state for logging/debugging."""
    return {
        "task_model": copy.deepcopy(memory.get("task_model", {})),
        "previous_frame_state": copy.deepcopy(memory.get("previous_frame_state", {})),
        "progress_state": copy.deepcopy(memory.get("progress_state", {})),
    }


def _extract_success_condition_ids(task_model: dict[str, Any]) -> list[str]:
    success_conditions = task_model.get("success_conditions")
    if not isinstance(success_conditions, list):
        return []
    ids: list[str] = []
    for idx, item in enumerate(success_conditions):
        if isinstance(item, dict):
            cond_id = item.get("id")
            if isinstance(cond_id, str) and cond_id.strip():
                ids.append(cond_id.strip())
                continue
        ids.append(f"cond_{idx + 1}")
    return ids


def _index_condition_status_items(
    condition_status: Any,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(condition_status, list):
        return indexed
    for idx, item in enumerate(condition_status):
        if not isinstance(item, dict):
            continue
        cond_id = item.get("condition_id")
        if not isinstance(cond_id, str) or not cond_id.strip():
            cond_id = f"cond_{idx + 1}"
        status = item.get("status")
        if status not in VALID_CONDITION_STATUS:
            continue
        indexed[cond_id] = copy.deepcopy(item)
        indexed[cond_id]["condition_id"] = cond_id
    return indexed


def _normalize_status_conflict(
    previous_item: dict[str, Any],
    patch_item: dict[str, Any],
) -> dict[str, Any]:
    prev_status = previous_item.get("status")
    patch_status = patch_item.get("status")
    if (
        prev_status == "confirmed_met"
        and patch_status in {"not_met", "partially_met", "likely_met"}
    ):
        degraded = copy.deepcopy(previous_item)
        degraded["status"] = "uncertain"
        degraded["evidence"] = (
            f"conflict_between_frames: previous={prev_status}, current={patch_status}"
        )
        return degraded
    if prev_status == "confirmed_met" and patch_status == "uncertain":
        # Do not roll back confirmed status on a single uncertain frame.
        return copy.deepcopy(previous_item)
    return copy.deepcopy(patch_item)


def _normalize_visual_cues(cues: Any) -> list[str]:
    if not isinstance(cues, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in cues:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def merge_task_model_patch(
    task_model: dict[str, Any],
    task_model_patch: dict[str, Any],
) -> dict[str, Any]:
    """Apply a constrained patch: only append entities[*].visual_cues."""
    merged = copy.deepcopy(task_model)
    patch = copy.deepcopy(task_model_patch)
    entities = merged.get("entities")
    patch_entities = patch.get("entities")
    if not isinstance(entities, list) or not isinstance(patch_entities, list):
        return merged

    index_by_id: dict[str, int] = {}
    index_by_name: dict[str, int] = {}
    for idx, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        entity_id = entity.get("id")
        if isinstance(entity_id, str) and entity_id.strip():
            index_by_id[entity_id.strip()] = idx
        name = entity.get("name")
        if isinstance(name, str) and name.strip():
            index_by_name[name.strip()] = idx

    for patch_entity in patch_entities:
        if not isinstance(patch_entity, dict):
            continue
        entity_idx: int | None = None
        entity_id = patch_entity.get("id")
        if isinstance(entity_id, str) and entity_id.strip():
            entity_idx = index_by_id.get(entity_id.strip())
        if entity_idx is None:
            name = patch_entity.get("name")
            if isinstance(name, str) and name.strip():
                entity_idx = index_by_name.get(name.strip())
        if entity_idx is None:
            continue
        entity = entities[entity_idx]
        if not isinstance(entity, dict):
            continue
        existing_cues = _normalize_visual_cues(entity.get("visual_cues"))
        patch_cues = _normalize_visual_cues(patch_entity.get("visual_cues"))
        if not patch_cues:
            continue
        seen = set(existing_cues)
        for cue in patch_cues:
            if cue not in seen:
                existing_cues.append(cue)
                seen.add(cue)
        entity["visual_cues"] = existing_cues
    merged["entities"] = entities
    return merged


def merge_progress_state(
    previous_progress_state: dict[str, Any],
    progress_state_patch: dict[str, Any],
) -> dict[str, Any]:
    """Merge current keyframe patch into running progress state."""
    merged = copy.deepcopy(previous_progress_state)
    patch = copy.deepcopy(progress_state_patch)

    prev_condition_map = _index_condition_status_items(merged.get("condition_status"))
    patch_condition_map = _index_condition_status_items(patch.get("condition_status"))
    merged_condition_ids = list(prev_condition_map.keys())
    for cond_id in patch_condition_map:
        if cond_id not in merged_condition_ids:
            merged_condition_ids.append(cond_id)
    merged_condition_status: list[dict[str, Any]] = []
    for cond_id in merged_condition_ids:
        previous_item = prev_condition_map.get(cond_id)
        patch_item = patch_condition_map.get(cond_id)
        if previous_item is None and patch_item is not None:
            merged_condition_status.append(copy.deepcopy(patch_item))
            continue
        if previous_item is not None and patch_item is None:
            merged_condition_status.append(copy.deepcopy(previous_item))
            continue
        if previous_item is not None and patch_item is not None:
            merged_condition_status.append(
                _normalize_status_conflict(previous_item, patch_item)
            )
    if merged_condition_status:
        merged["condition_status"] = merged_condition_status

    prev_tracking = merged.get("entity_tracking")
    patch_tracking = patch.get("entity_tracking")
    if isinstance(prev_tracking, list) or isinstance(patch_tracking, list):
        prev_tracking_map: dict[str, dict[str, Any]] = {}
        if isinstance(prev_tracking, list):
            for item in prev_tracking:
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("entity_id")
                if isinstance(entity_id, str) and entity_id.strip():
                    prev_tracking_map[entity_id] = copy.deepcopy(item)
        if isinstance(patch_tracking, list):
            for item in patch_tracking:
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("entity_id")
                if isinstance(entity_id, str) and entity_id.strip():
                    prev_tracking_map[entity_id] = copy.deepcopy(item)
        merged["entity_tracking"] = list(prev_tracking_map.values())

    prev_overall = merged.get("overall_progress")
    patch_overall = patch.get("overall_progress")
    if isinstance(prev_overall, dict) or isinstance(patch_overall, dict):
        overall = copy.deepcopy(prev_overall) if isinstance(prev_overall, dict) else {}
        if isinstance(patch_overall, dict):
            overall.update(copy.deepcopy(patch_overall))
        prev_stage = (
            prev_overall.get("stage") if isinstance(prev_overall, dict) else None
        )
        patch_stage = (
            patch_overall.get("stage") if isinstance(patch_overall, dict) else None
        )
        if (
            prev_stage == "completed"
            and patch_stage in {"not_started", "in_progress", "near_completion"}
        ):
            overall["stage"] = "uncertain"
            overall["blocking_factor"] = "insufficient_evidence"
        merged["overall_progress"] = overall

    for key, value in patch.items():
        if key in {"condition_status", "entity_tracking", "overall_progress"}:
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def should_terminate_from_task_state(
    task_state: dict[str, Any],
    episode_memory: dict[str, Any] | None = None,
) -> bool:
    """Return whether a parsed task state should trigger early termination."""
    decision = task_state["decision"]
    decision_ok = (
        decision["terminate"]
        and decision["status"] == "completed"
        and bool(decision["reason"])
    )
    if not decision_ok:
        return False

    if episode_memory is None:
        return True

    task_model = episode_memory.get("task_model", {})
    progress_state = episode_memory.get("progress_state", {})
    condition_ids = _extract_success_condition_ids(task_model)
    if not condition_ids:
        return True

    condition_map = _index_condition_status_items(progress_state.get("condition_status"))
    for cond_id in condition_ids:
        item = condition_map.get(cond_id)
        if not isinstance(item, dict):
            return False
        if item.get("status") != "confirmed_met":
            return False
    return True


def update_episode_memory(memory: dict[str, Any], task_state: dict[str, Any]) -> None:
    """Update episode memory from parsed bootstrap/keyframe state."""
    if not task_state.get("parse_ok", False):
        return

    parse_mode = task_state.get("parse_mode")
    if parse_mode == PARSE_MODE_BOOTSTRAP:
        memory["task_model"] = copy.deepcopy(task_state["task_model"])
        memory["previous_frame_state"] = copy.deepcopy(task_state["frame_state"])
        memory["progress_state"] = copy.deepcopy(task_state["progress_state"])
        return

    if parse_mode == PARSE_MODE_KEYFRAME:
        memory["previous_frame_state"] = copy.deepcopy(task_state["frame_state"])
        memory["task_model"] = merge_task_model_patch(
            task_model=memory.get("task_model", {}),
            task_model_patch=task_state.get("task_model_patch", {}),
        )
        memory["progress_state"] = merge_progress_state(
            previous_progress_state=memory.get("progress_state", {}),
            progress_state_patch=task_state["progress_state_patch"],
        )


def build_failed_task_state(
    error_message: str,
    parse_mode: str = PARSE_MODE_KEYFRAME,
) -> dict[str, Any]:
    """Return a non-terminating task-state record for VLM call failures."""
    if parse_mode == PARSE_MODE_BOOTSTRAP:
        task_state = empty_bootstrap_task_state(raw_text="", parse_ok=False)
    else:
        task_state = empty_keyframe_task_state(raw_text="", parse_ok=False)
    task_state["decision"]["reason"] = error_message.strip()
    return task_state


def build_bootstrap_vlm_prompt(
    base_prompt: str,
    task_name: str,
    prompt_version: str = "v1",
) -> str:
    """Build the bootstrap prompt to initialize generalized task state."""
    return (
        f"{base_prompt}\n\n"
        "阶段：bootstrap（首帧初始化）\n"
        f"prompt_version: {prompt_version}\n"
        f"任务描述：{task_name}\n"
        "请仅基于当前图像和任务描述，生成可跨帧复用的结构化任务模型与初始状态。\n"
        "你必须输出严格 JSON，且只能包含以下顶层字段：\n"
        "- task_model\n"
        "- frame_state\n"
        "- progress_state\n"
        "- decision\n"
        "decision 必须包含 terminate/status/reason，status 只能是 in_progress/completed/uncertain。\n"
        "注意：frame_state 只描述当前帧可见事实；不要输出自由散文。"
    )


def build_keyframe_vlm_prompt(
    base_prompt: str,
    task_name: str,
    memory: dict[str, Any],
    prompt_version: str = "v1",
) -> str:
    """Build the keyframe prompt with prior structured context."""
    task_model_json = json.dumps(
        memory.get("task_model", {}),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    previous_frame_json = json.dumps(
        memory.get("previous_frame_state", {}),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    progress_state_json = json.dumps(
        memory.get("progress_state", {}),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"{base_prompt}\n\n"
        "阶段：keyframe_update（关键帧更新）\n"
        f"prompt_version: {prompt_version}\n"
        f"任务描述：{task_name}\n"
        f"task_model: {task_model_json}\n"
        f"previous_frame_state: {previous_frame_json}\n"
        f"previous_progress_state: {progress_state_json}\n"
        "请基于当前图像与以上结构化上下文，按顺序完成：\n"
        "1) 输出当前帧结构化观察 frame_state；\n"
        "2) 输出与上一关键帧的差异 frame_delta；\n"
        "3) 可选输出 task_model_patch，仅允许补充 entities[*].visual_cues，不允许改动 success_conditions；\n"
        "4) 输出 progress_state_patch（本帧更新后的结构化进度）；\n"
        "5) 输出 decision（terminate/status/reason）。\n"
        "你必须输出严格 JSON，且只能包含以下顶层字段：\n"
        "- frame_state\n"
        "- frame_delta\n"
        "- task_model_patch（可选）\n"
        "- progress_state_patch\n"
        "- decision\n"
        "decision.status 只能是 in_progress/completed/uncertain。\n"
        "注意：不要复述长推理文本，字段应结构化、可机读。"
    )


def query_vlm_task_state(
    api_url: str,
    api_key: str | None,
    x_auth_token: str | None,
    model_name: str,
    prompt: str,
    image: Any,
    timeout: float,
    parse_mode: str = PARSE_MODE_KEYFRAME,
    previous_image: Any | None = None,
) -> dict[str, Any]:
    """Call the VLM endpoint and parse bootstrap/keyframe structured response."""
    image_data_url = encode_image_to_data_url(image)
    message_content: list[dict[str, Any]] = []
    if previous_image is not None:
        previous_image_data_url = encode_image_to_data_url(previous_image)
        message_content.extend(
            [
                {"type": "text", "text": "Reference previous keyframe image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": previous_image_data_url},
                },
                {"type": "text", "text": "Current keyframe image:"},
            ]
        )
    message_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    message_content.append({"type": "text", "text": prompt})
    request_body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": message_content,
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
                if parse_mode == PARSE_MODE_BOOTSTRAP:
                    return parse_vlm_bootstrap_state(response_payload)
                return parse_vlm_keyframe_state(response_payload)
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
