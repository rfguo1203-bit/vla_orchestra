from __future__ import annotations

"""Contextual VLM prompt building, response parsing, and memory state helpers."""

import json
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any

from .io_and_video import encode_image_to_data_url
from .paths_and_config import DEFAULT_VLM_HISTORY_SIZE


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


def empty_vlm_task_state(
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


def strip_vlm_thinking(text: str) -> str:
    """Drop hidden reasoning content when the model emits a closing think tag."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Extract the last valid top-level JSON object from free-form model output."""
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates[-1] if candidates else None


def parse_vlm_task_state(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Parse an OpenAI-compatible VLM response into the minimal task-state schema."""
    content = extract_vlm_text_content(response_payload)
    cleaned_content = strip_vlm_thinking(content)
    parsed = extract_last_json_object(cleaned_content)
    if parsed is None:
        return empty_vlm_task_state(raw_text=content, parse_ok=False)

    frame_state = parsed.get("frame_state")
    task_memory = parsed.get("task_memory")
    decision = parsed.get("decision")
    if not isinstance(frame_state, dict):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(task_memory, dict):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(decision, dict):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)

    frame_summary = frame_state.get("summary")
    memory_summary = task_memory.get("state_summary")
    terminate = decision.get("terminate")
    status = decision.get("status")
    reason = decision.get("reason")
    if not isinstance(frame_summary, str):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(memory_summary, str):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(terminate, bool):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)
    if status not in {"in_progress", "completed", "uncertain"}:
        return empty_vlm_task_state(raw_text=content, parse_ok=False)
    if not isinstance(reason, str):
        return empty_vlm_task_state(raw_text=content, parse_ok=False)

    task_state = empty_vlm_task_state(raw_text=content, parse_ok=True)
    task_state["frame_state"]["summary"] = frame_summary.strip()
    task_state["task_memory"]["state_summary"] = memory_summary.strip()
    task_state["decision"]["terminate"] = terminate
    task_state["decision"]["status"] = status
    task_state["decision"]["reason"] = reason.strip()
    return task_state


def init_episode_memory(history_size: int = DEFAULT_VLM_HISTORY_SIZE) -> dict[str, Any]:
    """Create the minimal per-episode memory container for contextual VLM state."""
    return {
        "recent_history": [],
        "running_summary": "",
        "history_size": max(1, int(history_size)),
    }


def snapshot_episode_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the current episode memory state for logging/debugging."""
    return {
        "recent_history": list(memory.get("recent_history", [])),
        "running_summary": str(memory.get("running_summary", "")),
        "history_size": int(memory.get("history_size", DEFAULT_VLM_HISTORY_SIZE)),
    }


def should_terminate_from_task_state(task_state: dict[str, Any]) -> bool:
    """Return whether a parsed task state should trigger early termination."""
    decision = task_state["decision"]
    return (
        decision["terminate"]
        and decision["status"] == "completed"
        and bool(decision["reason"])
    )


def update_episode_memory(memory: dict[str, Any], task_state: dict[str, Any]) -> None:
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


def build_failed_task_state(error_message: str) -> dict[str, Any]:
    """Return a non-terminating task-state record for VLM call failures."""
    task_state = empty_vlm_task_state(raw_text="", parse_ok=False)
    task_state["decision"]["reason"] = error_message.strip()
    return task_state


def build_contextual_vlm_prompt(
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
        history_text = "- 无"

    running_summary = str(memory.get("running_summary", "")).strip() or "无"
    return (
        f"{base_prompt}\n\n"
        f"任务描述：{task_name}\n"
        "目标：判断这个任务在当前图像中是否已经完成。\n"
        f"运行中摘要：{running_summary}\n"
        "最近历史：\n"
        f"{history_text}\n"
        "只能基于提供的主视角图像做判断。"
    )


def query_vlm_task_state(
    api_url: str,
    api_key: str | None,
    x_auth_token: str | None,
    model_name: str,
    prompt: str,
    image: Any,
    timeout: float,
) -> dict[str, Any]:
    """Call the local OpenAI-compatible VLM endpoint and return the parsed task state."""
    image_data_url = encode_image_to_data_url(image)
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
                return parse_vlm_task_state(response_payload)
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
