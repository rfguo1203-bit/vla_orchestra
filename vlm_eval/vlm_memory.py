from __future__ import annotations

"""Natural-language VLM prompts, parsing, and episode-state helpers."""

import copy
import json
import re
import socket
import ssl
import urllib.parse
import urllib.error
import urllib.request
from typing import Any

from .io_and_video import encode_image_to_data_url

VALID_DECISION_STATUS = {"in_progress", "completed", "uncertain"}
PARSE_MODE_BOOTSTRAP = "bootstrap"
PARSE_MODE_KEYFRAME = "keyframe"


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


def extract_text_only_message_content(
    message_content: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Keep only text parts from multimodal user content for chat history."""
    text_only_parts: list[dict[str, str]] = []
    for content_item in message_content:
        if content_item.get("type") != "text":
            continue
        text = content_item.get("text")
        if isinstance(text, str) and text.strip():
            text_only_parts.append({"type": "text", "text": text})
    return text_only_parts


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
    return {
        "parse_mode": PARSE_MODE_BOOTSTRAP,
        "task_profile": "",
        "frame_summary": "",
        "progress_summary": "",
        "decision": empty_decision(),
        "raw_text": raw_text,
        "parse_ok": parse_ok,
    }


def empty_keyframe_task_state(
    raw_text: str = "",
    parse_ok: bool = False,
) -> dict[str, Any]:
    return {
        "parse_mode": PARSE_MODE_KEYFRAME,
        "frame_summary": "",
        "change_summary": "",
        "progress_summary": "",
        "decision": empty_decision(),
        "raw_text": raw_text,
        "parse_ok": parse_ok,
    }


def strip_vlm_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_json_from_fenced_block(text: str) -> dict[str, Any] | None:
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
    objects = extract_top_level_json_objects(text)
    return objects[-1] if objects else None


def parse_decision(decision: Any) -> dict[str, Any] | None:
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


def _normalize_text_field(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _extract_legacy_text(parsed: dict[str, Any], key: str) -> str:
    item = parsed.get(key)
    if isinstance(item, dict):
        for candidate_key in ("summary", "state_summary"):
            value = item.get(candidate_key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def parse_vlm_bootstrap_state(response_payload: dict[str, Any]) -> dict[str, Any]:
    content = extract_vlm_text_content(response_payload)
    cleaned_content = strip_vlm_thinking(content)
    parsed = extract_json_from_fenced_block(cleaned_content)
    if parsed is None:
        parsed = extract_last_top_level_json_object(cleaned_content)
    if parsed is None:
        return empty_bootstrap_task_state(raw_text=content, parse_ok=False)

    task_profile = _normalize_text_field(parsed.get("task_profile"))
    frame_summary = _normalize_text_field(parsed.get("frame_summary"))
    progress_summary = _normalize_text_field(parsed.get("progress_summary"))

    if not frame_summary:
        frame_summary = _extract_legacy_text(parsed, "frame_state")
    if not progress_summary:
        progress_summary = _extract_legacy_text(parsed, "progress_state")
    if not progress_summary:
        progress_summary = _extract_legacy_text(parsed, "task_memory")

    normalized_decision = parse_decision(parsed.get("decision"))
    if normalized_decision is None:
        normalized_decision = empty_decision()

    if not frame_summary:
        return empty_bootstrap_task_state(raw_text=content, parse_ok=False)

    task_state = empty_bootstrap_task_state(raw_text=content, parse_ok=True)
    task_state["task_profile"] = task_profile
    task_state["frame_summary"] = frame_summary
    task_state["progress_summary"] = progress_summary
    task_state["decision"] = normalized_decision
    return task_state


def parse_vlm_keyframe_state(response_payload: dict[str, Any]) -> dict[str, Any]:
    content = extract_vlm_text_content(response_payload)
    cleaned_content = strip_vlm_thinking(content)
    parsed = extract_json_from_fenced_block(cleaned_content)
    if parsed is None:
        parsed = extract_last_top_level_json_object(cleaned_content)
    if parsed is None:
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)

    frame_summary = _normalize_text_field(parsed.get("frame_summary"))
    change_summary = _normalize_text_field(parsed.get("change_summary"))
    progress_summary = _normalize_text_field(parsed.get("progress_summary"))

    if not frame_summary:
        frame_summary = _extract_legacy_text(parsed, "frame_state")
    if not progress_summary:
        progress_summary = _extract_legacy_text(parsed, "progress_state_patch")
    if not progress_summary:
        progress_summary = _extract_legacy_text(parsed, "progress_state")
    if not progress_summary:
        progress_summary = _extract_legacy_text(parsed, "task_memory")
    if not change_summary:
        change_summary = _extract_legacy_text(parsed, "frame_delta")

    normalized_decision = parse_decision(parsed.get("decision"))
    if normalized_decision is None:
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)
    if not frame_summary:
        return empty_keyframe_task_state(raw_text=content, parse_ok=False)

    task_state = empty_keyframe_task_state(raw_text=content, parse_ok=True)
    task_state["frame_summary"] = frame_summary
    task_state["change_summary"] = change_summary
    task_state["progress_summary"] = progress_summary
    task_state["decision"] = normalized_decision
    return task_state


def init_episode_memory() -> dict[str, Any]:
    return {
        "task_profile": "",
        "running_summary": "",
    }


def init_vlm_conversation() -> list[dict[str, Any]]:
    """Initialize one VLM chat session for a single episode."""
    return []


def build_vlm_headers(
    api_key: str | None,
    x_auth_token: str | None,
    request_id: str | None = None,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if x_auth_token:
        headers["x-auth-token"] = x_auth_token
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def build_reset_caches_url(api_url: str) -> str:
    parsed = urllib.parse.urlsplit(api_url)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            "/internal/reset-caches",
            "",
            "",
        )
    )


def reset_vlm_caches(
    api_url: str,
    api_key: str | None,
    x_auth_token: str | None,
    timeout: float,
    reset_prefix_cache: bool = True,
    reset_mm_cache: bool = True,
    reset_running_requests: bool = False,
    request_id: str | None = None,
) -> dict[str, Any]:
    reset_url = build_reset_caches_url(api_url)
    request_body = {
        "reset_prefix_cache": reset_prefix_cache,
        "reset_mm_cache": reset_mm_cache,
        "reset_running_requests": reset_running_requests,
    }
    request = urllib.request.Request(
        reset_url,
        data=json.dumps(request_body).encode("utf-8"),
        headers=build_vlm_headers(
            api_key=api_key,
            x_auth_token=x_auth_token,
            request_id=request_id,
        ),
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=ssl.create_default_context(),
        ) as response:
            response_text = response.read().decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"VLM cache reset failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"VLM cache reset failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TimeoutError(f"VLM cache reset timed out after {timeout}s.") from exc
    except socket.timeout as exc:
        raise TimeoutError(f"VLM cache reset timed out after {timeout}s.") from exc

    if not response_text:
        return {}
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw_text": response_text}
    return parsed if isinstance(parsed, dict) else {"response": parsed}


def snapshot_episode_memory(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_profile": str(memory.get("task_profile", "")),
        "running_summary": str(memory.get("running_summary", "")),
    }


def should_terminate_from_task_state(
    task_state: dict[str, Any],
    episode_memory: dict[str, Any] | None = None,
) -> bool:
    decision = task_state["decision"]
    return (
        decision["terminate"]
        and decision["status"] == "completed"
        and bool(decision["reason"])
    )


def update_episode_memory(memory: dict[str, Any], task_state: dict[str, Any]) -> None:
    if not task_state.get("parse_ok", False):
        return

    parse_mode = task_state.get("parse_mode")
    if parse_mode == PARSE_MODE_BOOTSTRAP:
        task_profile = task_state.get("task_profile", "")
        progress_summary = task_state.get("progress_summary", "")
        if isinstance(task_profile, str) and task_profile.strip():
            memory["task_profile"] = task_profile.strip()
        if isinstance(progress_summary, str) and progress_summary.strip():
            memory["running_summary"] = progress_summary.strip()
        return

    if parse_mode == PARSE_MODE_KEYFRAME:
        progress_summary = task_state.get("progress_summary", "")
        if isinstance(progress_summary, str) and progress_summary.strip():
            memory["running_summary"] = progress_summary.strip()


def build_failed_task_state(
    error_message: str,
    parse_mode: str = PARSE_MODE_KEYFRAME,
) -> dict[str, Any]:
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
    return (
        f"{base_prompt}\n\n"
        "当前阶段：bootstrap（首帧初始化）\n"
        f"本次任务的整体任务描述是：{task_name}\n\n"
        "首帧职责：\n"
        "1) 建立任务语义基线（task_profile），明确完成标准；\n"
        "2) 给出首帧可见事实摘要（frame_summary）；\n"
        "3) 给出首帧任务进展总结（progress_summary）。\n\n"
        "输出约束：\n"
        "- 字段内容使用自然语言短句，语义明确；\n"
        "- decision.reason 必须与当前帧可见证据一致；\n"
        "- 仅当当前帧有明确视觉证据满足任务目标时，才可 terminate=true 且 status=completed。\n\n"
        "输出格式（严格）：\n"
        "你必须输出严格 JSON，且只能包含以下顶层字段：\n"
        "- task_profile：自然语言描述本次任务机器人需要做哪些操作，需要关注哪些物体，做任务步骤和细节拆解\n"
        "- frame_summary：针对角色目标对图像状态进行总结\n"
        "- progress_summary：当前任务进度\n"
        "- decision：（包含 terminate/status/reason）\n"
        "decision.status 只能是 in_progress/completed/uncertain。"
    )


def build_keyframe_vlm_prompt(
    base_prompt: str,
    task_name: str,
    frame_interval_seconds: float | None = None,
    prompt_version: str = "v1",
) -> str:
    interval_text = (
        f"{frame_interval_seconds:.2f} 秒"
        if isinstance(frame_interval_seconds, (int, float))
        and frame_interval_seconds > 0
        else "未知"
    )
    return (
        f"{base_prompt}\n\n"
        "阶段当前：keyframe_update（关键帧增量更新）\n"
        "- 这是下一帧图像；\n"
        f"- 与上一帧间隔约为 {interval_text}；\n"
        f"- 当前正在进行的任务是：{task_name}\n\n"
        "你需要根据你的角色目标，基于当前帧的图像，进行如下总结：\n"
        "1) frame_summary：针对角色目标对图像状态进行总结；\n"
        "2) change_summary：相比上一帧的图像状态，本帧图像发生了哪些变化；\n"
        "3) progress_summary：基于当前帧判断任务进展；\n"
        "4) decision：进行完成状态判定。\n\n"
        "输出格式（严格）：\n"
        "你必须输出严格 JSON，且只能包含以下顶层字段：\n"
        "- frame_summary\n"
        "- change_summary\n"
        "- progress_summary\n"
        "- decision（包含 terminate/status/reason）\n"
        "decision.status 只能是 in_progress/completed/uncertain。"
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
    conversation_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    image_data_url = encode_image_to_data_url(image)
    message_content: list[dict[str, Any]] = []
    if previous_image is not None:
        previous_image_data_url = encode_image_to_data_url(previous_image)
        message_content.extend(
            [
                {"type": "text", "text": "Reference previous keyframe image:"},
                {"type": "image_url", "image_url": {"url": previous_image_data_url}},
                {"type": "text", "text": "Current keyframe image:"},
            ]
        )
    message_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    message_content.append({"type": "text", "text": prompt})
    request_messages: list[dict[str, Any]]
    if conversation_messages is None:
        request_messages = [
            {
                "role": "user",
                "content": message_content,
            }
        ]
    else:
        request_messages = copy.deepcopy(conversation_messages)
        request_messages.append(
            {
                "role": "user",
                "content": message_content,
            }
        )

    request_body = {
        "model": model_name,
        "messages": request_messages,
        "stream": False,
        "max_tokens": 10240,
        "temperature": 0.1,
    }
    headers = build_vlm_headers(api_key=api_key, x_auth_token=x_auth_token)
    request_bytes = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=request_bytes,
        headers=headers,
        method="POST",
    )
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
                    task_state = parse_vlm_bootstrap_state(response_payload)
                else:
                    task_state = parse_vlm_keyframe_state(response_payload)

                if conversation_messages is not None:
                    text_only_user_content = extract_text_only_message_content(
                        message_content
                    )
                    conversation_messages.append(
                        {
                            "role": "user",
                            "content": text_only_user_content,
                        }
                    )
                    conversation_messages.append(
                        {
                            "role": "assistant",
                            "content": task_state["raw_text"],
                        }
                    )
                return task_state
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
