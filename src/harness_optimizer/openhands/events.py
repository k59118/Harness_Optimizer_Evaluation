from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness_optimizer.base import BackboneRunResult, BackboneTask
from harness_optimizer.io import write_json


def normalize_openhands_event(event: dict[str, Any]) -> dict[str, Any]:
    raw_type = str(event.get("type") or event.get("kind") or event.get("event_type") or "").lower()
    kind = _kind(raw_type, event)
    source = _source(kind, event)
    content = _content(event)
    return {
        "source": source,
        "kind": kind,
        "content": content,
        "native": event,
    }


def load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    text = _strip_ansi(path.read_text(errors="replace"))
    marker_events = _load_marked_json_events(text)
    if marker_events:
        return marker_events

    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            native = json.loads(line)
        except json.JSONDecodeError:
            events.append(
                {
                    "source": "runtime",
                    "kind": "raw",
                    "content": line,
                    "native": {"line_no": line_no, "parse_error": "invalid jsonl"},
                }
            )
            continue
        if isinstance(native, dict):
            events.append(normalize_openhands_event(native))
        else:
            events.append({"source": "runtime", "kind": "raw", "content": str(native), "native": native})
    return events


def extract_error_message(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    text = _strip_ansi(path.read_text(errors="replace"))
    for event in _load_marked_json_events(text):
        native = event.get("native", {})
        if isinstance(native, dict) and native.get("kind") == "ConversationErrorEvent":
            detail = native.get("detail") or native.get("message") or native.get("code")
            return str(detail) if detail else "OpenHands conversation error"
    for line in text.splitlines():
        if "ConversationErrorEvent" in line or "LLM Provider NOT provided" in line:
            return line.strip()
    return None


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _load_marked_json_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        marker = text.find("--JSON Event--", cursor)
        if marker < 0:
            break
        start = text.find("{", marker)
        if start < 0:
            break
        try:
            native, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            next_marker = text.find("--JSON Event--", start)
            raw_end = next_marker if next_marker >= 0 else len(text)
            events.append(
                {
                    "source": "runtime",
                    "kind": "raw",
                    "content": text[start:raw_end].strip(),
                    "native": {"parse_error": "invalid marked json event"},
                }
            )
            cursor = start + 1
            continue
        if isinstance(native, dict):
            events.append(normalize_openhands_event(native))
        else:
            events.append({"source": "runtime", "kind": "raw", "content": str(native), "native": native})
        cursor = end
    return events


def write_openhands_envelope(
    task: BackboneTask,
    result: BackboneRunResult,
    raw_stdout_path: Path | None,
    raw_format: str = "openhands-jsonl",
) -> None:
    events = load_jsonl_events(raw_stdout_path) if raw_stdout_path else []
    envelope = {
        "trajectory_format": "backbone-agent-1",
        "backbone": "openhands_cli",
        "status": result.status,
        "info": {
            "task_id": task.task_id,
            "model": result.native_artifacts.get("model"),
            "cost": result.cost,
            "n_steps": result.n_steps,
            "artifact_paths": {k: str(v) for k, v in task.artifact_paths.items()},
            "runtime_metadata": task.runtime_metadata,
        },
        "events": events,
        "native": {
            "raw_format": raw_format,
            "raw_path": str(raw_stdout_path) if raw_stdout_path else None,
            "stderr_path": str(result.raw_stderr_path) if result.raw_stderr_path else None,
            "artifacts": result.native_artifacts,
        },
    }
    write_json(task.trajectory_path, envelope)


def _kind(raw_type: str, event: dict[str, Any]) -> str:
    if "error" in raw_type or "error" in event:
        return "error"
    if raw_type in {"action", "agent_action"} or "action" in event or "command" in event:
        return "action"
    if raw_type in {"observation", "agent_observation"} or "observation" in event:
        return "observation"
    if (
        raw_type in {"message", "agent_message", "user_message", "messageevent"}
        or raw_type.endswith("messageevent")
        or "message" in event
        or "llm_message" in event
    ):
        return "message"
    if raw_type in {"state", "status"}:
        return "state"
    return "raw"


def _source(kind: str, event: dict[str, Any]) -> str:
    raw_source = str(event.get("source") or event.get("role") or "").lower()
    if raw_source in {"agent", "user", "environment", "tool", "runtime"}:
        return raw_source
    if kind == "observation":
        return "environment"
    if kind in {"action", "message"}:
        return "agent"
    return "runtime"


def _content(event: dict[str, Any]) -> str:
    llm_message = event.get("llm_message")
    if isinstance(llm_message, dict):
        text = _content_from_openhands_message(llm_message)
        if text:
            return text
    for key in ("content", "message", "thought", "command", "path", "result"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("action", "observation"):
        value = event.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nested in ("command", "content", "message", "path"):
                nested_value = value.get(nested)
                if isinstance(nested_value, str) and nested_value:
                    return nested_value
            return json.dumps(value, ensure_ascii=False)
    return json.dumps(event, ensure_ascii=False)


def _content_from_openhands_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""
