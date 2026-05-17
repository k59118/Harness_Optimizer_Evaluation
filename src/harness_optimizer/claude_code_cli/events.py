from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness_optimizer.base import BackboneRunResult, BackboneTask
from harness_optimizer.io import write_json


def normalize_claude_event(event: dict[str, Any]) -> dict[str, Any]:
    raw_type = str(event.get("type") or event.get("kind") or event.get("event_type") or "").lower()
    kind = _kind(raw_type, event)
    return {
        "source": _source(kind, event),
        "kind": kind,
        "content": _content(event),
        "native": event,
    }


def load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    text = _strip_ansi(path.read_text(errors="replace"))
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
            events.append(normalize_claude_event(native))
        else:
            events.append({"source": "runtime", "kind": "raw", "content": str(native), "native": native})
    return events


def extract_error_message(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    for event in load_jsonl_events(path):
        if event["kind"] != "error":
            continue
        native = event.get("native", {})
        if isinstance(native, dict):
            for key in ("message", "error", "detail", "subtype"):
                value = native.get(key)
                if isinstance(value, str) and value:
                    return value
            result = native.get("result")
            if isinstance(result, dict):
                for key in ("message", "error", "detail"):
                    value = result.get(key)
                    if isinstance(value, str) and value:
                        return value
        return event.get("content") or "Claude Code CLI conversation error"
    return None


def extract_result_metadata(path: Path | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"token_usage": {}}
    if not path or not path.exists():
        return metadata
    for event in load_jsonl_events(path):
        native = event.get("native")
        if not isinstance(native, dict):
            continue
        if str(native.get("type") or "").lower() != "result":
            continue
        cost = native.get("total_cost_usd") or native.get("cost_usd")
        if isinstance(cost, int | float):
            metadata["cost"] = float(cost)
        turns = native.get("num_turns") or native.get("turns") or native.get("n_steps")
        if isinstance(turns, int):
            metadata["n_steps"] = turns
        session_id = native.get("session_id")
        if isinstance(session_id, str) and session_id:
            metadata["session_id"] = session_id
        usage = native.get("usage") or native.get("token_usage")
        if isinstance(usage, dict):
            metadata["token_usage"].update(_flatten_usage(usage))
    return metadata


def write_claude_envelope(
    task: BackboneTask,
    result: BackboneRunResult,
    raw_stdout_path: Path | None,
    raw_format: str = "claude-code-stream-json",
) -> None:
    events = load_jsonl_events(raw_stdout_path) if raw_stdout_path else []
    envelope = {
        "trajectory_format": "backbone-agent-1",
        "backbone": "claude_code_cli",
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


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _kind(raw_type: str, event: dict[str, Any]) -> str:
    if "error" in raw_type or event.get("is_error") is True:
        return "error"
    if raw_type == "assistant" and _message_has_block(event, "tool_use"):
        return "action"
    if raw_type == "user" and _message_has_block(event, "tool_result"):
        return "observation"
    if raw_type in {"tool_use", "tool_call", "action"} or "tool_use" in event:
        return "action"
    if raw_type in {"tool_result", "observation"} or "tool_result" in event:
        return "observation"
    if raw_type in {"assistant", "user", "message"} or "message" in event:
        return "message"
    if raw_type in {"system", "result", "state", "status"}:
        return "state"
    return "raw"


def _source(kind: str, event: dict[str, Any]) -> str:
    raw_source = str(event.get("source") or event.get("role") or event.get("type") or "").lower()
    if raw_source in {"assistant", "agent"}:
        return "agent"
    if raw_source == "user":
        return "user"
    if raw_source in {"tool", "environment"}:
        return "environment"
    if kind == "observation":
        return "environment"
    if kind in {"action", "message"}:
        return "agent"
    return "runtime"


def _content(event: dict[str, Any]) -> str:
    message = event.get("message")
    if isinstance(message, dict):
        text = _content_from_message(message)
        if text:
            return text
    for key in ("content", "text", "message", "delta", "name", "command", "error", "detail", "result"):
        value = event.get(key)
        text = _text_from_value(value)
        if text:
            return text
    return json.dumps(event, ensure_ascii=False)


def _content_from_message(message: dict[str, Any]) -> str:
    return _text_from_value(message.get("content"))


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") == "tool_use":
            name = value.get("name") or "tool_use"
            tool_input = value.get("input")
            if tool_input is None:
                return str(name)
            return f"{name}: {json.dumps(tool_input, ensure_ascii=False)}"
        if value.get("type") == "tool_result":
            return _text_from_value(value.get("content")) or json.dumps(value, ensure_ascii=False)
        for key in ("text", "content", "message", "name", "command", "error", "detail"):
            text = _text_from_value(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts = [_text_from_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _message_has_block(event: dict[str, Any], block_type: str) -> bool:
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == block_type for item in content)


def _flatten_usage(usage: dict[str, Any]) -> dict[str, int]:
    flattened: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, int):
            flattened[str(key)] = value
    return flattened
