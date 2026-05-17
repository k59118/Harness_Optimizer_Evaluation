from __future__ import annotations

import json
from pathlib import Path


def _read_step_count(trajectory_path: Path) -> int:
    candidates = [
        trajectory_path.parent / ".backbone" / "openhands_cli" / "stdout.jsonl",
        trajectory_path.parent / ".backbone" / "claude_code_cli" / "stdout.jsonl",
        trajectory_path.parent / ".backbone" / "codex_cli" / "stdout.jsonl",
        trajectory_path,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        count = _extract_text_step_count(text)
        if count is not None:
            return count
    return 0


def _extract_text_step_count(text: str) -> int | None:
    try:
        return _extract_step_count(json.loads(text))
    except Exception:
        return _extract_jsonl_step_count(text)


def _extract_jsonl_step_count(text: str) -> int | None:
    events = _json_events_from_text(text)
    if events:
        count = sum(1 for event in events if _is_agent_step_event(event))
        return count if count else 0
    return None


def _json_events_from_text(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
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
            event, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(event, dict):
            events.append(event)
        cursor = end
    if events:
        return events

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except Exception:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _extract_step_count(data: object) -> int | None:
    if isinstance(data, dict):
        info = data.get("info")
        if isinstance(info, dict) and isinstance(info.get("n_steps"), int):
            return int(info["n_steps"])
        messages = data.get("messages")
        if isinstance(messages, list):
            assistant_messages = [
                m for m in messages
                if isinstance(m, dict) and str(m.get("role") or "").lower() == "assistant"
            ]
            return len(assistant_messages) if assistant_messages else len(messages)
        events = data.get("events")
        if isinstance(events, list):
            step_events = [
                e for e in events
                if isinstance(e, dict)
                and e.get("source") == "agent"
                and e.get("kind") in {"action", "message"}
            ]
            return len(step_events) if step_events else len(events)
    if isinstance(data, list):
        step_events = [item for item in data if isinstance(item, dict) and _is_agent_step_event(item)]
        return len(step_events) if step_events else len(data)
    return None


def _is_agent_step_event(event: dict[str, object]) -> bool:
    raw_type = str(event.get("type") or event.get("kind") or event.get("event_type") or "").lower()
    source = str(event.get("source") or event.get("role") or "").lower()
    item = event.get("item")
    if raw_type == "item.completed" and isinstance(item, dict):
        item_type = str(item.get("type") or "").lower()
        return item_type in {"agent_message", "command_execution"}
    if "observation" in raw_type or "error" in raw_type or raw_type in {"state", "status"}:
        return False
    if source in {"user", "environment", "tool", "runtime"}:
        return False
    if source in {"agent", "assistant"} and (
        "action" in raw_type
        or "message" in raw_type
        or "thought" in raw_type
        or event.get("action") is not None
        or event.get("command") is not None
        or event.get("llm_message") is not None
        or event.get("message") is not None
        or event.get("thought") is not None
    ):
        return True
    if "agent" in raw_type and (
        "action" in raw_type
        or "message" in raw_type
        or "thought" in raw_type
    ):
        return True
    if (
        "action" in raw_type
        or "message" in raw_type
        or "thought" in raw_type
        or event.get("action") is not None
        or event.get("command") is not None
        or event.get("llm_message") is not None
        or event.get("thought") is not None
    ):
        return True
    return False
