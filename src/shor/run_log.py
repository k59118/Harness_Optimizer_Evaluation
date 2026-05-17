from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_WRITE_LOCK = Lock()


def append_run_log(log_path: Path, *, task: str, event: str, **fields: Any) -> None:
    """Append a compact human-readable run event line."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        datetime.now(timezone.utc).isoformat(),
        f"task={_format_value(task)}",
        f"event={_format_value(event)}",
    ]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    line = " | ".join(parts) + "\n"
    with _WRITE_LOCK:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        value = round(value, 3)
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if not text or any(ch.isspace() for ch in text) or "|" in text:
        return repr(text)
    return text
