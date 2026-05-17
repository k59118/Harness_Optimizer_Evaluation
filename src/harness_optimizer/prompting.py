from __future__ import annotations

import platform
import re
from typing import Any

from .base import BackboneInstructions

_VAR_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def system_info_vars() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def render_template(template: str, vars_: dict[str, Any]) -> str:
    all_vars = {**system_info_vars(), **vars_}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in all_vars:
            raise KeyError(f"missing template variable: {key}")
        return str(all_vars[key])

    return _VAR_RE.sub(repl, template)


def build_instructions(
    system_template: str,
    instance_template: str,
    template_vars: dict[str, Any],
    *,
    completion_template: str = "",
    native_overrides: dict[str, Any] | None = None,
) -> BackboneInstructions:
    if not system_template or not instance_template:
        raise ValueError("system_template and instance_template are required")

    return BackboneInstructions(
        system=render_template(system_template, template_vars),
        instance=render_template(instance_template, template_vars),
        completion=render_template(completion_template, template_vars) if completion_template else "",
        native_overrides=native_overrides or {},
    )


def merged_prompt_text(instructions: BackboneInstructions) -> str:
    parts = [
        "# System Instructions",
        instructions.system,
        "# Task",
        instructions.instance,
    ]
    if instructions.completion:
        parts.extend(["# Completion Protocol", instructions.completion])
    return "\n\n".join(parts).strip() + "\n"
