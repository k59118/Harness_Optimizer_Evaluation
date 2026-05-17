from __future__ import annotations

from typing import Any

from harness_optimizer import BackboneInstructions
from harness_optimizer.prompting import build_instructions


def build_shor_instructions(
    shor_cfg: dict[str, Any],
    template_vars: dict[str, Any],
) -> BackboneInstructions:
    task_cfg = shor_cfg.get("task", {})
    system_template = task_cfg.get("system_template") or ""
    instance_template = task_cfg.get("instance_template") or ""
    completion_template = task_cfg.get("completion_template") or ""
    return build_instructions(
        system_template,
        instance_template,
        template_vars,
        completion_template=completion_template,
        native_overrides=task_cfg.get("native_overrides") or {},
    )
