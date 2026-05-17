from __future__ import annotations

import copy
from typing import Any

from .base import BackboneAgent
from .errors import BackboneConfigError


BACKBONE_NAMES = {
    "openhands_cli",
    "claude_code_cli",
    "codex_cli",
}

DEFAULT_BACKBONE_CONFIGS: dict[str, dict[str, Any]] = {
    "openhands_cli": {
        "name": "openhands_cli",
        "binary": "openhands",
        "wall_timeout_seconds": 18000,
        "use_json": True,
        "override_with_envs": True,
        "env": {
            "OH_DISABLE_TELEMETRY": "1",
            "RUNTIME": "process",
        },
    },
    "claude_code_cli": {
        "name": "claude_code_cli",
        "binary": "claude",
        "output_format": "stream-json",
        "verbose": True,
        "bare": True,
        "permission_mode": "bypassPermissions",
        "use_system_prompt_file": True,
        "use_isolated_config_dir": True,
        "no_session_persistence": True,
        "setting_sources": "",
        "wall_timeout_seconds": 3600,
        "env": {
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_UPDATES": "1",
        },
        "settings": {},
    },
    "codex_cli": {
        "name": "codex_cli",
        "binary": "codex",
        "sandbox": "workspace-write",
        "ask_for_approval": "never",
        "skip_git_repo_check": True,
        "ephemeral": True,
        "wall_timeout_seconds": 3600,
        "env": {
            "DISABLE_UPDATES": "1",
        },
    },
}

def canonical_backbone_name(name: str) -> str:
    key = name.strip()
    if key not in BACKBONE_NAMES:
        known = ", ".join(sorted(BACKBONE_NAMES))
        raise BackboneConfigError(f"unknown backbone {name!r}; known names: {known}")
    return key


def default_backbone_config(name: str) -> dict[str, Any]:
    canonical = canonical_backbone_name(name)
    return copy.deepcopy(DEFAULT_BACKBONE_CONFIGS[canonical])


def load_backbone(name: str, config: dict[str, Any] | None = None) -> BackboneAgent:
    canonical = canonical_backbone_name(name)
    config = default_backbone_config(canonical) if config is None else config
    if canonical == "openhands_cli":
        from .openhands.adapter_cli import OpenHandsCliBackbone

        return OpenHandsCliBackbone(config)
    if canonical == "claude_code_cli":
        from .claude_code_cli.adapter_cli import ClaudeCodeCliBackbone

        return ClaudeCodeCliBackbone(config)
    if canonical == "codex_cli":
        from .codex_cli.adapter_cli import CodexCliBackbone

        return CodexCliBackbone(config)
    raise BackboneConfigError(f"no adapter registered for {canonical!r}")
