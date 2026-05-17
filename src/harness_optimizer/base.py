from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


BackboneStatus = Literal["ok", "error", "timeout", "limit_exceeded"]


@dataclass(frozen=True)
class BackboneLimits:
    step_limit: int | None = None
    cost_limit: float | None = None
    wall_timeout_seconds: int | None = None
    command_timeout_seconds: int | None = None


@dataclass(frozen=True)
class BackboneInstructions:
    system: str
    instance: str
    completion: str = ""
    native_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackboneTask:
    task_id: str
    workspace_root: Path
    trajectory_path: Path
    instructions: BackboneInstructions
    limits: BackboneLimits
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    template_vars: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneRunResult:
    status: BackboneStatus
    message: str
    trajectory_path: Path
    n_steps: int | None = None
    cost: float | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_stdout_path: Path | None = None
    raw_stderr_path: Path | None = None
    native_artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


class BackboneAgent(Protocol):
    name: str

    def prepare(self, task: BackboneTask) -> None:
        """Create directories, prompt files, or validate runtime prerequisites."""

    def run(self, task: BackboneTask) -> BackboneRunResult:
        """Execute the task and return normalized metadata."""

    def normalize_trajectory(self, task: BackboneTask, result: BackboneRunResult) -> None:
        """Ensure task.trajectory_path exists in the common envelope format."""
