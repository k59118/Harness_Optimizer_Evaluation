from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from harness_optimizer.base import BackboneRunResult, BackboneTask
from harness_optimizer.io import tail_text

from .events import extract_error_message, write_openhands_envelope
from .prompt import write_task_file


class OpenHandsCliBackbone:
    name = "openhands_cli"
    model_name = "openai/gpt-5-mini"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.binary = str(config.get("binary", "openhands"))

    def prepare(self, task: BackboneTask) -> None:
        _prepare_artifact_dirs(task)
        write_task_file(task)

    def run(self, task: BackboneTask) -> BackboneRunResult:
        self.prepare(task)
        run_dir = task.trajectory_path.parent / ".backbone" / "openhands_cli"
        run_dir.mkdir(parents=True, exist_ok=True)
        task_file = run_dir / "task.txt"
        stdout_path = run_dir / "stdout.jsonl"
        stderr_path = run_dir / "stderr.log"

        cmd = [
            self.binary,
            "--headless",
            "--file",
            str(task_file),
        ]
        if self.config.get("use_json", True):
            cmd.append("--json")
        if self.config.get("override_with_envs", True):
            cmd.append("--override-with-envs")

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.config.get("env", {}).items()})
        env["LLM_MODEL"] = self.model_name
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("OH_PERSISTENCE_DIR", str(run_dir / "persistence"))
        env.setdefault("WORKSPACE_BASE", str(task.workspace_root))

        timeout = task.limits.wall_timeout_seconds or self.config.get("wall_timeout_seconds")
        native_artifacts = {"task_file": str(task_file), "model": self.model_name}
        missing_env = _missing_required_env(env) if self.config.get("override_with_envs", True) else []
        if missing_env:
            message = (
                "OpenHands CLI requires environment variable(s) when --override-with-envs is enabled: "
                + ", ".join(missing_env)
            )
            stderr_path.write_text(message + "\n")
            result = BackboneRunResult(
                status="error",
                message=message,
                trajectory_path=task.trajectory_path,
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
                native_artifacts={**native_artifacts, "missing_env": ",".join(missing_env)},
                error=message,
            )
            self.normalize_trajectory(task, result)
            return result

        unsupported_limits = [
            name
            for name, value in (
                ("step_limit", task.limits.step_limit),
                ("cost_limit", task.limits.cost_limit),
                ("command_timeout_seconds", task.limits.command_timeout_seconds),
            )
            if value is not None
        ]
        if unsupported_limits:
            native_artifacts["unsupported_limits"] = ",".join(unsupported_limits)

        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    cmd,
                    cwd=str(task.workspace_root),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            result = BackboneRunResult(
                status="timeout",
                message=f"OpenHands CLI timed out after {timeout}s",
                trajectory_path=task.trajectory_path,
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
                native_artifacts=native_artifacts,
                error="timeout",
            )
            self.normalize_trajectory(task, result)
            return result
        except FileNotFoundError as e:
            result = BackboneRunResult(
                status="error",
                message=f"OpenHands binary not found: {self.binary}",
                trajectory_path=task.trajectory_path,
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
                native_artifacts=native_artifacts,
                error=str(e),
            )
            self.normalize_trajectory(task, result)
            return result

        status = "ok" if completed.returncode == 0 else "error"
        conversation_error = extract_error_message(stdout_path)
        message = "" if status == "ok" else f"OpenHands exited {completed.returncode}: {tail_text(stderr_path)}"
        if conversation_error:
            status = "error"
            message = conversation_error
        missing_artifacts = [
            name
            for name, path in task.artifact_paths.items()
            if Path(path) != task.trajectory_path and not Path(path).exists()
        ]
        if status == "ok" and missing_artifacts:
            status = "error"
            message = "OpenHands exited 0 but did not write required artifact(s): " + ", ".join(missing_artifacts)
        if status == "ok" and stdout_path.exists() and stdout_path.stat().st_size == 0:
            stderr_tail = tail_text(stderr_path)
            if "Traceback" in stderr_tail or "KeyboardInterrupt" in stderr_tail:
                status = "error"
                message = f"OpenHands produced no JSON events and stderr contains a runtime failure: {stderr_tail}"

        result = BackboneRunResult(
            status=status,
            message=message,
            trajectory_path=task.trajectory_path,
            raw_stdout_path=stdout_path,
            raw_stderr_path=stderr_path,
            native_artifacts=native_artifacts,
            error=message or None,
        )
        self.normalize_trajectory(task, result)
        return result

    def normalize_trajectory(self, task: BackboneTask, result: BackboneRunResult) -> None:
        write_openhands_envelope(task, result, result.raw_stdout_path)


def _prepare_artifact_dirs(task: BackboneTask) -> None:
    task.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    for path in task.artifact_paths.values():
        path = Path(path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _missing_required_env(env: dict[str, str]) -> list[str]:
    return [name for name in ("LLM_API_KEY", "LLM_BASE_URL") if not str(env.get(name) or "").strip()]

