from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from harness_optimizer.base import BackboneRunResult, BackboneTask
from harness_optimizer.io import tail_text

from .events import extract_error_message, extract_result_metadata, write_claude_envelope


class ClaudeCodeCliBackbone:
    name = "claude_code_cli"
    model_name = "claude-sonnet-4-6"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.binary = str(config.get("binary", "claude"))

    def prepare(self, task: BackboneTask) -> None:
        _prepare_artifact_dirs(task)
        run_dir = _run_dir(task)
        claude_config = run_dir / "claude_config"
        claude_config.mkdir(parents=True, exist_ok=True)
        (run_dir / "task.txt").write_text(_user_prompt_text(task))
        (run_dir / "system_prompt.txt").write_text(task.instructions.system.strip() + "\n")
        settings = self.config.get("settings")
        if settings is None:
            settings = {}
        (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    def run(self, task: BackboneTask) -> BackboneRunResult:
        self.prepare(task)
        run_dir = _run_dir(task)
        task_file = run_dir / "task.txt"
        system_prompt_file = run_dir / "system_prompt.txt"
        settings_file = run_dir / "settings.json"
        stdout_path = run_dir / "stdout.jsonl"
        stderr_path = run_dir / "stderr.log"
        claude_config = run_dir / "claude_config"

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.config.get("env", {}).items()})
        env["CLAUDE_CONFIG_DIR"] = str(claude_config)
        env.setdefault("PYTHONUNBUFFERED", "1")
        _normalize_google_credentials_path(env, task.workspace_root)

        cmd = self._build_command(task, system_prompt_file, settings_file)
        native_artifacts = {
            "task_file": str(task_file),
            "system_prompt_file": str(system_prompt_file),
            "settings": str(settings_file),
            "claude_config": str(claude_config),
            "command": cmd,
            "model": self.model_name,
        }
        unsupported_limits = [
            name
            for name, value in (("command_timeout_seconds", task.limits.command_timeout_seconds),)
            if value is not None
        ]
        if unsupported_limits:
            native_artifacts["unsupported_limits"] = ",".join(unsupported_limits)

        auth_error = _auth_error(env)
        if auth_error:
            result = BackboneRunResult(
                status="error",
                message=auth_error,
                trajectory_path=task.trajectory_path,
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
                native_artifacts=native_artifacts,
                error=auth_error,
            )
            self.normalize_trajectory(task, result)
            return result

        timeout = task.limits.wall_timeout_seconds or self.config.get("wall_timeout_seconds")
        try:
            with task_file.open("r") as stdin, stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    cmd,
                    cwd=str(task.workspace_root),
                    env=env,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            result = BackboneRunResult(
                status="timeout",
                message=f"Claude Code CLI timed out after {timeout}s",
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
                message=f"Claude binary not found: {self.binary}",
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
        message = "" if status == "ok" else f"Claude exited {completed.returncode}: {tail_text(stderr_path)}"
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
            message = "Claude exited 0 but did not write required artifact(s): " + ", ".join(missing_artifacts)
        if status == "ok" and stdout_path.exists() and stdout_path.stat().st_size == 0:
            stderr_tail = tail_text(stderr_path)
            if "Traceback" in stderr_tail or "error" in stderr_tail.lower():
                status = "error"
                message = f"Claude produced no JSON events and stderr contains a runtime failure: {stderr_tail}"

        metadata = extract_result_metadata(stdout_path)
        if metadata.get("session_id"):
            native_artifacts["session_id"] = str(metadata["session_id"])
        result = BackboneRunResult(
            status=status,
            message=message,
            trajectory_path=task.trajectory_path,
            n_steps=metadata.get("n_steps"),
            cost=metadata.get("cost"),
            token_usage=metadata.get("token_usage", {}),
            raw_stdout_path=stdout_path,
            raw_stderr_path=stderr_path,
            native_artifacts=native_artifacts,
            error=message or None,
        )
        self.normalize_trajectory(task, result)
        return result

    def normalize_trajectory(self, task: BackboneTask, result: BackboneRunResult) -> None:
        write_claude_envelope(task, result, result.raw_stdout_path)

    def _build_command(self, task: BackboneTask, system_prompt_file: Path, settings_file: Path) -> list[str]:
        cmd = [self.binary]
        if self.config.get("bare", True):
            cmd.append("--bare")
        cmd.append("--print")
        cmd.extend(["--model", self.model_name])
        prompt_flag = "--system-prompt-file"
        if not self.config.get("use_system_prompt_file", True):
            prompt_flag = "--append-system-prompt-file"
        cmd.extend([prompt_flag, str(system_prompt_file)])
        setting_sources = self.config.get("setting_sources", "")
        if setting_sources is not None:
            cmd.extend(["--setting-sources", str(setting_sources)])
        cmd.extend(["--settings", str(settings_file)])
        permission_mode = self.config.get("permission_mode", "bypassPermissions")
        if permission_mode:
            cmd.extend(["--permission-mode", str(permission_mode)])
        output_format = self.config.get("output_format", "stream-json")
        if output_format:
            cmd.extend(["--output-format", str(output_format)])
        if self.config.get("verbose", True):
            cmd.append("--verbose")
        if task.limits.step_limit is not None:
            cmd.extend(["--max-turns", str(task.limits.step_limit)])
        if task.limits.cost_limit is not None:
            cmd.extend(["--max-budget-usd", str(task.limits.cost_limit)])
        if self.config.get("no_session_persistence", True):
            cmd.append("--no-session-persistence")
        for extra_arg in self.config.get("extra_args", []):
            cmd.append(str(extra_arg))
        return cmd


def _run_dir(task: BackboneTask) -> Path:
    return task.trajectory_path.parent / ".backbone" / "claude_code_cli"


def _prepare_artifact_dirs(task: BackboneTask) -> None:
    task.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    for path in task.artifact_paths.values():
        path = Path(path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _user_prompt_text(task: BackboneTask) -> str:
    parts = [task.instructions.instance.strip()]
    if task.instructions.completion:
        parts.extend(["", "# Completion Protocol", task.instructions.completion.strip()])
    return "\n".join(parts).strip() + "\n"


def _normalize_google_credentials_path(env: dict[str, str], workspace_root: Path) -> None:
    value = env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not value:
        return
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    env["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)


def _auth_error(env: dict[str, str]) -> str | None:
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"):
        return None
    if env.get("ANTHROPIC_BASE_URL") and (env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        return None
    if env.get("GOOGLE_APPLICATION_CREDENTIALS") or env.get("GOOGLE_CLOUD_PROJECT"):
        return None
    return (
        "Claude Code CLI auth env is missing; set ANTHROPIC_API_KEY, "
        "ANTHROPIC_AUTH_TOKEN, or provider env for Bedrock/Vertex"
    )
