from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from harness_optimizer.base import BackboneRunResult, BackboneTask
from harness_optimizer.io import tail_text

from .events import extract_error_message, extract_result_metadata, write_codex_envelope


class CodexCliBackbone:
    name = "codex_cli"
    model_name = "gpt-5.4"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.binary = str(config.get("binary", "codex"))

    def prepare(self, task: BackboneTask) -> None:
        _prepare_artifact_dirs(task)
        run_dir = _run_dir(task)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "task.txt").write_text(_prompt_text(task))

    def run(self, task: BackboneTask) -> BackboneRunResult:
        self.prepare(task)
        run_dir = _run_dir(task)
        task_file = run_dir / "task.txt"
        stdout_path = run_dir / "stdout.jsonl"
        stderr_path = run_dir / "stderr.log"

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.config.get("env", {}).items()})
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.config.get("codex_home"):
            env["CODEX_HOME"] = str(self.config["codex_home"])

        cmd = self._build_command(task)
        native_artifacts = {
            "task_file": str(task_file),
            "command": cmd,
            "model": self.model_name,
        }
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

        timeout = task.limits.wall_timeout_seconds or self.config.get("wall_timeout_seconds")
        try:
            with task_file.open("r") as stdin, stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                returncode = _run_process_group(
                    cmd,
                    cwd=str(task.workspace_root),
                    env=env,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            result = BackboneRunResult(
                status="timeout",
                message=f"Codex CLI timed out after {timeout}s",
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
                message=f"Codex binary not found: {self.binary}",
                trajectory_path=task.trajectory_path,
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
                native_artifacts=native_artifacts,
                error=str(e),
            )
            self.normalize_trajectory(task, result)
            return result

        status = "ok" if returncode == 0 else "error"
        conversation_error = extract_error_message(stdout_path)
        message = "" if status == "ok" else _exit_message(returncode, stderr_path)
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
            message = "Codex exited 0 but did not write required artifact(s): " + ", ".join(missing_artifacts)
        if status == "ok" and stdout_path.exists() and stdout_path.stat().st_size == 0:
            stderr_tail = tail_text(stderr_path)
            if "Traceback" in stderr_tail or "error" in stderr_tail.lower():
                status = "error"
                message = f"Codex produced no JSON events and stderr contains a runtime failure: {stderr_tail}"

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
        write_codex_envelope(task, result, result.raw_stdout_path)

    def _build_command(self, task: BackboneTask) -> list[str]:
        cmd = [self.binary]
        approval = self.config.get("ask_for_approval", "never")
        if approval:
            cmd.extend(["--ask-for-approval", str(approval)])
        cmd.extend(["exec", "--json", "--model", self.model_name, "--cd", str(task.workspace_root)])
        sandbox = self.config.get("sandbox", "workspace-write")
        if sandbox:
            cmd.extend(["--sandbox", str(sandbox)])
        if self.config.get("skip_git_repo_check", True):
            cmd.append("--skip-git-repo-check")
        if self.config.get("ephemeral", True):
            cmd.append("--ephemeral")
        if self.config.get("ignore_user_config", False):
            cmd.append("--ignore-user-config")
        if self.config.get("ignore_rules", False):
            cmd.append("--ignore-rules")
        if self.config.get("dangerously_bypass_approvals_and_sandbox", False):
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        if self.config.get("search", False):
            cmd.append("--search")
        for key, value in self.config.get("config_overrides", {}).items():
            cmd.extend(["--config", f"{key}={value}"])
        for extra_arg in self.config.get("extra_args", []):
            cmd.append(str(extra_arg))
        cmd.append("-")
        return cmd


def _exit_message(returncode: int, stderr_path: Path) -> str:
    if returncode < 0:
        signum = -returncode
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"signal {signum}"
        return f"Codex terminated by {signal_name}: {tail_text(stderr_path)}"
    return f"Codex exited {returncode}: {tail_text(stderr_path)}"


def _run_dir(task: BackboneTask) -> Path:
    return task.trajectory_path.parent / ".backbone" / "codex_cli"


def _prepare_artifact_dirs(task: BackboneTask) -> None:
    task.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    for path in task.artifact_paths.values():
        path = Path(path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _run_process_group(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdin: Any,
    stdout: Any,
    stderr: Any,
    text: bool,
    timeout: int | float | None,
) -> int:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise
    except KeyboardInterrupt:
        _terminate_process_group(process)
        raise


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    while process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            time.sleep(0.1)


def _prompt_text(task: BackboneTask) -> str:
    parts = [
        "# System Instructions",
        task.instructions.system.strip(),
        "",
        "# Task",
        task.instructions.instance.strip(),
    ]
    if task.instructions.completion:
        parts.extend(["", "# Completion Protocol", task.instructions.completion.strip()])
    return "\n".join(parts).strip() + "\n"
