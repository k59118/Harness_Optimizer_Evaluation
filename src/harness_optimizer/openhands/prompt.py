from __future__ import annotations

from pathlib import Path

from harness_optimizer.base import BackboneTask
from harness_optimizer.prompting import merged_prompt_text


def write_task_file(task: BackboneTask) -> Path:
    task_dir = task.trajectory_path.parent / ".backbone" / "openhands_cli"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_file = task_dir / "task.txt"
    task_file.write_text(_with_openhands_runtime_wrapper(merged_prompt_text(task.instructions)))
    return task_file


def _with_openhands_runtime_wrapper(task_text: str) -> str:
    return task_text.rstrip() + "\n\n" + _OPENHANDS_RUNTIME_WRAPPER + "\n"


_OPENHANDS_RUNTIME_WRAPPER = """
# OpenHands CLI Runtime

You are running in OpenHands headless CLI. Use the available terminal and file
tools to complete the instructions above. Keep file reads bounded for
large artifacts, respect the task's stated write boundaries, create any
required artifacts, run any requested validation, create the required result file, and then end with a
brief final message.

OpenHands treats any assistant message without a tool call as a final response.
Therefore, while work remains, do not write status updates, plans, or "I will
now ..." messages as plain assistant text. If you need to inspect, create, edit,
validate, or list files, immediately use an actual terminal or file-editor tool
action in that turn. Do not print a JSON command object as text; execute the
command through the terminal tool.

The terminal tool rejects commands that contain literal newline characters.
Run one shell command per terminal action. For multi-step inspections, use
`&&`, `;`, or a single `bash -lc '...'` command. For creating or editing
multi-line files, prefer the file editing tool when available; if using the
terminal, use a one-line command that writes the file without embedding literal
newlines in the terminal command.

Never generate a final message before every required artifact has been
written and validated.
""".strip()
