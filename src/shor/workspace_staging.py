from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class StagedWorkspace:
    root: Path
    input_root: Path
    data_root: Path
    agents_root: Path
    logs_root: Path
    library_dir: Path
    agent_path: Path
    logs_dir: Path
    output_root: Path
    tools_root: Path | None = None


def prepare_staged_workspace(
    *,
    workspace_root: Path,
    source_data_root: Path,
    agent_name: str,
    canonical_agents: list[str] | tuple[str, ...] | None = None,
    tools_root: Path | None = None,
) -> StagedWorkspace:
    """Create an isolated per-run workspace with copied inputs and empty output."""
    _reset_dir(workspace_root)

    input_root = workspace_root / "input"
    data_root = input_root / "data"
    agents_root = data_root / "agents"
    logs_root = data_root / "logs"
    library_dir = data_root / "library"
    output_root = workspace_root / "output"

    _copy_target_agent(source_data_root / "agents", agents_root, agent_name)
    _copy_canonical_agents(
        source_data_root / "agents",
        agents_root,
        canonical_agents or (),
        agent_name=agent_name,
    )
    _copy_tree(source_data_root / "logs" / agent_name, logs_root / agent_name)
    source_library = source_data_root / "library"
    if source_library.exists():
        _stage_library(source_library, library_dir)
    source_tools = _source_tools_path(source_data_root, tools_root)
    staged_tools_root = None
    if source_tools is not None:
        _copy_tree(source_tools, data_root / "tools")
        staged_tools_root = workspace_root / "tools"
        _copy_tree(source_tools, staged_tools_root)
    output_root.mkdir(parents=True, exist_ok=True)

    return StagedWorkspace(
        root=workspace_root,
        input_root=input_root,
        data_root=data_root,
        agents_root=agents_root,
        logs_root=logs_root,
        library_dir=library_dir,
        agent_path=agents_root / f"{agent_name}.py",
        logs_dir=logs_root / agent_name,
        output_root=output_root,
        tools_root=staged_tools_root,
    )


def copy_result_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def remove_staged_library(library_dir: Path) -> bool:
    if library_dir.is_symlink():
        library_dir.unlink()
        return True
    if not library_dir.exists():
        return False
    shutil.rmtree(library_dir)
    return True


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_target_agent(source: Path, destination: Path, agent_name: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    init_file = source / "__init__.py"
    if init_file.exists() and not init_file.is_symlink():
        shutil.copy2(init_file, destination / "__init__.py")

    agent_file = source / f"{agent_name}.py"
    if agent_file.is_symlink():
        raise OSError(f"target agent is a symlink and cannot be staged safely: {agent_file}")
    shutil.copy2(agent_file, destination / agent_file.name)


def _copy_canonical_agents(
    source: Path,
    destination: Path,
    canonical_agents: list[str] | tuple[str, ...],
    *,
    agent_name: str,
) -> None:
    for canonical in canonical_agents:
        if canonical == agent_name:
            continue
        agent_file = source / f"{canonical}.py"
        if agent_file.is_symlink():
            raise OSError(f"canonical agent is a symlink and cannot be staged safely: {agent_file}")
        if not agent_file.exists():
            raise FileNotFoundError(f"canonical agent is missing: {agent_file}")
        shutil.copy2(agent_file, destination / agent_file.name)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=_ignore_generated)


def _stage_library(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(source, destination)


def _source_tools_path(source_data_root: Path, tools_root: Path | None) -> Path | None:
    candidates = [tools_root, source_data_root / "tools"]
    for candidate in candidates:
        if candidate is not None and _has_python_tools(candidate):
            return candidate
    return None


def _has_python_tools(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.py"))


def _ignore_generated(dir_path: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    ignored.update(name for name in names if (Path(dir_path) / name).is_symlink())
    return ignored
