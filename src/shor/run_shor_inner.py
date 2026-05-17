#!/usr/bin/env python3
"""Run a backbone agent as a SHOR investigator.

Domain-aware: works for swe / spider / gaia / facts / tau2 (any harness
domain whose layout matches `data/<domain>/{agents,logs}/...`).

Single agent:
  python run_shor.py --agent <name> --domain <domain> [--optimizer openhands_cli]

Multiple agents (comma list):
  python run_shor.py --agent a,b,c --domain swe

All registered agents in the domain:
  python run_shor.py --agent all --domain swe

Limit to the first N selected agents:
  python run_shor.py --agent all --domain swe --limit 10

Parallel (N agents at once):
  python run_shor.py --agent all --domain swe --parallel 5

Skip already-done:
  python run_shor.py --agent all --domain swe --skip-done

Writes:
  result/<harness_optimizer>/<domain>/<agent>/ranking.json
  result/<harness_optimizer>/<domain>/<agent>/trajectory.json
  result/<harness_optimizer>/<domain>/<agent>/meta.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import shlex
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent

DEFAULT_CONFIG = REPO_ROOT / "config" / "shor.yaml"
DOMAIN_CONFIG_DIR = REPO_ROOT / "config" / "domains"
DEFAULT_TARGETS_PATH = PROJECT_ROOT / "shor_final.json"

from harness_optimizer import (
    BackboneLimits,
    BackboneTask,
    default_backbone_config,
    load_backbone,
)
from harness_optimizer.registry import canonical_backbone_name
from shor.prompting import build_shor_instructions
from shor.run_log import append_run_log
from shor.schemas import validate_shor_output
from shor.step_count import _read_step_count
from shor.workspace_staging import prepare_staged_workspace

RunResult = tuple[str, str, str]


def _print_exception_traceback(exc: BaseException) -> None:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def load_domain_config(domain: str) -> dict:
    p = DOMAIN_CONFIG_DIR / f"{domain}.yaml"
    if not p.exists():
        available = sorted(f.stem for f in DOMAIN_CONFIG_DIR.glob("*.yaml"))
        raise SystemExit(f"unknown domain {domain!r}; have: {available}")
    return yaml.safe_load(p.read_text())


def available_domains() -> list[str]:
    return sorted(p.stem for p in DOMAIN_CONFIG_DIR.glob("*.yaml") if p.is_file())


def load_registered_agents(path: Path = DEFAULT_TARGETS_PATH) -> dict[str, list[str]]:
    if not path.exists():
        raise SystemExit(f"SHOR target file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"SHOR target file must be a JSON list: {path}")

    targets: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise SystemExit(f"SHOR target row {i} must be an object: {path}")
        domain = row.get("domain")
        harness_id = row.get("harness_id")
        if not isinstance(domain, str) or not domain:
            raise SystemExit(f"SHOR target row {i} missing domain: {path}")
        if not isinstance(harness_id, str) or not harness_id:
            raise SystemExit(f"SHOR target row {i} missing harness_id: {path}")
        domain_seen = seen.setdefault(domain, set())
        if harness_id in domain_seen:
            continue
        domain_seen.add(harness_id)
        targets.setdefault(domain, []).append(harness_id)
    return targets


def resolve_agent_list(
    spec: str,
    *,
    domain: str,
    registered_agents: dict[str, list[str]],
) -> list[str]:
    domain_agents = registered_agents.get(domain, [])
    if not domain_agents:
        raise SystemExit(
            f"no agents registered in {DEFAULT_TARGETS_PATH.name} for domain {domain!r}"
        )
    if spec == "all":
        return domain_agents
    if "," in spec:
        requested = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        requested = [spec]

    registered = set(domain_agents)
    missing = [agent for agent in requested if agent not in registered]
    if missing:
        raise SystemExit(
            f"agent(s) not registered in {DEFAULT_TARGETS_PATH.name} "
            f"for domain {domain!r}: {', '.join(missing)}"
        )
    return requested


def apply_agent_limit(plans: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return plans

    remaining = limit
    limited_plans: list[dict[str, Any]] = []
    for plan in plans:
        if remaining <= 0:
            break

        selected_agents = plan["agents"][:remaining]
        if not selected_agents:
            continue

        limited_plan = dict(plan)
        limited_plan["agents"] = selected_agents
        limited_plans.append(limited_plan)
        remaining -= len(selected_agents)

    return limited_plans


def resolve_data_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_dir_for(
    domain: str,
    agent_name: str,
    harness_optimizer_name: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "result"
        / harness_optimizer_name
        / domain
        / agent_name
    )


def run_log_path_for(domain: str, harness_optimizer_name: str) -> Path:
    return PROJECT_ROOT / "result" / harness_optimizer_name / domain / "run.log"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_meta(
    *,
    task: str,
    started_at: str,
    ended_at: str,
    elapsed_seconds: float,
    command_line: str,
    domain: str,
    agent_name: str,
    backbone_name: str,
    data_root: Path,
    config_path: Path,
    backbone_config_path: str,
    optimizer_timeout: int | None,
    parallel: int,
    agent_limit: int | None,
    skip_done: bool,
    limits: BackboneLimits,
    status: str,
    cost: float | None,
    token_usage: dict[str, int] | None = None,
    llm_call_count: int | None = None,
) -> dict[str, Any]:
    meta = {
        "task": task,
        "status": status,
        "cost": cost,
        "token_usage": token_usage or {},
        "llm_call_count": llm_call_count,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "command": command_line,
        "config": {
            "domain": domain,
            "agent": agent_name,
            "backbone": backbone_name,
            "harness_optimizer": backbone_name,
            "data_root": str(data_root),
            "config_path": str(config_path),
            "backbone_config_path": backbone_config_path,
            "optimizer_timeout": optimizer_timeout,
            "parallel": parallel,
            "agent_limit": agent_limit,
            "skip_done": skip_done,
            "limits": {
                "step_limit": limits.step_limit,
                "cost_limit": limits.cost_limit,
                "wall_timeout_seconds": limits.wall_timeout_seconds,
                "command_timeout_seconds": limits.command_timeout_seconds,
            },
        },
    }
    return meta


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_backbone_task(
    agent_name: str,
    dom: dict,
    data_root: Path,
    shor_cfg: dict,
    backbone_cfg: dict,
    output_json_path: Path,
    trajectory_path: Path,
    optimizer_timeout: int | None,
    *,
    workspace_root: Path | None = None,
) -> BackboneTask:
    agents_root = data_root / "agents"
    logs_root = data_root / "logs"
    agent_path = agents_root / f"{agent_name}.py"
    logs_dir = logs_root / agent_name
    template_vars = {
        "agent_name": agent_name,
        "agent_path": str(agent_path),
        "agents_root": str(agents_root),
        "logs_dir": str(logs_dir),
        "output_json_path": str(output_json_path),
        "domain": dom["domain"],
        "trajectory_filename": dom["trajectory_filename"],
        "result_artifact_filename": dom["result_artifact_filename"],
        "domain_notes": dom["domain_notes"],
    }
    limits = build_limits(shor_cfg, backbone_cfg, optimizer_timeout)
    instructions = build_shor_instructions(shor_cfg, template_vars)
    library_dir = data_root / "library"
    python_paths = [str(library_dir)] if library_dir.exists() else []
    return BackboneTask(
        task_id=f"shor:{dom['domain']}:{agent_name}",
        workspace_root=workspace_root or PROJECT_ROOT,
        trajectory_path=trajectory_path,
        instructions=instructions,
        limits=limits,
        artifact_paths={"shor_json": output_json_path},
        template_vars=template_vars,
        runtime_metadata={
            "task": "shor",
            "domain": dom["domain"],
            "agent": agent_name,
            "data_root": str(data_root),
            "agent_path": str(agent_path),
            "agents_root": str(agents_root),
            "logs_dir": str(logs_dir),
            "trajectory_filename": dom["trajectory_filename"],
            "result_artifact_filename": dom["result_artifact_filename"],
            "python_paths": python_paths,
        },
    )


def build_limits(
    task_cfg: dict[str, Any],
    backbone_cfg: dict[str, Any],
    optimizer_timeout: int | None,
) -> BackboneLimits:
    task_limits = dict(task_cfg.get("agent", {}))
    task_limits.update(task_cfg.get("limits", {}))
    backbone_limits = dict(backbone_cfg.get("agent", {}))
    backbone_limits.update(backbone_cfg.get("limits", {}))
    return BackboneLimits(
        step_limit=task_limits.get("step_limit", backbone_limits.get("step_limit")),
        cost_limit=task_limits.get("cost_limit", backbone_limits.get("cost_limit")),
        wall_timeout_seconds=(
            optimizer_timeout
            or task_limits.get("wall_timeout_seconds")
            or backbone_limits.get("wall_timeout_seconds")
            or backbone_cfg.get("wall_timeout_seconds")
        ),
        command_timeout_seconds=(
            task_limits.get("command_timeout_seconds")
            or backbone_limits.get("command_timeout_seconds")
            or backbone_cfg.get("command_timeout_seconds")
        ),
    )


def cleanup_staging_workspace(workspace_path: Path | None, *, agent_out_dir: Path) -> None:
    if workspace_path is None or not workspace_path.exists():
        return
    if workspace_path.name != ".workspace" or workspace_path.parent != agent_out_dir:
        print(f"[shor] refusing to clean unexpected workspace path: {workspace_path}", file=sys.stderr)
        return
    try:
        shutil.rmtree(workspace_path)
    except Exception as exc:
        print(f"[shor] warning: failed to clean workspace {workspace_path}: {exc}", file=sys.stderr)


def cleanup_backbone_runtime(agent_out_dir: Path, *, trajectory_path: Path) -> None:
    backbone_dir = agent_out_dir / ".backbone"
    if trajectory_path.exists():
        try:
            strip_backbone_paths_from_trajectory(trajectory_path)
        except Exception as exc:
            print(
                f"[shor] warning: failed to scrub trajectory runtime paths {trajectory_path}: {exc}",
                file=sys.stderr,
            )
    if not backbone_dir.exists():
        return
    if backbone_dir.name != ".backbone" or backbone_dir.parent != agent_out_dir:
        print(f"[shor] refusing to clean unexpected backbone path: {backbone_dir}", file=sys.stderr)
        return
    try:
        shutil.rmtree(backbone_dir)
    except Exception as exc:
        print(f"[shor] warning: failed to clean backbone runtime {backbone_dir}: {exc}", file=sys.stderr)


def strip_backbone_paths_from_trajectory(trajectory_path: Path) -> None:
    data = json.loads(trajectory_path.read_text())
    if not isinstance(data, dict):
        return
    native = data.get("native")
    if not isinstance(native, dict):
        return

    changed = False
    for key in ("raw_path", "stderr_path"):
        if native.get(key) is not None:
            native[key] = None
            changed = True

    if changed:
        write_json(trajectory_path, data)


def run_one(
    agent_name: str,
    dom: dict,
    data_root: Path,
    shor_cfg: dict,
    backbone_name: str,
    backbone_cfg: dict,
    skip_done: bool,
    optimizer_timeout: int | None,
    command_line: str,
    config_path: Path,
    backbone_config_path: str,
    parallel: int,
    agent_limit: int | None,
    run_log_path: Path | None = None,
) -> RunResult:
    """Run SHOR for one agent.
    Returns (agent_name, status, message) where status is 'ok' / 'skipped' / 'error'.
    """
    agents_root = data_root / "agents"
    logs_root = data_root / "logs"
    canonical_backbone = canonical_backbone_name(backbone_name)
    agent_out_dir = output_dir_for(dom["domain"], agent_name, canonical_backbone)
    agent_out_dir.mkdir(parents=True, exist_ok=True)
    agent_path = agents_root / f"{agent_name}.py"
    logs_dir = logs_root / agent_name
    output_json_path = agent_out_dir / "ranking.json"
    trajectory_path = agent_out_dir / "trajectory.json"
    meta_path = agent_out_dir / "meta.json"
    limits = build_limits(shor_cfg, backbone_cfg, optimizer_timeout)
    staging_workspace_path: Path | None = None
    staging_output_json_path: Path | None = None
    started_at = utc_now_iso()
    monotonic_start = time.monotonic()

    def finalize(
        raw_status: str,
        message: str | None = None,
        cost: float | None = None,
        token_usage: dict[str, int] | None = None,
        llm_call_count: int | None = None,
    ) -> None:
        ended_at = utc_now_iso()
        elapsed_seconds = time.monotonic() - monotonic_start
        paths = {
            "ranking": str(output_json_path) if output_json_path.exists() else None,
            "trajectory": str(trajectory_path) if trajectory_path.exists() else None,
            "meta": str(meta_path),
        }
        meta = build_run_meta(
            task="shor",
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=elapsed_seconds,
            command_line=command_line,
            domain=dom["domain"],
            agent_name=agent_name,
            backbone_name=canonical_backbone,
            data_root=data_root,
            config_path=config_path,
            backbone_config_path=backbone_config_path,
            optimizer_timeout=optimizer_timeout,
            parallel=parallel,
            agent_limit=agent_limit,
            skip_done=skip_done,
            limits=limits,
            status=raw_status,
            cost=cost,
            token_usage=token_usage,
            llm_call_count=llm_call_count,
        )
        meta.update(
            {
                "purpose": "shor",
                "raw_status": raw_status,
                "error_message": message if raw_status == "error" else None,
                "cost_source": "backbone_result.cost",
                "cost_is_partial": cost is None and raw_status != "skipped",
                "result_paths": paths,
                "staging": {
                    "workspace_root": str(staging_workspace_path) if staging_workspace_path else None,
                    "output_json_path": str(staging_output_json_path) if staging_output_json_path else None,
                },
            }
        )
        write_json(meta_path, meta)

    if run_log_path:
        append_run_log(
            run_log_path,
            task="shor",
            event="start",
            agent=agent_name,
            domain=dom["domain"],
            backbone=canonical_backbone,
            output_dir=agent_out_dir,
            trajectory=trajectory_path,
            step_limit=limits.step_limit,
        )

    if not agent_path.exists():
        message = f"agent source not found: {agent_path}"
        finalize("error", message)
        if run_log_path:
            append_run_log(
                run_log_path,
                task="shor",
                event="end",
                agent=agent_name,
                status="error",
                message=message,
                output_dir=agent_out_dir,
                meta=meta_path,
            )
        return (agent_name, "error", message)
    if not logs_dir.exists():
        message = f"agent logs not found: {logs_dir}"
        finalize("error", message)
        if run_log_path:
            append_run_log(
                run_log_path,
                task="shor",
                event="end",
                agent=agent_name,
                status="error",
                message=message,
                output_dir=agent_out_dir,
                meta=meta_path,
            )
        return (agent_name, "error", message)

    if skip_done and output_json_path.exists():
        try:
            existing = json.loads(output_json_path.read_text())
            validate_shor_output(existing)
            if existing.get("schema_version") == "ranking.v2":
                ended_at = utc_now_iso()
                write_json(
                    meta_path,
                    build_run_meta(
                        task="shor",
                        started_at=started_at,
                        ended_at=ended_at,
                        elapsed_seconds=time.monotonic() - monotonic_start,
                        command_line=command_line,
                        domain=dom["domain"],
                        agent_name=agent_name,
                        backbone_name=canonical_backbone,
                        data_root=data_root,
                        config_path=config_path,
                        backbone_config_path=backbone_config_path,
                        optimizer_timeout=optimizer_timeout,
                        parallel=parallel,
                        agent_limit=agent_limit,
                        skip_done=skip_done,
                        limits=limits,
                        status="skipped",
                        cost=None,
                    ),
                )
                finalize("skipped", "ranking.json already exists")
                if run_log_path:
                    append_run_log(
                        run_log_path,
                        task="shor",
                        event="end",
                        agent=agent_name,
                        status="skipped",
                        elapsed_seconds=time.monotonic() - monotonic_start,
                        steps=_read_step_count(trajectory_path),
                        output=output_json_path,
                        meta=meta_path,
                        message="ranking.json already exists",
                    )
                return (agent_name, "skipped", "ranking.json already exists")
        except Exception:
            pass

    try:
        if output_json_path.exists():
            output_json_path.unlink()

        staged = prepare_staged_workspace(
            workspace_root=agent_out_dir / ".workspace",
            source_data_root=data_root,
            agent_name=agent_name,
        )
        staging_workspace_path = staged.root
        staging_output_json_path = staged.output_root / "ranking.json"

        task = build_backbone_task(
            agent_name=agent_name,
            dom=dom,
            data_root=staged.data_root,
            shor_cfg=shor_cfg,
            backbone_cfg=backbone_cfg,
            output_json_path=staging_output_json_path,
            trajectory_path=trajectory_path,
            optimizer_timeout=optimizer_timeout,
            workspace_root=staged.root,
        )
        backbone = load_backbone(backbone_name, backbone_cfg)
        result = backbone.run(task)

        if result.status != "ok":
            detail = f": {result.message}" if result.message else ""
            message = f"backbone status={result.status}{detail}"
            finalize("error", message, result.cost, result.token_usage, result.n_steps)
            if run_log_path:
                append_run_log(
                    run_log_path,
                    task="shor",
                    event="end",
                    agent=agent_name,
                    status="error",
                    backbone_status=result.status,
                    elapsed_seconds=time.monotonic() - monotonic_start,
                    cost=result.cost,
                    steps=_read_step_count(trajectory_path),
                    output_dir=agent_out_dir,
                    trajectory=trajectory_path,
                    message=message,
                    meta=meta_path,
                )
            return (agent_name, "error", message)

        if not staging_output_json_path.exists():
            detail = f": {result.message}" if result.message else ""
            message = f"ranking.json was not written by agent{detail}"
            finalize("error", message, result.cost, result.token_usage, result.n_steps)
            if run_log_path:
                append_run_log(
                    run_log_path,
                    task="shor",
                    event="end",
                    agent=agent_name,
                    status="error",
                    elapsed_seconds=time.monotonic() - monotonic_start,
                    cost=result.cost,
                    steps=_read_step_count(trajectory_path),
                    output_dir=agent_out_dir,
                    trajectory=trajectory_path,
                    message=message,
                    meta=meta_path,
                )
            return (agent_name, "error", message)

        data = json.loads(staging_output_json_path.read_text())
        validate_shor_output(data)
        output_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        ended_at = utc_now_iso()
        write_json(
            meta_path,
            build_run_meta(
                task="shor",
                started_at=started_at,
                ended_at=ended_at,
                elapsed_seconds=time.monotonic() - monotonic_start,
                command_line=command_line,
                domain=dom["domain"],
                agent_name=agent_name,
                backbone_name=canonical_backbone,
                data_root=data_root,
                config_path=config_path,
                backbone_config_path=backbone_config_path,
                optimizer_timeout=optimizer_timeout,
                parallel=parallel,
                agent_limit=agent_limit,
                skip_done=skip_done,
                limits=limits,
                status="ok",
                cost=result.cost,
                token_usage=result.token_usage,
                llm_call_count=result.n_steps,
            ),
        )
        finalize("ok", cost=result.cost, token_usage=result.token_usage, llm_call_count=result.n_steps)
        if run_log_path:
            append_run_log(
                run_log_path,
                task="shor",
                event="end",
                agent=agent_name,
                status="ok",
                elapsed_seconds=time.monotonic() - monotonic_start,
                cost=result.cost,
                steps=_read_step_count(trajectory_path),
                output=output_json_path,
                trajectory=trajectory_path,
                meta=meta_path,
            )
        return (agent_name, "ok", "")
    except SystemExit as e:
        _print_exception_traceback(e)
        finalize("error", str(e))
        if run_log_path:
            append_run_log(
                run_log_path,
                task="shor",
                event="end",
                agent=agent_name,
                status="error",
                elapsed_seconds=time.monotonic() - monotonic_start,
                steps=_read_step_count(trajectory_path),
                output_dir=agent_out_dir,
                message=str(e),
                meta=meta_path,
            )
        return (agent_name, "error", str(e))
    except Exception as e:
        _print_exception_traceback(e)
        message = f"{type(e).__name__}: {e}"
        finalize("error", message)
        if run_log_path:
            append_run_log(
                run_log_path,
                task="shor",
                event="end",
                agent=agent_name,
                status="error",
                elapsed_seconds=time.monotonic() - monotonic_start,
                steps=_read_step_count(trajectory_path),
                output_dir=agent_out_dir,
                message=message,
                meta=meta_path,
            )
        return (agent_name, "error", message)
    finally:
        cleanup_staging_workspace(staging_workspace_path, agent_out_dir=agent_out_dir)
        cleanup_backbone_runtime(agent_out_dir, trajectory_path=trajectory_path)


def build_domain_plan(
    args: argparse.Namespace,
    domain: str,
    registered_agents: dict[str, list[str]],
) -> dict[str, Any]:
    dom = load_domain_config(domain)
    data_root = resolve_data_root(dom["data_root"])
    agents_root = data_root / "agents"
    logs_root = data_root / "logs"
    if not agents_root.exists():
        raise SystemExit(f"agents_root not found: {agents_root}")
    if not logs_root.exists():
        raise SystemExit(f"logs_root not found: {logs_root}")

    agent_list = resolve_agent_list(
        args.agent,
        domain=domain,
        registered_agents=registered_agents,
    )
    if not agent_list:
        raise SystemExit("no agents to run")
    return {"domain": domain, "dom": dom, "data_root": data_root, "agents": agent_list}


def run_domain(
    args: argparse.Namespace,
    plan: dict[str, Any],
    command_line: str,
    progress: tqdm | None = None,
) -> tuple[int, int, int, list[RunResult]]:
    dom = plan["dom"]
    data_root = plan["data_root"]
    agent_list = plan["agents"]
    canonical_backbone = canonical_backbone_name(args.optimizer)

    config_path = DEFAULT_CONFIG
    shor_cfg = yaml.safe_load(config_path.read_text())
    backbone_config_path = f"harness_optimizer:{canonical_backbone}:hardcoded"
    backbone_cfg = default_backbone_config(canonical_backbone)
    results: list[RunResult] = []
    t0 = time.time()
    parallel = max(1, args.parallel)
    run_log_path = run_log_path_for(dom["domain"], canonical_backbone)
    append_run_log(
        run_log_path,
        task="shor",
        event="batch_start",
        domain=dom["domain"],
        agents=len(agent_list),
        backbone=canonical_backbone,
        limit=args.limit,
        parallel=parallel,
        skip_done=args.skip_done,
        command=command_line,
    )
    counts = {"ok": 0, "skipped": 0, "error": 0}

    def report(r: RunResult) -> None:
        counts[r[1]] += 1
        if progress is not None:
            progress.set_postfix(
                domain=dom["domain"],
                ok=counts["ok"],
                skip=counts["skipped"],
                err=counts["error"],
                refresh=False,
            )
            progress.update(1)
            if r[1] == "error":
                progress.write(f"[ERR] {dom['domain']}/{r[0]}: {r[2]}")

    if parallel == 1 or len(agent_list) == 1:
        for agent in agent_list:
            r = run_one(
                agent,
                dom,
                data_root,
                shor_cfg,
                canonical_backbone,
                backbone_cfg,
                args.skip_done,
                args.optimizer_timeout,
                command_line,
                config_path,
                backbone_config_path,
                parallel,
                args.limit,
                run_log_path,
            )
            results.append(r)
            report(r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {
                ex.submit(
                    run_one,
                    a,
                    dom,
                    data_root,
                    shor_cfg,
                    canonical_backbone,
                    backbone_cfg,
                    args.skip_done,
                    args.optimizer_timeout,
                    command_line,
                    config_path,
                    backbone_config_path,
                    parallel,
                    args.limit,
                    run_log_path,
                ): a
                for a in agent_list
            }
            for fut in concurrent.futures.as_completed(futures):
                r = fut.result()
                results.append(r)
                report(r)

    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_skip = sum(1 for _, s, _ in results if s == "skipped")
    n_err = sum(1 for _, s, _ in results if s == "error")
    elapsed = time.time() - t0

    append_run_log(
        run_log_path,
        task="shor",
        event="batch_end",
        domain=dom["domain"],
        status="error" if n_err else "ok",
        elapsed_seconds=elapsed,
        ok=n_ok,
        skipped=n_skip,
        errors=n_err,
        total=len(results),
    )
    return (n_ok, n_skip, n_err, results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="all",
                    help="registered agent name, comma-separated list, or 'all' (default)")
    ap.add_argument("--domain", default="all",
                    help="domain name or 'all' (default)")
    ap.add_argument("--optimizer",
                    default=os.getenv("SHOR_OPTIMIZER", "openhands_cli"),
                    help="harness_optimizer implementation: openhands_cli, claude_code_cli, or codex_cli")
    ap.add_argument("--optimizer-timeout", type=int, default=None,
                    help="override wall timeout in seconds for the optimizer run")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N selected agents across the chosen domain(s)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="number of agents to run in parallel (default 1)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip agents whose ranking.json already exists")
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    command_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    registered_agents = load_registered_agents()
    if args.domain == "all":
        selected_domains = []
        for domain in available_domains():
            if domain not in registered_agents:
                continue
            dom = load_domain_config(domain)
            data_root = resolve_data_root(dom["data_root"])
            if (data_root / "agents").exists() and (data_root / "logs").exists():
                selected_domains.append(domain)
        if not selected_domains:
            raise SystemExit(
                f"no domains from {DEFAULT_TARGETS_PATH.name} with agents/logs found"
            )
    else:
        selected_domains = [args.domain]

    plans = [
        build_domain_plan(args, domain, registered_agents)
        for domain in selected_domains
    ]
    plans = apply_agent_limit(plans, args.limit)
    total_agents = sum(len(plan["agents"]) for plan in plans)
    if total_agents == 0:
        raise SystemExit("no agents to run")
    canonical_backbone = canonical_backbone_name(args.optimizer)
    print(
        "settings: "
        f"domain={args.domain}  "
        f"domains={','.join(plan['domain'] for plan in plans)}  "
        f"optimizer={canonical_backbone}  "
        f"agent={args.agent}  "
        f"agents={total_agents}  "
        f"limit={args.limit}  "
        f"parallel={max(1, args.parallel)}  "
        f"skip_done={args.skip_done}  "
        f"optimizer_timeout={args.optimizer_timeout}"
    )

    totals = [0, 0, 0]
    all_results: list[tuple[str, RunResult]] = []
    with tqdm(total=total_agents, desc="SHOR", unit="agent", dynamic_ncols=True) as progress:
        for plan in plans:
            n_ok, n_skip, n_err, results = run_domain(args, plan, command_line, progress)
            totals[0] += n_ok
            totals[1] += n_skip
            totals[2] += n_err
            all_results.extend((plan["domain"], result) for result in results)
    print(
        f"summary: ok={totals[0]}  skipped={totals[1]}  "
        f"error={totals[2]}  total={total_agents}  domains={len(plans)}"
    )
    missing_ranking = [
        (domain, agent, message)
        for domain, (agent, status, message) in all_results
        if status == "error" and message.startswith("ranking.json was not written by agent")
    ]
    if missing_ranking:
        print(
            f"{len(missing_ranking)} case(s) ended without the harness optimizer "
            "creating ranking.json. Re-run run_shor with --skip-done to retry only "
            "unfinished cases."
        )
        preview = missing_ranking[:10]
        for domain, agent, _message in preview:
            print(f"  missing ranking.json: {domain}/{agent}")
        if len(missing_ranking) > len(preview):
            print(f"  ... and {len(missing_ranking) - len(preview)} more")
    if totals[2]:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0) and not isinstance(exc.code, int):
            _print_exception_traceback(exc)
        raise SystemExit(exc.code if isinstance(exc.code, int) else 1) from None
    except Exception as exc:
        _print_exception_traceback(exc)
        raise SystemExit(1) from None
