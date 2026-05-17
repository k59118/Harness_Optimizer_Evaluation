#!/usr/bin/env python3
"""Evaluate SHOR result files against shor_final-style GT rankings."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GT_PATH = PROJECT_ROOT / "shor_final.json"
DEFAULT_RESULT_PATH = PROJECT_ROOT / "result"

CODE_TO_COMPONENT = {"A": "Prompt", "B": "Tool", "C": "Memory", "D": "Workflow"}
COMPONENT_ORDER = ["Prompt", "Tool", "Memory", "Workflow"]


def gt_components_with_best_rank(gt_ordering: dict[str, int]) -> set[str]:
    best = min(gt_ordering.values())
    return {comp for comp, rank in gt_ordering.items() if rank == best}


def compute_top1_acc(pred_components: list[str], gt_ordering: dict[str, int]) -> float:
    return 1.0 if pred_components[0] in gt_components_with_best_rank(gt_ordering) else 0.0


def compute_ndcg(pred_components: list[str], gt_ordering: dict[str, int]) -> float:
    relevance = {comp: 5 - rank for comp, rank in gt_ordering.items()}
    dcg = sum(
        (2 ** relevance[comp] - 1) / math.log2(i + 2)
        for i, comp in enumerate(pred_components)
    )
    ideal_order = sorted(gt_ordering.keys(), key=lambda comp: gt_ordering[comp])
    idcg = sum(
        (2 ** relevance[comp] - 1) / math.log2(i + 2)
        for i, comp in enumerate(ideal_order)
    )
    return dcg / idcg if idcg > 0 else 0.0


def parse_pred_ordering(pred_codes: list[str]) -> tuple[list[str], str]:
    if len(pred_codes) != 4:
        return [], f"length={len(pred_codes)} (expected 4)"
    counts = Counter(pred_codes)
    if any(counts[code] != 1 for code in CODE_TO_COMPONENT):
        return [], f"codes not exactly A/B/C/D once each: {pred_codes}"
    return [CODE_TO_COMPONENT[code] for code in pred_codes], ""


def load_gt(gt_path: Path) -> dict[tuple[str, str], dict[str, int]]:
    raw = json.loads(gt_path.read_text())
    if not isinstance(raw, list):
        raise SystemExit(f"GT must be a shor_final-style list: {gt_path}")

    out: dict[tuple[str, str], dict[str, int]] = {}
    for i, row in enumerate(raw, 1):
        if not isinstance(row, dict):
            raise SystemExit(f"GT row {i} is not an object")
        domain = row.get("domain")
        harness_id = row.get("harness_id")
        ranking = row.get("ranking")
        if not isinstance(domain, str) or not isinstance(harness_id, str):
            raise SystemExit(f"GT row {i} missing domain or harness_id")
        if not isinstance(ranking, dict) or sorted(ranking) != sorted(COMPONENT_ORDER):
            raise SystemExit(f"GT row {i} has invalid SHOR ordering keys: {ranking!r}")
        out[(domain, harness_id)] = {comp: int(ranking[comp]) for comp in COMPONENT_ORDER}
    return out


def iter_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"result path not found: {path}")
    return sorted(path.rglob("ranking.json"))


def infer_domain(path: Path, result_base: Path | None) -> str | None:
    parts = path.resolve().parts
    if result_base is not None:
        try:
            rel_parts = path.resolve().relative_to(result_base.resolve()).parts
        except ValueError:
            rel_parts = ()
        if len(rel_parts) >= 3:
            return rel_parts[-3]
        if len(rel_parts) >= 2:
            return rel_parts[0]

    if "result" in parts:
        idx = len(parts) - 1 - parts[::-1].index("result")
        if len(parts) > idx + 2:
            return parts[idx + 2]
    return path.parent.parent.name if path.parent.parent != path.parent else None


def load_prediction(path: Path, result_base: Path | None) -> tuple[str | None, str | None, list[str], str]:
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except Exception as exc:
        return None, None, [], f"invalid json: {exc}"

    ranking = data.get("ranking")
    if not isinstance(ranking, list) or not all(isinstance(code, str) for code in ranking):
        return None, None, [], f"SHOR output ranking field is not a string list: {ranking!r}"
    domain = data.get("domain") if isinstance(data.get("domain"), str) else infer_domain(path, result_base)
    harness_id = data.get("agent") if isinstance(data.get("agent"), str) else path.parent.name
    pred_components, err = parse_pred_ordering(ranking)
    return domain, harness_id, pred_components, err


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def group_rows_by_domain(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    return by_domain


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"evaluated": len(rows)}
    if not rows:
        return summary

    summary["acc@1"] = mean([row["acc@1"] for row in rows])
    summary["ndcg"] = mean([row["ndcg"] for row in rows])
    by_domain = group_rows_by_domain(rows)
    if len(by_domain) > 1:
        summary["by_domain"] = {
            domain: {
                "evaluated": len(domain_rows),
                "acc@1": mean([row["acc@1"] for row in domain_rows]),
                "ndcg": mean([row["ndcg"] for row in domain_rows]),
            }
            for domain, domain_rows in sorted(by_domain.items())
        }
    return summary


def print_metric_summary(summary: dict[str, Any]) -> None:
    print(f"evaluated: {summary['evaluated']}")
    if summary["evaluated"] == 0:
        return

    print(f"acc@1:    {summary['acc@1']:.4f}")
    print(f"ndcg:     {summary['ndcg']:.4f}")
    by_domain = summary.get("by_domain")
    if by_domain:
        print()
        print("by domain:")
        for domain, domain_summary in by_domain.items():
            print(
                f"  {domain}: n={domain_summary['evaluated']} "
                f"acc@1={domain_summary['acc@1']:.4f} "
                f"ndcg={domain_summary['ndcg']:.4f}"
            )


def print_rows(rows: list[dict[str, Any]], verbose: bool) -> None:
    print_metric_summary(metric_summary(rows))
    if not rows:
        return

    if verbose:
        print()
        print("per result:")
        for row in rows:
            print(
                f"  {row['domain']}/{row['harness_id']}: "
                f"acc@1={row['acc@1']:.1f} ndcg={row['ndcg']:.4f} "
                f"pred={row['pred']} gt={row['gt']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SHOR result files against shor_final.json."
    )
    parser.add_argument(
        "result_path",
        nargs="?",
        default=str(DEFAULT_RESULT_PATH),
        help="SHOR result file or directory to scan recursively (default: result)",
    )
    parser.add_argument(
        "--gt",
        default=str(DEFAULT_GT_PATH),
        help="shor_final-style GT JSON path (default: shor_final.json)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="optional domain filter, also used when a single result file cannot infer domain",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-result metrics")
    args = parser.parse_args()

    result_path = Path(args.result_path)
    result_base = result_path if result_path.is_dir() else None
    gt = load_gt(Path(args.gt))
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for result_file in iter_result_files(result_path):
        domain, harness_id, pred_components, err = load_prediction(result_file, result_base)
        if args.domain:
            domain = args.domain if domain is None else domain
            if domain != args.domain:
                continue
        if err:
            skipped.append(f"{result_file}: {err}")
            continue
        if not domain or not harness_id:
            skipped.append(f"{result_file}: could not infer domain or harness_id")
            continue
        gt_ordering = gt.get((domain, harness_id))
        if gt_ordering is None:
            skipped.append(f"{result_file}: no GT for {domain}/{harness_id}")
            continue
        rows.append(
            {
                "domain": domain,
                "harness_id": harness_id,
                "pred": pred_components,
                "gt": gt_ordering,
                "acc@1": compute_top1_acc(pred_components, gt_ordering),
                "ndcg": compute_ndcg(pred_components, gt_ordering),
                "path": str(result_file),
            }
        )

    print_rows(rows, verbose=args.verbose)
    if skipped:
        print()
        print(f"skipped: {len(skipped)}")
        if args.verbose:
            for item in skipped:
                print(f"  {item}")


if __name__ == "__main__":
    main()
