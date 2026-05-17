#!/usr/bin/env python3
"""Evaluate SHOR predictions against ground-truth rankings.

Default per-domain layout:
  result/<harness_optimizer>/<domain>/<agent>/ranking.json
  data/<domain>/ranking/gt/rankings.json
  data/<domain>/ranking/eval/report.json

Single domain (everything default):
  python eval_runner.py --domain swe

All domains:
  python eval_runner.py --domain all

Override GT file (flat for one domain, or nested {domain:{agent:[...]}}
for any --domain value):
  python eval_runner.py --gt /path/to/gt.json --domain swe
  python eval_runner.py --gt /path/to/nested.json --domain all

Override optimizer, pred-dir, or report path:
  python eval_runner.py --domain swe --optimizer openhands_cli --pred-dir /custom/pred --report /custom/out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOR_ROOT = HERE.parent
if str(SHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOR_ROOT))
PROJECT_ROOT = SHOR_ROOT.parent.parent

from eval.metrics import reciprocal_rank, top1_correct, mrr, top1_accuracy

DEFAULT_DATA_BASE = PROJECT_ROOT / "data"
DEFAULT_RESULT_BASE = PROJECT_ROOT / "result"
DOMAIN_CONFIG_DIR = SHOR_ROOT / "config" / "domains"


def list_known_domains() -> list[str]:
    return sorted(p.stem for p in DOMAIN_CONFIG_DIR.glob("*.yaml"))


def load_gt_file(gt_path: Path) -> tuple[dict, bool]:
    """Return (raw, is_nested). Nested means {domain: {agent: [...]}}."""
    raw = json.loads(gt_path.read_text())
    if not isinstance(raw, dict) or not raw:
        return raw, False
    sample = next(iter(raw.values()))
    return raw, isinstance(sample, dict)


def collect_predictions(pred_dir: Path, backbone: str | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not pred_dir.exists():
        return out
    for sub in sorted(pred_dir.iterdir()):
        if not sub.is_dir():
            continue
        rj = sub / "ranking.json"
        if not rj.exists() and backbone:
            rj = sub / backbone / "ranking.json"
            if not rj.exists():
                backbone_dir = sub / backbone
                nested = (
                    sorted(p / "ranking.json" for p in backbone_dir.iterdir() if p.is_dir())
                    if backbone_dir.is_dir()
                    else []
                )
                existing = [p for p in nested if p.exists()]
                if len(existing) == 1:
                    rj = existing[0]
        if not rj.exists():
            nested = sorted(p / "ranking.json" for p in sub.iterdir() if p.is_dir())
            nested.extend(sorted(sub.glob("*/*/ranking.json")))
            existing = [p for p in nested if p.exists()]
            preferred = sub / "openhands_cli" / "ranking.json"
            preferred_dir = sub / "openhands_cli"
            preferred_nested = (
                sorted(p / "ranking.json" for p in preferred_dir.iterdir() if p.is_dir())
                if preferred_dir.is_dir()
                else []
            )
            preferred_existing = [p for p in preferred_nested if p.exists()]
            if preferred.exists():
                rj = preferred
            elif len(preferred_existing) == 1:
                rj = preferred_existing[0]
            elif len(existing) == 1:
                rj = existing[0]
        if not rj.exists():
            continue
        try:
            d = json.loads(rj.read_text())
            r = d.get("ranking")
            if isinstance(r, list):
                out[sub.name] = r
        except Exception:
            continue
    return out


def evaluate_domain(gt: dict[str, list[str]], pred: dict[str, list[str]]) -> dict:
    common = sorted(set(gt) & set(pred))
    only_gt = sorted(set(gt) - set(pred))
    only_pred = sorted(set(pred) - set(gt))
    pairs = [(gt[a], pred[a]) for a in common]
    per_agent = []
    for a in common:
        per_agent.append({
            "agent": a,
            "gt": gt[a],
            "pred": pred[a],
            "rr": reciprocal_rank(gt[a], pred[a]),
            "top1": top1_correct(gt[a], pred[a]),
        })
    return {
        "n_evaluated": len(common),
        "n_only_gt": len(only_gt),
        "n_only_pred": len(only_pred),
        "only_gt": only_gt,
        "only_pred": only_pred,
        "mrr": mrr(pairs),
        "top1_accuracy": top1_accuracy(pairs),
        "per_agent": per_agent,
    }


def print_domain_report(domain: str, rep: dict, verbose: bool = False) -> None:
    print(f"=== {domain} ===")
    print(f"  evaluated:     {rep['n_evaluated']}")
    print(f"  only in GT:    {rep['n_only_gt']}  only in pred: {rep['n_only_pred']}")
    print(f"  MRR:           {rep['mrr']:.4f}")
    print(f"  top1 accuracy: {rep['top1_accuracy']:.4f}")
    if verbose and rep["per_agent"]:
        print("  per-agent:")
        for row in rep["per_agent"]:
            mark = "OK " if row["top1"] else "MISS"
            print(
                f"    [{mark}] rr={row['rr']:.3f}  {row['agent']}  "
                f"gt={row['gt']}  pred={row['pred']}"
            )


def default_pred_dir(domain: str, harness_optimizer: str) -> Path:
    return DEFAULT_RESULT_BASE / harness_optimizer / domain


def default_gt_path(domain: str) -> Path:
    return DEFAULT_DATA_BASE / domain / "ranking" / "gt" / "rankings.json"


def default_report_path(domain: str) -> Path:
    return DEFAULT_DATA_BASE / domain / "ranking" / "eval" / "report.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    help="domain name, or 'all' to iterate every known domain")
    ap.add_argument("--gt", default=None,
                    help="GT file path. If omitted: per-domain "
                         "data/<domain>/ranking/gt/rankings.json")
    ap.add_argument("--pred-dir", default=None,
                    help="default: result/<optimizer>/<domain>")
    ap.add_argument("--optimizer", default="openhands_cli",
                    help="harness optimizer result root to evaluate")
    ap.add_argument("--report", default=None,
                    help="combined JSON report path. If omitted: per-domain "
                         "data/<domain>/ranking/eval/report.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.domain == "all":
        domains = list_known_domains()
    else:
        domains = [args.domain]

    # Resolve a `get_gt(domain) -> dict | None` lookup.
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            raise SystemExit(f"gt file not found: {gt_path}")
        raw, is_nested = load_gt_file(gt_path)
        if is_nested:
            def get_gt(d, _raw=raw):
                return _raw.get(d)
        else:
            if len(domains) > 1:
                raise SystemExit(
                    "--gt is a flat file but multiple domains were requested; "
                    "use a nested gt ({domain:{agent:[...]}}) or a single --domain"
                )
            def get_gt(d, _raw=raw):
                return _raw
    else:
        if args.pred_dir and len(domains) > 1:
            raise SystemExit("--pred-dir is incompatible with --domain all")
        def get_gt(d):
            p = default_gt_path(d)
            if not p.exists():
                return None
            return json.loads(p.read_text())

    full_report: dict[str, dict] = {}
    overall_pairs: list[tuple[list[str], list[str]]] = []

    for d in domains:
        gt = get_gt(d)
        if not gt:
            print(f"=== {d} === (skipped: gt not found)\n")
            continue
        pred_dir = (
            Path(args.pred_dir) if args.pred_dir
            else default_pred_dir(d, args.optimizer)
        )
        pred = collect_predictions(pred_dir, backbone=args.optimizer)
        rep = evaluate_domain(gt, pred)
        full_report[d] = rep
        for a in set(gt) & set(pred):
            overall_pairs.append((gt[a], pred[a]))
        print_domain_report(d, rep, verbose=args.verbose)

        if not args.report:
            out_path = default_report_path(d)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps({"domain": d, **rep}, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"  → {out_path}")
        print()

    if len(domains) > 1:
        print("=== overall ===")
        n = len(overall_pairs)
        print(f"  evaluated:     {n}")
        if n:
            print(f"  MRR:           {mrr(overall_pairs):.4f}")
            print(f"  top1 accuracy: {top1_accuracy(overall_pairs):.4f}")

    if args.report:
        out = {"domains": full_report}
        if len(domains) > 1 and overall_pairs:
            out["overall"] = {
                "n_evaluated": len(overall_pairs),
                "mrr": mrr(overall_pairs),
                "top1_accuracy": top1_accuracy(overall_pairs),
            }
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"\ncombined report: {rp}")


if __name__ == "__main__":
    main()
