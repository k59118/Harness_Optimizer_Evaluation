from __future__ import annotations

from typing import Any


SHOR_OUTPUT_SCHEMA = {
    "schema_version": "ranking.v2",
    "required": ["schema_version", "agent", "ranking", "evidence", "rationale"],
    "axes": ["A", "B", "C", "D"],
}


def validate_shor_output(data: dict[str, Any]) -> None:
    if data.get("schema_version") != "ranking.v2":
        raise ValueError(f"schema_version != 'ranking.v2': {data.get('schema_version')!r}")
    ranking = data.get("ranking")
    if not (isinstance(ranking, list) and sorted(ranking) == ["A", "B", "C", "D"]):
        raise ValueError(f"ranking must be a permutation of A,B,C,D: {ranking!r}")
    evidence = data.get("evidence", {})
    rationale = data.get("rationale", {})
    for axis in ["A", "B", "C", "D"]:
        if axis not in evidence or not isinstance(evidence[axis], list):
            raise ValueError(f"evidence.{axis} missing or not a list")
        if axis not in rationale or not isinstance(rationale[axis], str) or not rationale[axis].strip():
            raise ValueError(f"rationale.{axis} missing or empty")
