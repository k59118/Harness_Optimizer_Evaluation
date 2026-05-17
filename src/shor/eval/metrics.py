"""SHOR evaluation metrics.

Each ranking is a permutation of axis labels (e.g. ["D","A","B","C"]),
strongest first. Functions accept any sequence of comparable labels.

Two primary metrics:
  - reciprocal_rank(gt, pred) and its mean (mrr) across many pairs:
    measures where the GT top-1 axis lands in the predicted ranking.
    1.0 when pred ranks the GT top-1 first; 1/2, 1/3, 1/4 if it lands
    second / third / fourth; 0.0 if absent.
  - top1_correct(gt, pred) and its mean (top1_accuracy):
    strict yes/no on whether pred[0] == gt[0].
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def _as_list(ranking: Sequence[str], name: str) -> list[str]:
    if not isinstance(ranking, (list, tuple)):
        raise TypeError(f"{name} must be a list/tuple, got {type(ranking).__name__}")
    return list(ranking)


def reciprocal_rank(gt: Sequence[str], pred: Sequence[str]) -> float:
    """Reciprocal rank of GT's top-1 axis in `pred`.

    Returns 1/position (1-indexed) if gt[0] appears in pred, else 0.0.
    """
    gt = _as_list(gt, "gt")
    pred = _as_list(pred, "pred")
    if not gt:
        raise ValueError("gt is empty")
    target = gt[0]
    for i, item in enumerate(pred, 1):
        if item == target:
            return 1.0 / i
    return 0.0


def top1_correct(gt: Sequence[str], pred: Sequence[str]) -> bool:
    """True iff pred's top-1 matches GT's top-1."""
    gt = _as_list(gt, "gt")
    pred = _as_list(pred, "pred")
    if not gt or not pred:
        return False
    return gt[0] == pred[0]


def mrr(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> float:
    """Mean reciprocal rank over a collection of (gt, pred) pairs."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(reciprocal_rank(gt, pred) for gt, pred in pairs) / len(pairs)


def top1_accuracy(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> float:
    """Fraction of pairs whose top-1 matches."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(1 for gt, pred in pairs if top1_correct(gt, pred)) / len(pairs)
