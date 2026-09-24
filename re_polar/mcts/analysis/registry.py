"""Insight registry, the "select what insights I want" mechanism.

An insight is a small, self-contained function:
    fn(groups: dict[str, list[sample]], num_layers: int) -> dict
returning JSON-serializable numbers, keyed however that insight likes (each
insight's own docstring/tests are the contract for its output shape).
`chart_kind` just tags how the result is naturally visualized (a caller's
own choice how to use that).

Insights register themselves via the `@insight(...)` decorator in
`re_polar/mcts/analysis/insights.py` at import time, `list_insights()`/`get_insight()`
below are only meaningful after that module has been imported.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

_REGISTRY: Dict[str, "InsightSpec"] = {}


@dataclass(frozen=True)
class InsightSpec:
    id: str
    title: str
    chart_kind: str  # "heatmap" | "table" | "boxplot" | "line" | "stackedBar" | "hbar"
    fn: Callable
    description: str = ""


def insight(id: str, title: str, chart_kind: str, description: str = ""):
    """Decorator: registers `fn` under `id`. Raises on a duplicate id (fail
    loud at import time rather than silently shadowing an existing insight)."""
    if chart_kind not in ("heatmap", "table", "boxplot", "line", "stackedBar", "hbar"):
        raise ValueError(f"insight {id!r}: unknown chart_kind {chart_kind!r}")

    def wrap(fn):
        if id in _REGISTRY:
            raise ValueError(
                f"duplicate insight id {id!r} (already registered by "
                f"{_REGISTRY[id].fn.__module__}.{_REGISTRY[id].fn.__name__})"
            )
        _REGISTRY[id] = InsightSpec(
            id=id, title=title, chart_kind=chart_kind, fn=fn, description=description
        )
        return fn

    return wrap


def get_insight(id: str) -> InsightSpec:
    if id not in _REGISTRY:
        known = (
            ", ".join(sorted(_REGISTRY))
            or "(none registered, import re_polar.mcts.analysis.insights first)"
        )
        raise KeyError(f"unknown insight id {id!r}. Known insights: {known}")
    return _REGISTRY[id]


def list_insights() -> List[InsightSpec]:
    return sorted(_REGISTRY.values(), key=lambda s: s.id)
