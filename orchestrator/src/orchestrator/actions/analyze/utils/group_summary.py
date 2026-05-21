from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .stats import GroupStats, compute_group_stats


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    decimals: int
    higher_is: Literal["CSR", "SSR"]


_GROUP_ORDER: tuple[str, ...] = ("CSR", "SSR")
_HEADER_COLS: tuple[str, ...] = (
    "Grupa",
    "n",
    "mediana",
    "średnia",
    "std",
    "min",
    "max",
    "p25",
    "p75",
    "IQR",
)


def _fmt(value: float, decimals: int) -> str:
    if value != value:  # NaN
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ")


def _row(group: str, s: GroupStats, decimals: int) -> str:
    cells = [
        group,
        str(s.n),
        _fmt(s.median, decimals),
        _fmt(s.mean, decimals),
        _fmt(s.std, decimals),
        _fmt(s.min, decimals),
        _fmt(s.max, decimals),
        _fmt(s.p25, decimals),
        _fmt(s.p75, decimals),
        _fmt(s.iqr, decimals),
    ]
    return "| " + " | ".join(cells) + " |"


def render_group_summary_section(
    per_app_values: Mapping[str, Mapping[str, Sequence[float]]],
    metric_specs: Sequence[MetricSpec],
) -> str:
    lines: list[str] = ["## Podsumowanie zbiorcze (CSR vs SSR)", ""]
    header = "| " + " | ".join(_HEADER_COLS) + " |"
    sep = "|" + "|".join(["---"] * len(_HEADER_COLS)) + "|"

    for spec in metric_specs:
        groups = per_app_values[spec.name]
        stats_by_group = {g: compute_group_stats(groups[g]) for g in _GROUP_ORDER}

        heading = spec.name if "(" in spec.name else f"{spec.name} ({spec.unit})"
        lines.append(f"### {heading}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for g in _GROUP_ORDER:
            lines.append(_row(g, stats_by_group[g], spec.decimals))

        hi, lo = spec.higher_is, "SSR" if spec.higher_is == "CSR" else "CSR"
        ratio = stats_by_group[hi].median / stats_by_group[lo].median
        diff = stats_by_group[hi].median - stats_by_group[lo].median
        lines.append("")
        lines.append(
            f"_Stosunek {hi}/{lo} (mediana): {ratio:.2f}×; "
            f"różnica bezwzględna: {_fmt(diff, spec.decimals)} {spec.unit}_"
        )
        lines.append("")

    return "\n".join(lines)
