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
        # Use rounded medians so the displayed ratio and difference are internally
        # consistent with the medians shown in the table above.
        hi_med = round(stats_by_group[hi].median, spec.decimals)
        lo_med = round(stats_by_group[lo].median, spec.decimals)
        ratio = hi_med / lo_med if lo_med != 0 else float("inf")
        diff = hi_med - lo_med
        lines.append("")
        lines.append(
            f"_Stosunek {hi}/{lo} (mediana): {ratio:.2f}×; "
            f"różnica bezwzględna: {_fmt(diff, spec.decimals)} {spec.unit}_"
        )
        lines.append("")

    return "\n".join(lines)


def load_family_map() -> dict[str, str]:
    """Return mapping of Application ID -> framework family (e.g. 'react')."""
    from orchestrator.config import APPLICATIONS_DIR
    from orchestrator.shared.research.application import ApplicationRegistry

    registry = ApplicationRegistry(APPLICATIONS_DIR)
    return {app.id: app.family for app in registry.all()}


def _family_display(family: str) -> str:
    overrides = {"nextjs": "Next.js", "nuxtjs": "Nuxt"}
    return overrides.get(family.lower(), family.capitalize())


def render_per_family_group_summary_section(
    per_app_values_by_family: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    metric_specs: Sequence[MetricSpec],
) -> str:
    families = sorted(
        f
        for f, per_metric in per_app_values_by_family.items()
        if any(
            "CSR" in per_metric.get(spec.name, {}) and "SSR" in per_metric.get(spec.name, {})
            for spec in metric_specs
        )
    )
    if not families:
        return ""

    lines: list[str] = ["## Podsumowanie zbiorcze według frameworku (CSR vs SSR)", ""]
    header = "| " + " | ".join(_HEADER_COLS) + " |"
    sep = "|" + "|".join(["---"] * len(_HEADER_COLS)) + "|"

    for family in families:
        per_metric = per_app_values_by_family[family]
        lines.append(f"### Framework: {_family_display(family)}")
        lines.append("")
        for spec in metric_specs:
            groups = per_metric.get(spec.name, {})
            if "CSR" not in groups or "SSR" not in groups:
                continue
            stats_by_group = {g: compute_group_stats(groups[g]) for g in _GROUP_ORDER}

            heading = spec.name if "(" in spec.name else f"{spec.name} ({spec.unit})"
            lines.append(f"#### {heading}")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for g in _GROUP_ORDER:
                lines.append(_row(g, stats_by_group[g], spec.decimals))

            hi, lo = spec.higher_is, "SSR" if spec.higher_is == "CSR" else "CSR"
            hi_med = round(stats_by_group[hi].median, spec.decimals)
            lo_med = round(stats_by_group[lo].median, spec.decimals)
            ratio = hi_med / lo_med if lo_med != 0 else float("inf")
            diff = hi_med - lo_med
            lines.append("")
            lines.append(
                f"_Stosunek {hi}/{lo} (mediana): {ratio:.2f}×; "
                f"różnica bezwzględna: {_fmt(diff, spec.decimals)} {spec.unit}_"
            )
            lines.append("")

    return "\n".join(lines)
