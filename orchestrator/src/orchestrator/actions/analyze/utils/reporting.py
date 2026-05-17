from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..engine import PerformanceAnalyzer


def render_metadata_md(analyzer: PerformanceAnalyzer) -> str:
    md = ["\n## Appendix: Experiment Parameters"]
    meta = analyzer.experiment.metadata if analyzer.experiment else {}
    params = meta.get("parameters", {})
    durations = meta.get("calculated_durations_sec", {})
    rows = [
        ["Run Timestamp (UTC)", f"`{meta.get('repetition_timestamp_utc', 'N/A')}`"],
        ["Runs per Technology", params.get("num_repetitions", "N/A")],
        ["Target RPS per Instance", params.get("rps", "N/A")],
        ["k6 Test Duration", f"`{params.get('k6_duration', 'N/A')}`"],
        ["Warm-up Duration", f"`{params.get('warmup_duration', 'N/A')}`"],
        ["Measurement Duration (sec)", durations.get("measurement", "N/A")],
    ]
    md.append(pd.DataFrame(rows, columns=["Parameter", "Value"]).to_markdown(index=False))
    return "\n".join(md)


def write_report(analyzer: PerformanceAnalyzer, content: str) -> None:
    with open(analyzer.report_path, "w", encoding="utf-8") as f:
        f.write(content)
