from pathlib import Path
from typing import List, Optional

from .engine import DataAggregator


def run(
    sources: List[str],
    output: Optional[Path] = None,
    lax: bool = False,
    copy_logs: bool = True,
) -> None:
    """
    Entry point for the aggregate action.
    """
    aggregator = DataAggregator(
        source_specs=sources, output_dir=output, lax=lax, copy_logs=copy_logs
    )
    aggregator.run()
