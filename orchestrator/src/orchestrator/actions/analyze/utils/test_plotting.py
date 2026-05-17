from typing import Any, List

import pandas as pd
from pytest_mock import MockerFixture

from orchestrator.actions.analyze.utils.plotting import (
    get_ordered_tech_list,
    order_techs_by_avg_rank,
)
from orchestrator.shared.research import Column


class MockAnalyzer:
    def __init__(
        self, families: List[List[str] | str], runtime_priority: List[str], experiment: Any = None
    ) -> None:
        self.families = families
        self.runtime_priority = runtime_priority
        self.experiment = experiment


def test_sorting_logic_uses_structured_metadata(mocker: MockerFixture) -> None:
    # Setup mock data with explicit metadata
    # We simulate an experiment where subjects have structured metadata
    subjects_metadata = {
        "csr-react": {
            "family": "react",
            "meta_framework": None,
            "strategy": "csr",
            "runtime": "node",
        },
        "ssr-nextjs": {
            "family": "react",
            "meta_framework": "nextjs",
            "strategy": "ssr",
            "runtime": "node",
        },
        "ssr-solid-start-bun": {
            "family": "solid",
            "meta_framework": "solid-start",
            "strategy": "ssr",
            "runtime": "bun",
        },
        "ssr-solid-start-node": {
            "family": "solid",
            "meta_framework": "solid-start",
            "strategy": "ssr",
            "runtime": "node",
        },
    }

    # In the real system, this metadata will be in experiment.metadata["subjects"]
    # or we might need a better way to look it up.
    # For now, let's assume we update the Experiment object to have a subjects lookup.

    mock_experiment = mocker.Mock()
    # We'll need to define how the experiment stores this.
    # Let's say experiment.subject_metadata is a dict mapping tech_id -> metadata
    mock_experiment.subject_metadata = subjects_metadata

    analyzer = MockAnalyzer(
        families=[["vanilla"], ["solid"], ["react"]],
        runtime_priority=["bun", "node"],
        experiment=mock_experiment,
    )

    df = pd.DataFrame(
        {
            Column.SERVER_TYPE: [
                "ssr-nextjs",
                "csr-react",
                "ssr-solid-start-node",
                "ssr-solid-start-bun",
            ]
        }
    )

    ordered = get_ordered_tech_list(analyzer, df)  # type: ignore[arg-type]

    # Expected order:
    # 1. Solid Family (Family index 1)
    #    - SSR (Strategy 1)
    #    - Solid-start (Meta)
    #    - Bun (Runtime index 0) -> ssr-solid-start-bun
    #    - Node (Runtime index 1) -> ssr-solid-start-node
    # 2. React Family (Family index 2)
    #    - CSR (Strategy 0)
    #    - Pure (Meta None) -> csr-react
    #    - SSR (Strategy 1)
    #    - NextJS (Meta) -> ssr-nextjs

    expected = ["ssr-solid-start-bun", "ssr-solid-start-node", "csr-react", "ssr-nextjs"]
    assert ordered == expected


def test_order_techs_by_avg_rank_sorts_columns_and_preserves_values() -> None:
    # rows = metrics, columns = technologies (post-transpose shape from
    # compute_scorecard_and_winner). Lower average rank = better.
    scorecard_ranks_df = pd.DataFrame(
        {
            "tech_best": [1.0, 2.0, 1.0],  # avg 1.33
            "tech_worst": [3.0, 3.0, 3.0],  # avg 3.0
            "tech_mid": [2.0, 1.0, 2.0],  # avg 1.67
        },
        index=["metric_a", "metric_b", "metric_c"],
    )

    ranks_ordered = order_techs_by_avg_rank(scorecard_ranks_df)

    # Regression guard for axis=1 bug: under that bug, reindex selects
    # metric names as columns, producing an all-NaN frame.
    assert not ranks_ordered.isna().all().all()
    # Values must be preserved (not NaN) after reindex.
    assert ranks_ordered.notna().all().all()
    # Columns ordered by ascending mean rank.
    assert list(ranks_ordered.columns) == ["tech_best", "tech_mid", "tech_worst"]
    # Row order (metrics) preserved.
    assert list(ranks_ordered.index) == ["metric_a", "metric_b", "metric_c"]
    # Cell values match the source.
    assert ranks_ordered.loc["metric_a", "tech_best"] == 1.0
    assert ranks_ordered.loc["metric_b", "tech_mid"] == 1.0
