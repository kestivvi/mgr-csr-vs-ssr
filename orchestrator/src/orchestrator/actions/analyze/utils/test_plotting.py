import pytest
import pandas as pd
from typing import List
from orchestrator.actions.analyze.utils.plotting import get_ordered_tech_list
from orchestrator.shared.research import Column

class MockAnalyzer:
    def __init__(self, families: List[List[str]], runtime_priority: List[str], experiment=None):
        self.families = families
        self.runtime_priority = runtime_priority
        self.experiment = experiment

def test_sorting_logic_uses_structured_metadata(mocker):
    # Setup mock data with explicit metadata
    # We simulate an experiment where subjects have structured metadata
    subjects_metadata = {
        "csr-react": {"family": "react", "meta_framework": None, "strategy": "csr", "runtime": "node"},
        "ssr-nextjs": {"family": "react", "meta_framework": "nextjs", "strategy": "ssr", "runtime": "node"},
        "ssr-solid-start-bun": {"family": "solid", "meta_framework": "solid-start", "strategy": "ssr", "runtime": "bun"},
        "ssr-solid-start-node": {"family": "solid", "meta_framework": "solid-start", "strategy": "ssr", "runtime": "node"},
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
        experiment=mock_experiment
    )
    
    df = pd.DataFrame({
        Column.SERVER_TYPE: ["ssr-nextjs", "csr-react", "ssr-solid-start-node", "ssr-solid-start-bun"]
    })
    
    ordered = get_ordered_tech_list(analyzer, df)
    
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
