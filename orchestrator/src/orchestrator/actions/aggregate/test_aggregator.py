from pathlib import Path

import pytest
import yaml

from orchestrator.actions.aggregate.engine import DataAggregator


@pytest.fixture
def sample_study_directories(tmp_path: Path) -> tuple[Path, Path]:
    """Sets up two mock study artifacts with distinct repetitions and identical contracts."""
    art_a = tmp_path / "study_a"
    art_a.mkdir()
    (art_a / "metrics").mkdir()
    (art_a / "tool_results").mkdir()
    (art_a / "logs").mkdir()

    # Study A: 2 runs for ssr-nextjs-node
    with open(art_a / "metrics" / "01_ssr_nextjs_node.csv", "w") as f:
        f.write("A1")
    with open(art_a / "tool_results" / "01_ssr_nextjs_node_wrk.json", "w") as f:
        f.write("A1_wrk")
    with open(art_a / "logs" / "01_ssr_nextjs_node.log", "w") as f:
        f.write("A1_logs")

    with open(art_a / "metrics" / "02_ssr_nextjs_node.csv", "w") as f:
        f.write("A2")
    with open(art_a / "tool_results" / "02_ssr_nextjs_node_wrk.json", "w") as f:
        f.write("A2_wrk")

    meta_a = {
        "test_type": "capacity_k6",
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 1000,
            "warmup_duration": "60s",
        },
    }
    with open(art_a / "metadata.yaml", "w") as f:
        yaml.dump(meta_a, f)

    # Study B: 1 run for ssr-nextjs-node and 1 run for csr-vanilla-nginx
    art_b = tmp_path / "study_b"
    art_b.mkdir()
    (art_b / "metrics").mkdir()
    (art_b / "tool_results").mkdir()
    (art_b / "logs").mkdir()

    with open(art_b / "metrics" / "01_ssr_nextjs_node.csv", "w") as f:
        f.write("B1")
    with open(art_b / "tool_results" / "01_ssr_nextjs_node_wrk.json", "w") as f:
        f.write("B1_wrk")

    with open(art_b / "metrics" / "01_csr_vanilla_nginx.csv", "w") as f:
        f.write("B2")
    with open(art_b / "tool_results" / "01_csr_vanilla_nginx_wrk.json", "w") as f:
        f.write("B2_wrk")

    meta_b = {
        "test_type": "capacity_k6",
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 1000,
            "warmup_duration": "60s",
        },
    }
    with open(art_b / "metadata.yaml", "w") as f:
        yaml.dump(meta_b, f)

    return art_a, art_b


def test_aggregator_merges_compatible_runs(
    tmp_path: Path, sample_study_directories: tuple[Path, Path]
) -> None:
    """Verifies that DataAggregator correctly chains repetitions and maps lineages."""
    art_a, art_b = sample_study_directories
    out_dir = tmp_path / "aggregated_output"

    aggregator = DataAggregator(
        source_specs=[str(art_a), str(art_b)], output_dir=out_dir, copy_logs=True
    )
    aggregator.run()

    # 1. Assert copied metrics file sequences (A1->01, A2->02, B1->03, B2->04)
    # A: ssr-nextjs-node (01, 02) -> 01_ssr_nextjs_node.csv, 02_ssr_nextjs_node.csv
    # B: ssr-nextjs-node (01) -> 03_ssr_nextjs_node.csv
    # B: csr-vanilla-nginx (01) -> 03_csr_vanilla_nginx.csv (global ID = 3)
    assert (out_dir / "metrics" / "01_ssr_nextjs_node.csv").exists()
    assert (out_dir / "metrics" / "02_ssr_nextjs_node.csv").exists()
    assert (out_dir / "metrics" / "03_ssr_nextjs_node.csv").exists()
    assert (out_dir / "metrics" / "03_csr_vanilla_nginx.csv").exists()

    # Logs should copy
    assert (out_dir / "logs" / "01_ssr_nextjs_node.log").exists()

    # 2. Check metadata
    with open(out_dir / "metadata.yaml", "r") as f:
        master_meta = yaml.safe_load(f)

    assert master_meta["num_repetitions"] == 3
    assert master_meta["is_consistent"] is True
    assert len(master_meta["lineage"]) == 2

    lin_a = master_meta["lineage"][0]
    assert lin_a["source"] == str(art_a.absolute())
    assert lin_a["mapped_repetitions"] == {1: 1, 2: 2}

    lin_b = master_meta["lineage"][1]
    assert lin_b["source"] == str(art_b.absolute())
    assert lin_b["mapped_repetitions"] == {1: 3}


def test_aggregator_filtering_subjects(
    tmp_path: Path, sample_study_directories: tuple[Path, Path]
) -> None:
    """Verifies that DataAggregator filters subjects when [include,!exclude] syntax is supplied."""
    art_a, art_b = sample_study_directories
    out_dir = tmp_path / "aggregated_filtered"

    # Aggregates only ssr-nextjs-node
    spec_a = f"{art_a}[ssr-nextjs-node]"
    spec_b = f"{art_b}[!csr-vanilla-nginx]"

    aggregator = DataAggregator(source_specs=[spec_a, spec_b], output_dir=out_dir)
    aggregator.run()

    # Only ssr-nextjs-node should exist
    assert (out_dir / "metrics" / "01_ssr_nextjs_node.csv").exists()
    assert (out_dir / "metrics" / "02_ssr_nextjs_node.csv").exists()
    assert (out_dir / "metrics" / "03_ssr_nextjs_node.csv").exists()
    assert not (out_dir / "metrics" / "03_csr_vanilla_nginx.csv").exists()


def test_aggregator_contract_violations(
    tmp_path: Path, sample_study_directories: tuple[Path, Path]
) -> None:
    """Verifies that DataAggregator rejects incompatible runs, unless lax override is set."""
    art_a, art_b = sample_study_directories
    out_dir = tmp_path / "agg_violations"

    # Set incompatible parameters for B
    with open(art_b / "metadata.yaml", "r") as f:
        meta_b = yaml.safe_load(f)
    meta_b["parameters"]["rate"] = 9999  # Violation!
    with open(art_b / "metadata.yaml", "w") as f:
        yaml.dump(meta_b, f)

    # 1. Without Lax Mode: raises ValueError
    aggregator = DataAggregator(
        source_specs=[str(art_a), str(art_b)], output_dir=out_dir, lax=False
    )
    with pytest.raises(ValueError, match="Research Contract violated. Use --lax to override."):
        aggregator.run()

    # 2. With Lax Mode: succeeds but flags metadata as inconsistent
    out_dir_lax = tmp_path / "agg_violations_lax"
    aggregator_lax = DataAggregator(
        source_specs=[str(art_a), str(art_b)], output_dir=out_dir_lax, lax=True
    )
    aggregator_lax.run()

    with open(out_dir_lax / "metadata.yaml", "r") as f:
        master_meta = yaml.safe_load(f)

    assert master_meta["is_consistent"] is False
