from pathlib import Path

import pytest
import yaml

from orchestrator.shared.research.artifact import ResearchArtifact


@pytest.fixture
def mock_artifact_dir(tmp_path: Path) -> Path:
    """Creates a temporary artifact directory with sample metrics, results, and logs."""
    artifact_dir = tmp_path / "study_run"
    artifact_dir.mkdir()

    # Create subfolders
    (artifact_dir / "metrics").mkdir()
    (artifact_dir / "tool_results").mkdir()
    (artifact_dir / "logs").mkdir()

    # Create repetitive runs for ssr-nextjs-node and csr-vanilla-nginx
    # Repetition 1: Complete
    with open(artifact_dir / "metrics" / "01_ssr_nextjs_node.csv", "w") as f:
        f.write("time,cpu,mem\n0,10.0,50.0")
    with open(artifact_dir / "tool_results" / "01_ssr_nextjs_node_wrk.json", "w") as f:
        f.write('{"rps": 1000}')
    with open(artifact_dir / "logs" / "01_ssr_nextjs_node.log", "w") as f:
        f.write("logs run 1")

    # Repetition 2: Complete
    with open(artifact_dir / "metrics" / "02_ssr_nextjs_node.csv", "w") as f:
        f.write("time,cpu,mem\n0,12.0,52.0")
    with open(artifact_dir / "tool_results" / "02_ssr_nextjs_node_wrk.json", "w") as f:
        f.write('{"rps": 1050}')
    with open(artifact_dir / "logs" / "02_ssr_nextjs_node.log", "w") as f:
        f.write("logs run 2")

    # Repetition 1: csr-vanilla-nginx Complete
    with open(artifact_dir / "metrics" / "01_csr_vanilla_nginx.csv", "w") as f:
        f.write("time,cpu,mem\n0,5.0,20.0")
    with open(artifact_dir / "tool_results" / "01_csr_vanilla_nginx_wrk.json", "w") as f:
        f.write('{"rps": 5000}')

    # Create study metadata file
    meta = {
        "test_type": "capacity_k6",
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 1000,
            "warmup_duration": "60s",
            "capacity_k6_options": {
                "peak_rate": 5000,
                "max_vus": 400,
                "ramp_up": "10m",
                "sustain": "1m",
            },
        },
    }
    with open(artifact_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)

    return artifact_dir


def test_research_artifact_loads_metadata(mock_artifact_dir: Path) -> None:
    """Verifies that ResearchArtifact correctly loads metadata.yaml."""
    artifact = ResearchArtifact(mock_artifact_dir)
    assert artifact.metadata["test_type"] == "capacity_k6"
    assert artifact.metadata["parameters"]["rate"] == 1000
    assert artifact.is_consistent is True


def test_research_artifact_get_repetitions(mock_artifact_dir: Path) -> None:
    """Verifies that get_repetitions correctly parses, groups, and sorts run files."""
    artifact = ResearchArtifact(mock_artifact_dir)
    runs = artifact.get_repetitions()

    # We expect 3 distinct runs (2 for ssr-nextjs-node, 1 for csr-vanilla-nginx)
    assert len(runs) == 3

    # Check sorting (sorted by repetition_id, then server_type)
    r1 = runs[0]
    assert r1.repetition_id == 1
    assert r1.server_type == "csr-vanilla-nginx"
    assert r1.metrics_path is not None
    assert r1.results_path is not None
    assert r1.metrics_path.name == "01_csr_vanilla_nginx.csv"
    assert r1.results_path.name == "01_csr_vanilla_nginx_wrk.json"
    assert r1.logs_path is None
    assert r1.is_complete is True

    r2 = runs[1]
    assert r2.repetition_id == 1
    assert r2.server_type == "ssr-nextjs-node"
    assert r2.metrics_path is not None
    assert r2.results_path is not None
    assert r2.logs_path is not None
    assert r2.metrics_path.name == "01_ssr_nextjs_node.csv"
    assert r2.results_path.name == "01_ssr_nextjs_node_wrk.json"
    assert r2.logs_path.name == "01_ssr_nextjs_node.log"
    assert r2.is_complete is True

    r3 = runs[2]
    assert r3.repetition_id == 2
    assert r3.server_type == "ssr-nextjs-node"
    assert r3.metrics_path is not None
    assert r3.results_path is not None
    assert r3.logs_path is not None
    assert r3.metrics_path.name == "02_ssr_nextjs_node.csv"
    assert r3.results_path.name == "02_ssr_nextjs_node_wrk.json"
    assert r3.logs_path.name == "02_ssr_nextjs_node.log"
    assert r3.is_complete is True


def test_research_artifact_filtering(mock_artifact_dir: Path) -> None:
    """Verifies that subject inclusion and exclusion filtering works correctly."""
    artifact = ResearchArtifact(mock_artifact_dir)

    # 1. Include only nextjs
    runs_inc = artifact.get_repetitions(include_subjects={"ssr-nextjs-node"})
    assert len(runs_inc) == 2
    for r in runs_inc:
        assert r.server_type == "ssr-nextjs-node"

    # 2. Exclude nextjs
    runs_exc = artifact.get_repetitions(exclude_subjects={"ssr-nextjs-node"})
    assert len(runs_exc) == 1
    assert runs_exc[0].server_type == "csr-vanilla-nginx"


def test_research_artifact_compatibility_check(mock_artifact_dir: Path) -> None:
    """Verifies that check_compatibility correctly catches contract violations."""
    artifact = ResearchArtifact(mock_artifact_dir)

    # 1. Perfectly compatible other dict
    other_ok = {
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 1000,
            "warmup_duration": "60s",
            "capacity_k6_options": {
                "peak_rate": 5000,
                "max_vus": 400,
                "ramp_up": "10m",
                "sustain": "1m",
            },
        }
    }
    assert len(artifact.check_compatibility(other_ok)) == 0

    # 2. Violates global rate parameter
    other_bad_global = {
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 2000,  # 2000 vs 1000
            "warmup_duration": "60s",
            "capacity_k6_options": {
                "peak_rate": 5000,
                "max_vus": 400,
                "ramp_up": "10m",
                "sustain": "1m",
            },
        }
    }
    diffs = artifact.check_compatibility(other_bad_global)
    assert len(diffs) == 1
    assert "Global Contract violation: rate differs" in diffs[0]

    # 3. Violates tool option parameters
    other_bad_k6 = {
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 1000,
            "warmup_duration": "60s",
            "capacity_k6_options": {
                "peak_rate": 5000,
                "max_vus": 100,  # 100 vs 400
                "ramp_up": "10m",
                "sustain": "2m",  # 2m vs 1m
            },
        }
    }
    diffs = artifact.check_compatibility(other_bad_k6)
    assert len(diffs) == 2
    assert any("max_vus differs" in d for d in diffs)
    assert any("sustain differs" in d for d in diffs)
