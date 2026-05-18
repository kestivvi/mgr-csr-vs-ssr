from pathlib import Path

import pytest

from orchestrator.shared.research.artifact import ResearchArtifact
from orchestrator.shared.research.experiment import ExperimentLoader


@pytest.fixture
def mock_artifact(tmp_path: Path) -> ResearchArtifact:
    """Creates a mock artifact directory with a metadata.yaml including application data."""
    meta_path = tmp_path / "metadata.yaml"
    import yaml

    application_data = {
        "family": "solid",
        "meta_framework": "solid-start",
        "strategy": "ssr",
        "runtime": "bun",
    }

    metadata = {"test_type": "capacity_k6", "applications": {"mock-application": application_data}}

    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

    return ResearchArtifact(tmp_path)


def test_experiment_loader_captures_structured_application_metadata(
    mock_artifact: ResearchArtifact,
) -> None:
    loader = ExperimentLoader()
    experiment = loader.load(mock_artifact)

    assert "applications" in experiment.metadata
    assert experiment.metadata["applications"]["mock-application"]["family"] == "solid"
    assert (
        experiment.metadata["applications"]["mock-application"]["meta_framework"] == "solid-start"
    )
    assert experiment.metadata["applications"]["mock-application"]["strategy"] == "ssr"
    assert experiment.metadata["applications"]["mock-application"]["runtime"] == "bun"


def test_experiment_loader_legacy_fallback_fails_fast(tmp_path: Path) -> None:
    """If application metadata is missing, it should fail (as per our Fail-Fast agreement)."""
    meta_path = tmp_path / "metadata.yaml"
    import yaml

    with open(meta_path, "w") as f:
        yaml.dump({"test_type": "capacity_k6"}, f)

    artifact = ResearchArtifact(tmp_path)
    loader = ExperimentLoader()

    with pytest.raises(ValueError, match="Missing structured application metadata"):
        loader.load(artifact)
