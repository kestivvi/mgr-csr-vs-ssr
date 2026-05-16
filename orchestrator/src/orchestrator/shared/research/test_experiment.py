from pathlib import Path
from typing import Any
import pytest
from orchestrator.shared.research.experiment import ExperimentLoader
from orchestrator.shared.research.artifact import ResearchArtifact


@pytest.fixture
def mock_artifact(tmp_path: Path) -> ResearchArtifact:
    """Creates a mock artifact directory with a metadata.yaml including subject data."""
    meta_path = tmp_path / "metadata.yaml"
    # Note: We are using metadata.yaml as per current implementation,
    # but the PRD mentions subject.json in the app folder.
    # The ExperimentLoader should ideally have access to the subject manifest
    # that was captured during 'mgr test' and stored in the artifact's metadata.
    import yaml

    subject_data = {
        "family": "solid",
        "meta_framework": "solid-start",
        "strategy": "ssr",
        "runtime": "bun",
    }

    metadata = {"test_type": "capacity_k6", "subjects": {"mock-subject": subject_data}}

    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

    return ResearchArtifact(tmp_path)


def test_experiment_loader_captures_structured_subject_metadata(
    mock_artifact: ResearchArtifact,
) -> None:
    loader = ExperimentLoader()
    experiment = loader.load(mock_artifact)

    assert "subjects" in experiment.metadata
    assert experiment.metadata["subjects"]["mock-subject"]["family"] == "solid"
    assert experiment.metadata["subjects"]["mock-subject"]["meta_framework"] == "solid-start"
    assert experiment.metadata["subjects"]["mock-subject"]["strategy"] == "ssr"
    assert experiment.metadata["subjects"]["mock-subject"]["runtime"] == "bun"


def test_experiment_loader_legacy_fallback_fails_fast(tmp_path: Path) -> None:
    """If subject metadata is missing, it should fail (as per our Fail-Fast agreement)."""
    meta_path = tmp_path / "metadata.yaml"
    import yaml

    with open(meta_path, "w") as f:
        yaml.dump({"test_type": "capacity_k6"}, f)

    artifact = ResearchArtifact(tmp_path)
    loader = ExperimentLoader()

    with pytest.raises(ValueError, match="Missing structured subject metadata"):
        loader.load(artifact)
