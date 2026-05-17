import json
from pathlib import Path

import pytest

from orchestrator.shared.research.subject import Subject, SubjectRegistry


@pytest.fixture
def mock_subjects_dir(tmp_path: Path) -> Path:
    """Creates a temporary subjects directory with some valid subjects."""
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()

    # 1. Valid CSR Vanilla Nginx Subject
    s1 = subjects_dir / "csr-vanilla-nginx"
    s1.mkdir()
    with open(s1 / "subject.json", "w") as f:
        json.dump(
            {
                "strategy": "csr",
                "family": "vanilla",
                "meta_framework": None,
                "runtime": "nginx",
            },
            f,
        )

    # 2. Valid SSR Next.js Node Subject
    s2 = subjects_dir / "ssr-nextjs-node"
    s2.mkdir()
    with open(s2 / "subject.json", "w") as f:
        json.dump(
            {
                "strategy": "ssr",
                "family": "react",
                "meta_framework": "nextjs",
                "runtime": "node",
            },
            f,
        )

    # 3. Valid SSR Astro React Bun Subject (Framework-agnostic metaframework)
    s3 = subjects_dir / "ssr-astro-react-bun"
    s3.mkdir()
    with open(s3 / "subject.json", "w") as f:
        json.dump(
            {
                "strategy": "ssr",
                "family": "react",
                "meta_framework": "astro",
                "runtime": "bun",
            },
            f,
        )

    return subjects_dir


def test_subject_properties(mock_subjects_dir: Path) -> None:
    """Verifies that properties (slug, display_name) are computed correctly."""
    subject_path = mock_subjects_dir / "ssr-astro-react-bun"
    subject = Subject.from_path(subject_path)

    assert subject.id == "ssr-astro-react-bun"
    assert subject.slug == "ssr_astro_react_bun"
    # Display name matches uppercase strategy, capitalized framework/metaframework and runtime
    assert subject.display_name == "SSR Astro React (Bun)"


def test_subject_registry_discovers_valid_subjects(mock_subjects_dir: Path) -> None:
    """Verifies that SubjectRegistry discovers and loads only valid subjects."""
    registry = SubjectRegistry(mock_subjects_dir)
    subjects = registry.all()

    assert len(subjects) == 3
    subject_ids = [s.id for s in subjects]
    assert "csr-vanilla-nginx" in subject_ids
    assert "ssr-nextjs-node" in subject_ids
    assert "ssr-astro-react-bun" in subject_ids


def test_subject_registry_grouping(mock_subjects_dir: Path) -> None:
    """Verifies that get_default_groups clusters strategy and runtime correctly."""
    registry = SubjectRegistry(mock_subjects_dir)
    groups = registry.get_default_groups()

    assert "CSR-Nginx" in groups
    assert "csr-vanilla-nginx" in groups["CSR-Nginx"]
    assert "SSR-Node" in groups
    assert "ssr-nextjs-node" in groups["SSR-Node"]
    assert "SSR-Bun" in groups
    assert "ssr-astro-react-bun" in groups["SSR-Bun"]


def test_naming_contract_enforces_strategy_prefix(mock_subjects_dir: Path) -> None:
    """Naming Contract Rule 3: All Subject IDs must start with csr- or ssr-."""
    invalid_dir = mock_subjects_dir / "bad-nextjs-node"
    invalid_dir.mkdir()
    with open(invalid_dir / "subject.json", "w") as f:
        json.dump(
            {
                "strategy": "bad",  # Invalid strategy
                "family": "react",
                "meta_framework": "nextjs",
                "runtime": "node",
            },
            f,
        )

    # Directly loading should raise ValueError
    with pytest.raises(ValueError, match="Strategy must start with 'csr-' or 'ssr-'"):
        Subject.from_path(invalid_dir)

    # Registry should ignore it and continue
    registry = SubjectRegistry(mock_subjects_dir)
    assert "bad-nextjs-node" not in [s.id for s in registry.all()]


def test_naming_contract_enforces_runtime_suffix(mock_subjects_dir: Path) -> None:
    """Naming Contract Rule 4: All Subject IDs must end with the production runtime suffix."""
    invalid_dir = mock_subjects_dir / "ssr-nextjs-invalid"
    invalid_dir.mkdir()
    with open(invalid_dir / "subject.json", "w") as f:
        json.dump(
            {
                "strategy": "ssr",
                "family": "react",
                "meta_framework": "nextjs",
                "runtime": "invalid",  # Invalid runtime
            },
            f,
        )

    # Directly loading should raise ValueError
    with pytest.raises(ValueError, match="Runtime suffix must end with a valid server runtime"):
        Subject.from_path(invalid_dir)

    # Registry should ignore it and continue
    registry = SubjectRegistry(mock_subjects_dir)
    assert "ssr-nextjs-invalid" not in [s.id for s in registry.all()]


def test_naming_contract_folder_matches_id(mock_subjects_dir: Path) -> None:
    """Naming Contract Rule 2: The folder name must match the Subject ID exactly."""
    # Create folder ssr-nextjs-node but with mismatching manifest metadata
    mismatch_dir = mock_subjects_dir / "ssr-nextjs-node"
    # We overwrite the subject.json in the existing valid dir to represent a mismatch
    with open(mismatch_dir / "subject.json", "w") as f:
        json.dump(
            {
                "strategy": "csr",  # Mismatches the folder prefix 'ssr-'
                "family": "react",
                "meta_framework": "nextjs",
                "runtime": "node",
            },
            f,
        )

    # Directly loading should raise ValueError
    with pytest.raises(ValueError, match="Folder name does not match Subject ID parts"):
        Subject.from_path(mismatch_dir)

    # Registry should ignore it and continue
    registry = SubjectRegistry(mock_subjects_dir)
    assert "ssr-nextjs-node" not in [s.id for s in registry.all()]
