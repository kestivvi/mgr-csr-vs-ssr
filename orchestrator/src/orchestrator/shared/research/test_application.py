import json
from pathlib import Path

import pytest

from orchestrator.shared.research.application import Application, ApplicationRegistry


@pytest.fixture
def mock_applications_dir(tmp_path: Path) -> Path:
    """Creates a temporary applications directory with some valid applications."""
    apps_dir = tmp_path / "applications"
    apps_dir.mkdir()

    # 1. Valid CSR Vanilla Nginx Application
    s1 = apps_dir / "csr-vanilla-nginx"
    s1.mkdir()
    with open(s1 / "application.json", "w") as f:
        json.dump(
            {
                "strategy": "csr",
                "family": "vanilla",
                "meta_framework": None,
                "runtime": "nginx",
            },
            f,
        )

    # 2. Valid SSR Next.js Node Application
    s2 = apps_dir / "ssr-nextjs-node"
    s2.mkdir()
    with open(s2 / "application.json", "w") as f:
        json.dump(
            {
                "strategy": "ssr",
                "family": "react",
                "meta_framework": "nextjs",
                "runtime": "node",
            },
            f,
        )

    # 3. Valid SSR Astro React Bun Application (Framework-agnostic metaframework)
    s3 = apps_dir / "ssr-astro-react-bun"
    s3.mkdir()
    with open(s3 / "application.json", "w") as f:
        json.dump(
            {
                "strategy": "ssr",
                "family": "react",
                "meta_framework": "astro",
                "runtime": "bun",
            },
            f,
        )

    return apps_dir


def test_application_properties(mock_applications_dir: Path) -> None:
    """Verifies that properties (slug, display_name) are computed correctly."""
    app_path = mock_applications_dir / "ssr-astro-react-bun"
    app = Application.from_path(app_path)

    assert app.id == "ssr-astro-react-bun"
    assert app.slug == "ssr_astro_react_bun"
    # Display name matches uppercase strategy, capitalized framework/metaframework and runtime
    assert app.display_name == "SSR Astro React (Bun)"


def test_application_registry_discovers_valid_applications(mock_applications_dir: Path) -> None:
    """Verifies that ApplicationRegistry discovers and loads only valid applications."""
    registry = ApplicationRegistry(mock_applications_dir)
    apps = registry.all()

    assert len(apps) == 3
    app_ids = [s.id for s in apps]
    assert "csr-vanilla-nginx" in app_ids
    assert "ssr-nextjs-node" in app_ids
    assert "ssr-astro-react-bun" in app_ids


def test_application_registry_grouping(mock_applications_dir: Path) -> None:
    """Verifies that get_default_groups clusters strategy and runtime correctly."""
    registry = ApplicationRegistry(mock_applications_dir)
    groups = registry.get_default_groups()

    assert "CSR-Nginx" in groups
    assert "csr-vanilla-nginx" in groups["CSR-Nginx"]
    assert "SSR-Node" in groups
    assert "ssr-nextjs-node" in groups["SSR-Node"]
    assert "SSR-Bun" in groups
    assert "ssr-astro-react-bun" in groups["SSR-Bun"]


def test_naming_contract_enforces_strategy_prefix(mock_applications_dir: Path) -> None:
    """Naming Contract Rule 3: All Application IDs must start with csr- or ssr-."""
    invalid_dir = mock_applications_dir / "bad-nextjs-node"
    invalid_dir.mkdir()
    with open(invalid_dir / "application.json", "w") as f:
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
        Application.from_path(invalid_dir)

    # Registry should ignore it and continue
    registry = ApplicationRegistry(mock_applications_dir)
    assert "bad-nextjs-node" not in [s.id for s in registry.all()]


def test_naming_contract_enforces_runtime_suffix(mock_applications_dir: Path) -> None:
    """Naming Contract Rule 4: All Application IDs must end with the production runtime suffix."""
    invalid_dir = mock_applications_dir / "ssr-nextjs-invalid"
    invalid_dir.mkdir()
    with open(invalid_dir / "application.json", "w") as f:
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
        Application.from_path(invalid_dir)

    # Registry should ignore it and continue
    registry = ApplicationRegistry(mock_applications_dir)
    assert "ssr-nextjs-invalid" not in [s.id for s in registry.all()]


def test_naming_contract_folder_matches_id(mock_applications_dir: Path) -> None:
    """Naming Contract Rule 2: The folder name must match the Application ID exactly."""
    # Create folder ssr-nextjs-node but with mismatching manifest metadata
    mismatch_dir = mock_applications_dir / "ssr-nextjs-node"
    # We overwrite the application.json in the existing valid dir to represent a mismatch
    with open(mismatch_dir / "application.json", "w") as f:
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
    with pytest.raises(ValueError, match="Folder name does not match Application ID parts"):
        Application.from_path(mismatch_dir)

    # Registry should ignore it and continue
    registry = ApplicationRegistry(mock_applications_dir)
    assert "ssr-nextjs-node" not in [s.id for s in registry.all()]
