from pathlib import Path
from typing import Any

import pytest

from orchestrator.actions.verify.provider import run_verify


def test_run_verify_fails_if_application_json_missing(tmp_path: Path, mocker: Any) -> None:
    # Setup mock applications directory
    apps_dir = tmp_path / "applications"
    apps_dir.mkdir()

    app1 = apps_dir / "csr-application"
    app1.mkdir()
    (app1 / "Dockerfile").write_text("FROM scratch")
    # application.json is missing!

    # Mock APPLICATIONS_DIR and console
    mocker.patch("orchestrator.actions.verify.provider.APPLICATIONS_DIR", apps_dir)
    mocker.patch("orchestrator.actions.verify.provider.console")

    # We expect it to raise a ValueError or similar, or at least log an error and stop
    # Given our "Fail Fast" agreement, it should probably raise an exception
    with pytest.raises(ValueError, match="Missing application.json"):
        run_verify(app_filter=None)
