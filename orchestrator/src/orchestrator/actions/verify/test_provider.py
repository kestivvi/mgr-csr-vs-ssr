import pytest
from pathlib import Path
from orchestrator.actions.verify.provider import run_verify

def test_run_verify_fails_if_subject_json_missing(tmp_path, mocker):
    # Setup mock apps directory
    apps_dir = tmp_path / "apps"
    apps_dir.mkdir()
    
    app1 = apps_dir / "csr-app"
    app1.mkdir()
    (app1 / "Dockerfile").write_text("FROM scratch")
    # subject.json is missing!
    
    # Mock APPS_DIR and console
    mocker.patch("orchestrator.actions.verify.provider.APPS_DIR", apps_dir)
    mock_console = mocker.patch("orchestrator.actions.verify.provider.console")
    
    # We expect it to raise a ValueError or similar, or at least log an error and stop
    # Given our "Fail Fast" agreement, it should probably raise an exception
    with pytest.raises(ValueError, match="Missing subject.json"):
        run_verify()
