from pathlib import Path
from typing import Any

import pytest

from orchestrator.actions.verify.provider import run_verify


def test_run_verify_fails_if_subject_json_missing(tmp_path: Path, mocker: Any) -> None:
    # Setup mock subjects directory
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()

    subject1 = subjects_dir / "csr-subject"
    subject1.mkdir()
    (subject1 / "Dockerfile").write_text("FROM scratch")
    # subject.json is missing!

    # Mock SUBJECTS_DIR and console
    mocker.patch("orchestrator.actions.verify.provider.SUBJECTS_DIR", subjects_dir)
    mocker.patch("orchestrator.actions.verify.provider.console")

    # We expect it to raise a ValueError or similar, or at least log an error and stop
    # Given our "Fail Fast" agreement, it should probably raise an exception
    with pytest.raises(ValueError, match="Missing subject.json"):
        run_verify(subject_filter=None)
