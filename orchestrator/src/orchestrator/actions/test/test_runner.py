from pathlib import Path
from typing import Any

import yaml

from orchestrator.actions.test.runner import TestRunner


def test_runner_captures_subject_manifests_in_metadata(tmp_path: Path, mocker: Any) -> None:
    # Setup mock subjects directory with manifests
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()

    subject1_dir = subjects_dir / "ssr-solid"
    subject1_dir.mkdir()
    manifest = {
        "family": "solid",
        "meta_framework": "solid-start",
        "strategy": "ssr",
        "runtime": "node",
    }
    import json

    with open(subject1_dir / "subject.json", "w") as f:
        json.dump(manifest, f)

    # Mock inventory to return this subject
    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_INVENTORY", tmp_path / "inventory.yaml")
    mocker.patch.object(
        TestRunner,
        "_parse_inventory",
        return_value=[
            {
                "name": "ssr-solid",
                "subject_server_ip": "1.2.3.4",
                "load_generator_group": "lg1",
                "monitoring_host_public_ip": "5.6.7.8",
                "monitoring_host_private_ip": "10.0.0.1",
            }
        ],
    )

    # Mock ANSIBLE_DIR and RESULTS_DIR
    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_DIR", tmp_path / "ansible")
    mocker.patch("orchestrator.actions.test.runner.RESULTS_DIR", tmp_path / "results")
    mocker.patch("orchestrator.actions.test.runner.SUBJECTS_DIR", subjects_dir)

    # Actually, I'll just mock the manifest loading logic once I implement it.
    # For now, let's see how I implement the capture.

    # Setup TestRunner
    runner = TestRunner(
        config_path=None,
        config_dict={"test_type": "capacity_k6", "auto_approve": True},
        output_dir=tmp_path / "result",
    )

    # Mock dependencies of run_all
    mocker.patch.object(runner, "_sync_k6_script", return_value=True)
    mock_scenario = {
        "name": "ssr-solid",
        "load_generator_group": "lg1",
        "subject_server_ip": "1.2.3.4",
        "monitoring_host_public_ip": "5.6.7.8",
        "monitoring_host_private_ip": "10.0.0.1",
    }
    mocker.patch.object(
        runner,
        "run_scenario",
        return_value={
            "success": True,
            "name": "ssr-solid",
            "timestamps": {"start": 100, "end": 200},
            "scenario": mock_scenario,
        },
    )
    mocker.patch("orchestrator.actions.test.runner.collect_metrics")

    runner.run_all()

    # Verify metadata.yaml
    meta_path = tmp_path / "result" / "metadata.yaml"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    assert "subjects" in meta
    assert meta["subjects"]["ssr-solid"] == manifest
