import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from orchestrator.actions.test.runner import TestRunner as OrchestratorTestRunner


def test_runner_captures_application_manifests_in_metadata(tmp_path: Path, mocker: Any) -> None:
    # 1. Setup mock environment files (Filesystem Boundary)
    apps_dir = tmp_path / "applications"
    apps_dir.mkdir()

    app1_dir = apps_dir / "ssr-solid"
    app1_dir.mkdir()
    manifest = {
        "family": "solid",
        "meta_framework": "solid-start",
        "strategy": "ssr",
        "runtime": "node",
    }
    with open(app1_dir / "application.json", "w") as f:
        json.dump(manifest, f)

    inventory_yaml = tmp_path / "inventory.yaml"
    inventory_data = {
        "all": {
            "children": {
                "role_monitoring_host": {
                    "hosts": {
                        "mon1": {
                            "public_ip": "5.6.7.8",
                            "private_ip": "10.0.0.1",
                        }
                    }
                },
                "application_server_ssr-solid": {
                    "hosts": {
                        "server1": {
                            "private_ip": "1.2.3.4",
                        }
                    }
                },
            }
        }
    }
    with open(inventory_yaml, "w") as f:
        yaml.dump(inventory_data, f)

    # 2. Patch system boundary constants in runner module
    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_INVENTORY", inventory_yaml)
    mocker.patch("orchestrator.actions.test.runner.APPLICATIONS_DIR", apps_dir)
    mocker.patch("orchestrator.actions.test.runner.RESULTS_DIR", tmp_path / "results")
    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_DIR", tmp_path / "ansible")

    # 3. Mock external execution boundaries (ansible_runner and collector)
    mock_run_res = mocker.Mock()
    mock_run_res.status = "successful"
    mock_ar_run = mocker.patch("ansible_runner.run", return_value=mock_run_res)

    mock_thread = mocker.Mock()
    mock_async_res = mocker.Mock()
    mock_async_res.status = "successful"
    mock_async_res.stdout.read.return_value = (
        'ORCHESTRATOR_TIMESTAMPS::{"start": 100.0, "end": 200.0}'
    )
    mock_ar_run_async = mocker.patch(
        "ansible_runner.run_async", return_value=(mock_thread, mock_async_res)
    )

    mock_collect = mocker.patch("orchestrator.actions.test.runner.collect_metrics")

    # 4. Setup OrchestratorTestRunner and execute run_all
    runner = OrchestratorTestRunner(
        config_path=None,
        config_dict={"test_type": "capacity_k6", "auto_approve": True},
        output_dir=tmp_path / "result",
    )

    runner.run_all()

    # 5. Verify actual SUT integration behaviors
    # Check that ansible_runner was executed for script sync
    mock_ar_run.assert_called_once()
    assert mock_ar_run.call_args[1]["playbook"] == "ops/test_sync_script.yml"

    # Check that ansible_runner was executed asynchronously for capacity scenario
    mock_ar_run_async.assert_called_once()
    call_kwargs = mock_ar_run_async.call_args[1]
    assert call_kwargs["playbook"] == "ops/test_capacity_run.yml"
    assert call_kwargs["extravars"]["server_type"] == "ssr-solid"
    assert call_kwargs["extravars"]["target_url"] == "https://1.2.3.4"

    # Check that telemetry collector was triggered
    mock_collect.assert_called_once_with(
        prometheus_url="http://5.6.7.8:9090",
        start_epoch=100,
        end_epoch=200,
        server_type="ssr-solid",
        repetition_number=1,
        output_dir=tmp_path / "result",
    )

    # Verify final compiled metadata on the filesystem contains parsed application manifest
    meta_path = tmp_path / "result" / "metadata.yaml"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    assert meta["test_type"] == "capacity_k6"
    assert "applications" in meta
    assert meta["applications"]["ssr-solid"] == manifest


def test_runner_raises_value_error_if_manifest_missing(tmp_path: Path, mocker: Any) -> None:
    # 1. Setup mock environment files without application.json manifest
    apps_dir = tmp_path / "applications"
    apps_dir.mkdir()

    app1_dir = apps_dir / "ssr-solid"
    app1_dir.mkdir()
    # manifest is missing!

    inventory_yaml = tmp_path / "inventory.yaml"
    inventory_data = {
        "all": {
            "children": {
                "role_monitoring_host": {
                    "hosts": {
                        "mon1": {
                            "public_ip": "5.6.7.8",
                            "private_ip": "10.0.0.1",
                        }
                    }
                },
                "application_server_ssr-solid": {
                    "hosts": {
                        "server1": {
                            "private_ip": "1.2.3.4",
                        }
                    }
                },
            }
        }
    }
    with open(inventory_yaml, "w") as f:
        yaml.dump(inventory_data, f)

    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_INVENTORY", inventory_yaml)
    mocker.patch("orchestrator.actions.test.runner.APPLICATIONS_DIR", apps_dir)

    runner = OrchestratorTestRunner(
        config_path=None,
        config_dict={"test_type": "capacity_k6", "auto_approve": True},
        output_dir=tmp_path / "result",
    )

    with pytest.raises(ValueError, match="Missing application.json in ssr-solid"):
        runner.run_all()


def test_runner_filters_applications(tmp_path: Path, mocker: Any) -> None:
    # 1. Setup mock environment files with multiple applications
    apps_dir = tmp_path / "applications"
    apps_dir.mkdir()

    # Application 1 (solid)
    app1_dir = apps_dir / "ssr-solid"
    app1_dir.mkdir()
    with open(app1_dir / "application.json", "w") as f:
        json.dump({"strategy": "ssr", "runtime": "node"}, f)

    # Application 2 (react)
    app2_dir = apps_dir / "ssr-react"
    app2_dir.mkdir()
    with open(app2_dir / "application.json", "w") as f:
        json.dump({"strategy": "ssr", "runtime": "node"}, f)

    inventory_yaml = tmp_path / "inventory.yaml"
    inventory_data = {
        "all": {
            "children": {
                "role_monitoring_host": {
                    "hosts": {
                        "mon1": {
                            "public_ip": "5.6.7.8",
                            "private_ip": "10.0.0.1",
                        }
                    }
                },
                "application_server_ssr-solid": {
                    "hosts": {
                        "server1": {
                            "private_ip": "1.2.3.4",
                        }
                    }
                },
                "application_server_ssr-react": {
                    "hosts": {
                        "server2": {
                            "private_ip": "1.2.3.5",
                        }
                    }
                },
            }
        }
    }
    with open(inventory_yaml, "w") as f:
        yaml.dump(inventory_data, f)

    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_INVENTORY", inventory_yaml)
    mocker.patch("orchestrator.actions.test.runner.APPLICATIONS_DIR", apps_dir)
    mocker.patch("orchestrator.actions.test.runner.RESULTS_DIR", tmp_path / "results")
    mocker.patch("orchestrator.actions.test.runner.ANSIBLE_DIR", tmp_path / "ansible")

    # Mocks
    mock_run_res = mocker.Mock()
    mock_run_res.status = "successful"
    mocker.patch("ansible_runner.run", return_value=mock_run_res)

    mock_thread = mocker.Mock()
    mock_async_res = mocker.Mock()
    mock_async_res.status = "successful"
    mock_async_res.stdout.read.return_value = (
        'ORCHESTRATOR_TIMESTAMPS::{"start": 100.0, "end": 200.0}'
    )
    mock_ar_run_async = mocker.patch(
        "ansible_runner.run_async", return_value=(mock_thread, mock_async_res)
    )

    mocker.patch("orchestrator.actions.test.runner.collect_metrics")

    # Filter only react!
    runner = OrchestratorTestRunner(
        config_path=None,
        config_dict={"test_type": "capacity_k6", "auto_approve": True},
        output_dir=tmp_path / "result",
        apps_filter="react",
    )

    runner.run_all()

    # Check that ansible_runner was only called for ssr-react
    mock_ar_run_async.assert_called_once()
    call_kwargs = mock_ar_run_async.call_args[1]
    assert call_kwargs["extravars"]["server_type"] == "ssr-react"
