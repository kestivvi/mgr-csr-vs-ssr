import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from orchestrator.shared.infra.ansible import AnsibleAdapter
from orchestrator.shared.infra.exceptions import TerraformError
from orchestrator.shared.infra.terraform import TerraformAdapter


@pytest.fixture
def mock_popen(mocker: MockerFixture) -> MagicMock:
    """Mocks subprocess.Popen to avoid spawning real system processes."""
    mock_process = mocker.Mock()
    mock_process.stdout.readline.side_effect = ["line 1\n", "line 2\n", ""]
    mock_process.wait.return_value = 0
    return mocker.patch("subprocess.Popen", return_value=mock_process)


def test_terraform_adapter_init(mock_popen: MagicMock, tmp_path: Path) -> None:
    """Verifies that TerraformAdapter.init formats and runs the correct command."""
    adapter = TerraformAdapter(tmp_path)
    adapter.init(verbose=True)

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    command = args[0]
    assert command == ["terraform", "init"]
    assert kwargs["cwd"] == str(tmp_path)


def test_terraform_adapter_apply_maps_vars(mock_popen: MagicMock, tmp_path: Path) -> None:
    """Verifies that TerraformAdapter.apply serializes variables to CLI flags."""
    adapter = TerraformAdapter(tmp_path)
    variables = {
        "region": "us-west-2",
        "tags": {"environment": "production"},
        "count": 3,
    }

    adapter.apply(variables, verbose=False)

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    command = args[0]

    # Must contain main command and options
    assert command[0:3] == ["terraform", "apply", "-auto-approve"]
    # Check variable serializations
    assert "-var" in command
    assert "region=us-west-2" in command
    assert f"tags={json.dumps({'environment': 'production'})}" in command
    assert "count=3" in command
    assert kwargs["env"]["TF_IN_AUTOMATION"] == "true"


def test_terraform_adapter_destroy(mock_popen: MagicMock, tmp_path: Path) -> None:
    """Verifies that TerraformAdapter.destroy formats the teardown command."""
    adapter = TerraformAdapter(tmp_path)
    adapter.destroy(verbose=False)

    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    assert args[0] == ["terraform", "destroy", "-auto-approve"]


def test_terraform_adapter_failure_raises(mocker: MockerFixture, tmp_path: Path) -> None:
    """Verifies that subprocess failures correctly raise a TerraformError exception."""
    mock_process = mocker.Mock()
    mock_process.stdout.readline.side_effect = ["error output\n", ""]
    mock_process.wait.return_value = 1
    mocker.patch("subprocess.Popen", return_value=mock_process)

    adapter = TerraformAdapter(tmp_path)
    with pytest.raises(TerraformError) as exc_info:
        adapter.init()

    assert "Command failed: terraform init" in str(exc_info.value)
    assert exc_info.value.return_code == 1
    assert exc_info.value.logs is not None
    assert "error output" in exc_info.value.logs


def test_ansible_adapter_run_playbook(mock_popen: MagicMock, tmp_path: Path) -> None:
    """Verifies that AnsibleAdapter.run_playbook constructs CLI calls correctly."""
    adapter = AnsibleAdapter(tmp_path)
    extra_vars = {"app_version": "1.2.3", "debug": True}

    adapter.run_playbook(
        playbook="playbooks/site.yml",
        inventory="inventory/hosts.ini",
        extra_vars=extra_vars,
    )

    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    command = args[0]

    assert command[0:2] == ["ansible-playbook", "playbooks/site.yml"]
    assert "-i" in command
    assert "inventory/hosts.ini" in command
    assert "--extra-vars" in command
    assert json.dumps(extra_vars) in command
