import json
from pathlib import Path

import pytest
import yaml
from pytest_mock import MockerFixture

from orchestrator.actions.campaign.provider import run_campaign


@pytest.fixture
def mock_campaign_files(tmp_path: Path) -> tuple[Path, Path]:
    """Sets up a temporary campaign yaml and a mock infrastructure config file."""
    campaign_yaml = tmp_path / "campaign.yaml"
    campaign_meta = {
        "experiment": {
            "test_type": "capacity_k6",
            "num_repetitions": 2,
            "subjects": ["ssr-nextjs-node", "csr-vanilla-nginx"],
            "capacity_k6_options": {
                "peak_rate": 1000,
                "max_vus": 100,
            },
        }
    }
    with open(campaign_yaml, "w") as f:
        yaml.dump(campaign_meta, f)

    infra_yaml = tmp_path / "infra.yaml"
    infra_meta = {
        "technologies": {
            "ssr-nextjs-node": {"runtime": "node"},
            "csr-vanilla-nginx": {"runtime": "nginx"},
        }
    }
    with open(infra_yaml, "w") as f:
        yaml.dump(infra_meta, f)

    return campaign_yaml, infra_yaml


def test_run_campaign_orchestrates_full_flow(
    mocker: MockerFixture, mock_campaign_files: tuple[Path, Path]
) -> None:
    """Verifies that run_campaign provisions, tests, and tears down each subject sequentially."""
    campaign_yaml, infra_yaml = mock_campaign_files

    # 1. Mock CloudEnvironment setup/teardown
    mock_env_class = mocker.patch("orchestrator.actions.campaign.provider.CloudEnvironment")
    mock_env_instance = mock_env_class.return_value

    # 2. Mock TestRunner
    mock_runner_class = mocker.patch("orchestrator.actions.campaign.provider.TestRunner")

    # Run the campaign
    run_campaign(
        path=campaign_yaml,
        infra_path=infra_yaml,
        verbose=True,
    )

    # 3. Assert setup was called for each subject in config
    assert mock_env_instance.setup.call_count == 2
    # Verify first call setup subject
    setup_call_1 = mock_env_instance.setup.call_args_list[0][0][0]
    assert "ssr-nextjs-node" in setup_call_1["technologies"]
    # Verify second call setup subject
    setup_call_2 = mock_env_instance.setup.call_args_list[1][0][0]
    assert "csr-vanilla-nginx" in setup_call_2["technologies"]

    # 4. Assert TestRunner run_all was called for warmup and experiment (2 subjects * 2 runs)
    assert mock_runner_class.return_value.run_all.call_count == 4

    # 5. Assert final teardown was called once in finally block
    assert mock_env_instance.teardown.call_count == 1


def test_run_campaign_subject_filtering(
    mocker: MockerFixture, mock_campaign_files: tuple[Path, Path]
) -> None:
    """Verifies that run_campaign filters and runs only requested subjects."""
    campaign_yaml, infra_yaml = mock_campaign_files

    mock_env_class = mocker.patch("orchestrator.actions.campaign.provider.CloudEnvironment")
    mock_env_instance = mock_env_class.return_value
    mocker.patch("orchestrator.actions.campaign.provider.TestRunner")

    run_campaign(
        path=campaign_yaml,
        infra_path=infra_yaml,
        subject_filter="nginx",
        verbose=True,
    )

    # Should only run csr-vanilla-nginx
    assert mock_env_instance.setup.call_count == 1
    setup_call = mock_env_instance.setup.call_args[0][0]
    assert "csr-vanilla-nginx" in setup_call["technologies"]
    assert "ssr-nextjs-node" not in setup_call["technologies"]


def test_run_campaign_resume_skips_completed(
    mocker: MockerFixture, tmp_path: Path, mock_campaign_files: tuple[Path, Path]
) -> None:
    """Verifies that run_campaign resumes from campaign_state.json and skips completed subjects."""
    campaign_yaml, infra_yaml = mock_campaign_files

    # Setup pre-existing campaign directory
    campaign_dir = tmp_path / "campaign_active"
    campaign_dir.mkdir()
    state_file = campaign_dir / "campaign_state.json"

    # Pre-complete ssr-nextjs-node
    state = {
        "completed_subjects": ["ssr-nextjs-node"],
        "failed_subjects": [],
    }
    with open(state_file, "w") as f:
        json.dump(state, f)

    mock_env_class = mocker.patch("orchestrator.actions.campaign.provider.CloudEnvironment")
    mock_env_instance = mock_env_class.return_value
    mocker.patch("orchestrator.actions.campaign.provider.TestRunner")

    run_campaign(
        path=campaign_yaml,
        infra_path=infra_yaml,
        resume=campaign_dir,
        verbose=True,
    )

    # Should only setup csr-vanilla-nginx (ssr-nextjs-node skipped)
    assert mock_env_instance.setup.call_count == 1
    setup_call = mock_env_instance.setup.call_args[0][0]
    assert "csr-vanilla-nginx" in setup_call["technologies"]
    assert "ssr-nextjs-node" not in setup_call["technologies"]
