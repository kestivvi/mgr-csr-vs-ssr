from abc import ABC, abstractmethod
from typing import Any, Optional

from orchestrator.config import ANSIBLE_DIR, TERRAFORM_DIR
from orchestrator.shared.infra.ansible import AnsibleAdapter
from orchestrator.shared.infra.docker import DockerAdapter
from orchestrator.shared.infra.terraform import TerraformAdapter


class Environment(ABC):
    """
    Higher-level seam representing a research environment.
    Hides the coordination between provisioning and configuration.
    """

    @abstractmethod
    def setup(self, config: dict[str, Any], verbose: bool = False) -> None:
        pass

    @abstractmethod
    def teardown(self, verbose: bool = False) -> None:
        pass


class CloudEnvironment(Environment):
    """Coordinates Terraform and Ansible for AWS runs."""

    def __init__(self):
        self.tf = TerraformAdapter(TERRAFORM_DIR)
        self.ansible = AnsibleAdapter(ANSIBLE_DIR)

    def setup(self, config: dict[str, Any], verbose: bool = False) -> None:
        # 1. Provision Hardware
        self.tf.init(verbose=verbose)
        self.tf.apply(config, verbose=verbose)

        # 2. Configure Hardware
        # site.yml is located in the 'project' subdirectory
        # inventory is located in 'inventory/inventory.yml'
        self.ansible.run_playbook(
            playbook="project/site.yml",
            inventory="inventory/inventory.yml",
            verbose=verbose,
        )

    def teardown(self, verbose: bool = False) -> None:
        self.tf.destroy(verbose=verbose)


class LocalEnvironment(Environment):
    """Coordinates Docker for local verification runs."""

    def __init__(self, app_path: Any):
        # Local env is app-specific
        self.docker = DockerAdapter(app_path)

    def setup(self, config: dict[str, Any], verbose: bool = False) -> None:
        self.docker.build(verbose=verbose)
        self.docker.up(verbose=verbose)

    def teardown(self, verbose: bool = False) -> None:
        self.docker.down(verbose=verbose)
