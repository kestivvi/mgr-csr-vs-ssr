import json
from pathlib import Path
from typing import Any

from orchestrator.shared.ansible import get_ansible_env
from orchestrator.shared.infra.base import BaseAdapter
from orchestrator.shared.infra.exceptions import AnsibleError


class AnsibleAdapter(BaseAdapter):
    """
    Deep adapter for Ansible.
    Replaces ansible_runner with raw CLI calls for consistent output streaming.
    """

    def run_playbook(
        self,
        playbook: str,
        inventory: str | None = None,
        extra_vars: dict[str, Any] | None = None,
        log_path: Path | None = None,
        verbose: bool = False,
    ) -> str:
        """
        Executes an ansible-playbook.

        Args:
            playbook: Path to the playbook file.
            inventory: Path to the inventory file.
            extra_vars: Dictionary of variables to pass as --extra-vars.
        """
        cmd = ["ansible-playbook", playbook]

        if inventory:
            cmd.extend(["-i", inventory])

        if extra_vars:
            # Pass as JSON to ensure types are preserved
            cmd.extend(["--extra-vars", json.dumps(extra_vars)])

        return self._run(
            cmd,
            env=get_ansible_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=AnsibleError,
        )
