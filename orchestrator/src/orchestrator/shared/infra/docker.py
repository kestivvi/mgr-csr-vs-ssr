from pathlib import Path
from typing import Optional

from orchestrator.shared.infra.base import BaseAdapter
from orchestrator.shared.infra.exceptions import DockerError


class DockerAdapter(BaseAdapter):
    """
    Deep adapter for Docker-Compose.
    Provides a consistent 'Provisioner' interface for local verification.
    """

    def build(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Runs docker-compose build."""
        self._run(
            ["docker-compose", "build"],
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def up(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Starts containers in detached mode."""
        self._run(
            ["docker-compose", "up", "-d", "--force-recreate"],
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def down(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Stops and removes containers."""
        self._run(
            ["docker-compose", "down"],
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def logs(self, log_path: Optional[Path] = None, verbose: bool = False) -> str:
        """Fetches container logs."""
        return self._run(
            ["docker-compose", "logs"],
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )
