from pathlib import Path
from typing import Optional

from orchestrator.shared.infra.base import BaseAdapter
from orchestrator.shared.infra.exceptions import DockerError


class DockerAdapter(BaseAdapter):
    """
    Deep adapter for Docker-Compose using Universal Master Files.
    """

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        self.app_dir = workdir.name
        self.app_id = self.app_dir.replace("-", "_")
        self.infra_dir = workdir.parent / "_infra"

        # Detect app type (SSR vs CSR vs Apache)
        if self.app_dir.startswith("ssr-"):
            self.app_type = "ssr"
            self.compose_file = self.infra_dir / "compose" / "ssr.yml"
        elif "apache" in self.app_dir:
            self.app_type = "apache"
            self.compose_file = self.infra_dir / "compose" / "apache.yml"
        else:
            self.app_type = "csr"
            self.compose_file = self.infra_dir / "compose" / "csr.yml"

    def _get_env(self) -> dict[str, str]:
        """Prepares environment variables for the master compose file."""
        # Default ports/paths - these could later be moved to an app-specific config.yaml
        static_path = "/_next/static/"
        if "nuxt" in self.app_dir:
            static_path = "/_nuxt/"
        elif "solid" in self.app_dir or "svelte" in self.app_dir:
            static_path = "/_build/"
        elif "astro" in self.app_dir:
            static_path = "/_astro/"

        return {
            "APP_DIR": self.app_dir,
            "APP_ID": self.app_id,
            "APP_PORT": "3000",
            "HOST_PORT": "80",  # Default host port
            "STATIC_PATH": static_path,
            "BUILD_TARGET": "runner"
            if self.app_type == "ssr"
            else ("nginx" if self.app_type == "csr" else ""),
        }

    def build(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Runs docker-compose build using the master file."""
        self._run(
            ["docker-compose", "-f", str(self.compose_file), "build"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def up(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Starts containers using the master file."""
        self._run(
            ["docker-compose", "-f", str(self.compose_file), "up", "-d", "--force-recreate"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def down(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Stops and removes containers using the master file."""
        self._run(
            ["docker-compose", "-f", str(self.compose_file), "down"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def logs(self, log_path: Optional[Path] = None, verbose: bool = False) -> str:
        """Fetches container logs using the master file."""
        return self._run(
            ["docker-compose", "-f", str(self.compose_file), "logs"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )
