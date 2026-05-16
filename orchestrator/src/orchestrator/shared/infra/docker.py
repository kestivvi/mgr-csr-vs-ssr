from pathlib import Path

from orchestrator.config import (
    COMPOSE_CSR_APACHE,
    COMPOSE_CSR_NGINX,
    COMPOSE_SSR_NGINX,
)
from orchestrator.shared.infra.base import BaseAdapter
from orchestrator.shared.infra.exceptions import DockerError


class DockerAdapter(BaseAdapter):
    """
    Deep adapter for Docker-Compose using Universal Master Files.
    """

    subject_dir_name: str
    subject_id: str
    infra_dir: Path
    subject_type: str
    compose_file: Path

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        self.subject_dir_name = workdir.name
        self.subject_id = self.subject_dir_name.replace("-", "_")
        self.infra_dir = workdir.parent / "_infra"

        # Detect subject type (SSR vs Nginx vs Apache)
        if self.subject_dir_name.startswith("ssr-"):
            self.subject_type = "ssr"
            self.compose_file = COMPOSE_SSR_NGINX
        elif "apache" in self.subject_dir_name:
            self.subject_type = "apache"
            self.compose_file = COMPOSE_CSR_APACHE
        else:
            self.subject_type = "nginx"
            self.compose_file = COMPOSE_CSR_NGINX

    def _get_env(self) -> dict[str, str]:
        """Prepares environment variables for the master compose file."""
        # Default ports/paths - these could later be moved to a subject-specific config.yaml
        static_path = "/_next/static/"
        if "nuxt" in self.subject_dir_name:
            static_path = "/_nuxt/"
        elif "solid" in self.subject_dir_name or "svelte" in self.subject_dir_name:
            static_path = "/_build/"
        elif "astro" in self.subject_dir_name:
            static_path = "/_astro/"

        # Set build target based on subject type
        build_target = "nginx"
        if self.subject_type == "ssr":
            build_target = "runner"
        elif self.subject_type == "apache":
            build_target = "apache"

        return {
            "SUBJECT_DIR": self.subject_dir_name,
            "SUBJECT_ID": self.subject_id,
            "SUBJECT_PORT": "3000",
            "HOST_PORT": "80",  # Default host port
            "STATIC_PATH": static_path,
            "BUILD_TARGET": build_target,
        }

    def build(self, log_path: Path | None = None, verbose: bool = False) -> None:
        """Runs docker-compose build using the master file."""
        self._run(
            ["docker-compose", "-f", str(self.compose_file), "build"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def up(self, log_path: Path | None = None, verbose: bool = False) -> None:
        """Starts containers using the master file."""
        self._run(
            ["docker-compose", "-f", str(self.compose_file), "up", "-d", "--force-recreate"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def down(self, log_path: Path | None = None, verbose: bool = False) -> None:
        """Stops and removes containers using the master file."""
        self._run(
            ["docker-compose", "-f", str(self.compose_file), "down"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )

    def logs(self, log_path: Path | None = None, verbose: bool = False) -> str:
        """Fetches container logs using the master file."""
        return self._run(
            ["docker-compose", "-f", str(self.compose_file), "logs"],
            env=self._get_env(),
            log_path=log_path,
            verbose=verbose,
            error_type=DockerError,
        )
