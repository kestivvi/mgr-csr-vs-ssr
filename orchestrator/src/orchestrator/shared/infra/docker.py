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

    app_dir_name: str
    app_id: str
    infra_dir: Path
    app_type: str
    compose_file: Path

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        self.app_dir_name = workdir.name
        self.app_id = self.app_dir_name.replace("-", "_")
        self.infra_dir = workdir.parent / "_infra"

        # Detect application type (SSR vs Nginx vs Apache)
        if self.app_dir_name.startswith("ssr-"):
            self.app_type = "ssr"
            self.compose_file = COMPOSE_SSR_NGINX
        elif "apache" in self.app_dir_name:
            self.app_type = "apache"
            self.compose_file = COMPOSE_CSR_APACHE
        else:
            self.app_type = "nginx"
            self.compose_file = COMPOSE_CSR_NGINX

    def _get_env(self) -> dict[str, str]:
        """Prepares environment variables for the master compose file."""
        static_path = "/_next/static/"
        if "nuxt" in self.app_dir_name:
            static_path = "/_nuxt/"
        elif "solid" in self.app_dir_name or "svelte" in self.app_dir_name:
            static_path = "/_build/"
        elif "astro" in self.app_dir_name:
            static_path = "/_astro/"

        # Set build target based on application type
        build_target = "nginx"
        if self.app_type == "ssr":
            build_target = "runner"
        elif self.app_type == "apache":
            build_target = "apache"

        return {
            "APPLICATION_DIR": self.app_dir_name,
            "APPLICATION_ID": self.app_id,
            "APPLICATION_PORT": "3000",
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
