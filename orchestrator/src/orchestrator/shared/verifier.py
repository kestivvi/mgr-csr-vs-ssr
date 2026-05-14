import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from orchestrator.shared.infra.base import BaseAdapter
from orchestrator.shared.infra.exceptions import InfrastructureError


@dataclass
class HealthCheck:
    """Defines a single HTTP check to perform against an endpoint."""

    path: str
    check_content: bool = False
    verify_gzip: bool = False
    headers_only: bool = False


@dataclass
class HealthProfile:
    """A collection of health checks defining 'healthy' for a category of apps."""

    name: str
    checks: list[HealthCheck]


# Standard Profiles for the MGR Research Project
SSR_PROFILE = HealthProfile(
    name="SSR",
    checks=[
        HealthCheck("/", check_content=True, verify_gzip=True),
        HealthCheck("/favicon.ico", headers_only=True),
        HealthCheck("/dynamic/verify", check_content=True, verify_gzip=True),
    ],
)

CSR_PROFILE = HealthProfile(
    name="CSR",
    checks=[
        HealthCheck("/", verify_gzip=True),
        HealthCheck("/favicon.ico", headers_only=True),
        HealthCheck("/dynamic/verify", verify_gzip=True),
    ],
)


class AppVerifier(BaseAdapter):
    """
    A deep module for verifying application health.
    Leverages BaseAdapter for unified process execution and logging.
    """

    on_output: Callable[[str], None] | None

    def __init__(self, workdir: Path, on_output: Callable[[str], None] | None = None):
        super().__init__(workdir)
        self.on_output = on_output

    def _log(self, message: str) -> None:
        if self.on_output:
            self.on_output(message)

    def wait_until_healthy(
        self,
        base_url: str,
        profile: HealthProfile,
        retries: int = 4,
        delay: int = 2,
        log_path: Path | None = None,
    ) -> bool:
        """
        Polls the application until all health checks in the profile pass.

        Args:
            base_url: The root URL of the app (e.g. http://localhost:80).
            profile: The HealthProfile (SSR/CSR) to validate against.
            retries: Maximum number of attempts.
            delay: Seconds to wait between attempts.
            log_path: Optional file to append full curl outputs to.

        Returns:
            True if all checks passed within the retry limit, False otherwise.
        """
        self._log(f"Verifying {profile.name} health at {base_url}...")

        for i in range(retries):
            attempt = i + 1
            if log_path:
                with open(log_path, "a") as f:
                    f.write(f"\n\n--- Attempt {attempt} for {base_url} ({profile.name}) ---\n")

            success = True
            for check in profile.checks:
                url = f"{base_url.rstrip('/')}/{check.path.lstrip('/')}"
                if not self._run_check(url, check, log_path):
                    success = False
                    break

            if success:
                self._log(f"Success! {profile.name} health checks passed after {attempt} attempts.")
                return True

            if attempt < retries:
                self._log(f"Attempt {attempt} failed, retrying in {delay}s...")
                time.sleep(delay)

        self._log(f"Verification failed after {retries} attempts.")
        return False

    def _run_check(self, url: str, check: HealthCheck, log_path: Path | None) -> bool:
        """Executes one or more curl commands for a single HealthCheck."""
        # We always test standard mode
        if not self._curl(url, check, use_gzip=False, log_path=log_path):
            return False

        # Optionally test gzip compression
        if check.verify_gzip:
            if not self._curl(url, check, use_gzip=True, log_path=log_path):
                return False

        return True

    def _curl(self, url: str, check: HealthCheck, use_gzip: bool, log_path: Path | None) -> bool:
        """Internal helper to execute a single curl call and validate output."""
        flags = ["-IsLk"] if check.headers_only else ["-isLk"]
        if use_gzip:
            flags += ["--compressed", "-H", "Accept-Encoding: gzip"]

        command = ["curl"] + flags + [url]

        try:
            # We use _run from BaseAdapter to get unified logging and error handling
            # Note: verbose=False because we handle our own high-level logging via self._log
            output = self._run(command, log_path=log_path, verbose=False)
            output_lower = output.lower()

            # 1. Validate Status Code (200 OK)
            if "http/1.1 200" not in output_lower and "http/2 200" not in output_lower:
                return False

            # 2. Validate Content (if required)
            if check.check_content and "hello world" not in output_lower:
                return False

            # 3. Validate Gzip Header (if required)
            if use_gzip and "content-encoding: gzip" not in output_lower:
                return False

            return True

        except InfrastructureError:
            # curl failed to execute or returned non-zero
            return False
