class InfrastructureError(Exception):
    """Base exception for all infrastructure-related failures."""

    command: list[str] | None
    return_code: int | None
    logs: str | None

    def __init__(
        self,
        message: str,
        command: list[str] | None = None,
        return_code: int | None = None,
        logs: str | None = None,
    ):
        super().__init__(message)
        self.command = command
        self.return_code = return_code
        self.logs = logs

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.return_code is not None:
            return f"{base_msg} (Exit Code: {self.return_code})"
        return base_msg


class TerraformError(InfrastructureError):
    """Raised when a Terraform operation fails."""

    pass


class AnsibleError(InfrastructureError):
    """Raised when an Ansible operation fails."""

    pass


class DockerError(InfrastructureError):
    """Raised when a Docker/Docker-Compose operation fails."""

    pass
