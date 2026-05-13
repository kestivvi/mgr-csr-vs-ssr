import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text

from orchestrator.shared.infra.exceptions import InfrastructureError

console = Console()


class BaseAdapter:
    """
    Base class for deep infrastructure adapters.
    Handles process execution, real-time output streaming, and error wrapping.
    """

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def _run(
        self,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        log_path: Optional[Path] = None,
        verbose: bool = False,
        error_type: type[InfrastructureError] = InfrastructureError,
    ) -> str:
        """
        Executes a command and returns the full output.
        
        Args:
            command: List of command arguments.
            env: Optional environment variables.
            log_path: Optional file path to append logs to.
            verbose: If True, streams output to the console in real-time.
            error_type: The exception class to raise on failure.
        """
        full_output = []
        
        env = env or {}
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            log_file = None
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(log_path, "a")

            if process.stdout:
                # Use a line iterator for immediate feedback
                for line in iter(process.stdout.readline, ""):
                    full_output.append(line)

                    if verbose:
                        console.print(Text.from_ansi(line), end="")

                    if log_file:
                        log_file.write(line)
                        log_file.flush()

            if log_file:
                log_file.close()

            rc = process.wait()
            
            output_str = "".join(full_output)
            
            if rc != 0:
                raise error_type(
                    f"Command failed: {' '.join(command)}",
                    command=command,
                    return_code=rc,
                    logs=output_str,
                )
                
            return output_str

        except InfrastructureError:
            # Re-raise specialized errors
            raise
        except Exception as e:
            raise error_type(
                f"Unexpected error executing {' '.join(command)}: {e}",
                command=command,
            ) from e
