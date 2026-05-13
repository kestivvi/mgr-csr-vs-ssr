import json
import os
from pathlib import Path
from typing import Any, Optional

from orchestrator.shared.infra.base import BaseAdapter
from orchestrator.shared.infra.exceptions import TerraformError


class TerraformAdapter(BaseAdapter):
    """
    Deep adapter for Terraform.
    Hides CLI complexities like variable passing and JSON output parsing.
    """

    def init(self, verbose: bool = False) -> None:
        """Runs terraform init."""
        self._run(["terraform", "init"], verbose=verbose, error_type=TerraformError)

    def apply(
        self,
        variables: dict[str, Any],
        log_path: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        """
        Runs terraform apply with auto-approval and variable mapping.

        Args:
            variables: Dictionary of Terraform variables to pass as -var flags.
        """
        cmd = ["terraform", "apply", "-auto-approve"]

        for key, value in variables.items():
            if isinstance(value, (dict, list)):
                # Complex types must be passed as JSON strings for Terraform
                tf_value = json.dumps(value)
            else:
                tf_value = str(value)
            cmd.extend(["-var", f"{key}={tf_value}"])

        env = os.environ.copy()
        env["TF_IN_AUTOMATION"] = "true"

        self._run(
            cmd,
            env=env,
            log_path=log_path,
            verbose=verbose,
            error_type=TerraformError,
        )

    def destroy(self, log_path: Optional[Path] = None, verbose: bool = False) -> None:
        """Runs terraform destroy with auto-approval."""
        cmd = ["terraform", "destroy", "-auto-approve"]
        self._run(
            cmd,
            log_path=log_path,
            verbose=verbose,
            error_type=TerraformError,
        )

    def get_outputs(self) -> dict[str, Any]:
        """Fetches terraform outputs as a dictionary."""
        output_json = self._run(
            ["terraform", "output", "-json"],
            error_type=TerraformError,
        )
        try:
            raw_outputs = json.loads(output_json)
            # Terraform -json output is { "key": { "value": ... } }
            return {k: v["value"] for k, v in raw_outputs.items()}
        except (json.JSONDecodeError, KeyError) as e:
            raise TerraformError(f"Failed to parse terraform output: {e}", logs=output_json) from e
