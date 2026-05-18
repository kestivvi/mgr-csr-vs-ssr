import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Application:
    id: str  # Canonical ID (folder name, e.g. csr-angular-nginx)
    strategy: str
    family: str
    meta_framework: Optional[str]
    runtime: str
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> "Application":
        manifest_path = path / "application.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing application.json in {path}")

        with open(manifest_path, "r") as f:
            data = json.load(f)

        app = cls(
            id=path.name,
            strategy=data["strategy"],
            family=data["family"],
            meta_framework=data.get("meta_framework"),
            runtime=data["runtime"],
            path=path,
        )
        app.validate()
        return app

    def validate(self) -> None:
        """Enforces the official Naming Contract for Research Applications."""
        if self.strategy not in ["csr", "ssr"]:
            raise ValueError(f"Strategy must start with 'csr-' or 'ssr-', got '{self.strategy}'")
        if not (self.id.startswith("csr-") or self.id.startswith("ssr-")):
            raise ValueError(f"Strategy must start with 'csr-' or 'ssr-', got ID '{self.id}'")
        if not self.id.startswith(f"{self.strategy}-"):
            raise ValueError(
                "Folder name does not match Application ID parts: "
                f"strategy mismatch ('{self.strategy}' vs '{self.id}')"
            )

        valid_runtimes = ["nginx", "apache", "node", "bun", "deno"]
        if self.runtime not in valid_runtimes:
            raise ValueError(
                f"Runtime suffix must end with a valid server runtime, got '{self.runtime}'"
            )
        if not self.id.endswith(f"-{self.runtime}"):
            raise ValueError(
                "Folder name does not match Application ID parts: "
                f"runtime mismatch ('{self.runtime}' vs '{self.id}')"
            )

        # Check that metaframework if present is in the ID
        if self.meta_framework and self.meta_framework not in self.id:
            raise ValueError(
                "Folder name does not match Application ID parts: "
                f"missing metaframework '{self.meta_framework}'"
            )
        # Check that family is in ID (except for non-agnostic frameworks)
        if self.family and self.family not in self.id:
            if self.meta_framework in ["astro", "tanstack-start"] or not self.meta_framework:
                raise ValueError(
                    f"Folder name does not match Application ID parts: "
                    f"missing family '{self.family}'"
                )

    @property
    def slug(self) -> str:
        """Slug version of the ID (used for Prometheus/Metric storage)."""
        return self.id.replace("-", "_")

    @property
    def display_name(self) -> str:
        """Human readable name for plots."""
        parts = [self.strategy.upper()]
        if self.meta_framework:
            # Capitalize metaframework nicely
            mf = self.meta_framework.lower()
            if mf == "nextjs":
                mf_name = "Next.js"
            elif mf == "nuxtjs":
                mf_name = "Nuxt"
            elif mf == "svelte-kit":
                mf_name = "SvelteKit"
            elif mf == "tanstack-start":
                mf_name = "TanStack Start"
            else:
                mf_name = self.meta_framework.capitalize()
            parts.append(mf_name)

            # Framework-agnostic metaframeworks require appending the framework family
            if mf in ["astro", "tanstack-start"]:
                parts.append(self.family.capitalize())
        else:
            parts.append(self.family.capitalize())
        parts.append(f"({self.runtime.capitalize()})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "family": self.family,
            "meta_framework": self.meta_framework,
            "runtime": self.runtime,
            "path": str(self.path),
            "slug": self.slug,
            "display_name": self.display_name,
        }


class ApplicationRegistry:
    def __init__(self, applications_dir: Path):
        self.applications_dir = applications_dir
        self._applications: Dict[str, Application] = {}
        self.reload()

    def reload(self) -> None:
        self._applications = {}
        if not self.applications_dir.exists():
            return

        for item in self.applications_dir.iterdir():
            if item.is_dir() and (item / "application.json").exists():
                try:
                    app = Application.from_path(item)
                    self._applications[app.id] = app
                except Exception as e:
                    # Log but continue
                    print(f"Warning: Failed to load application at {item}: {e}")
                    continue

    def get(self, application_id: str) -> Application:
        if application_id not in self._applications:
            raise KeyError(f"Application '{application_id}' not found in registry.")
        return self._applications[application_id]

    def all(self) -> List[Application]:
        return sorted(list(self._applications.values()), key=lambda x: x.id)

    def get_default_groups(self) -> Dict[str, List[str]]:
        """
        Grouping logic for the Analyzer.
        Clusters applications by strategy and runtime environment.
        """
        groups: Dict[str, List[str]] = {}
        for s in self.all():
            # Group by Strategy + Runtime (e.g. "SSR-Node", "CSR-Nginx")
            group_key = f"{s.strategy.upper()}-{s.runtime.capitalize()}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(s.id)  # Use ID here for grouping logic
        return groups
