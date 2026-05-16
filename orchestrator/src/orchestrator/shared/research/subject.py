import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Subject:
    id: str  # Canonical ID (folder name, e.g. csr-angular-nginx)
    strategy: str
    family: str
    meta_framework: Optional[str]
    runtime: str
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> "Subject":
        manifest_path = path / "subject.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing subject.json in {path}")

        with open(manifest_path, "r") as f:
            data = json.load(f)

        return cls(
            id=path.name,
            strategy=data["strategy"],
            family=data["family"],
            meta_framework=data.get("meta_framework"),
            runtime=data["runtime"],
            path=path,
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
            parts.append(self.meta_framework.capitalize())
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


class SubjectRegistry:
    def __init__(self, subjects_dir: Path):
        self.subjects_dir = subjects_dir
        self._subjects: Dict[str, Subject] = {}
        self.reload()

    def reload(self) -> None:
        self._subjects = {}
        if not self.subjects_dir.exists():
            return

        for item in self.subjects_dir.iterdir():
            if item.is_dir() and (item / "subject.json").exists():
                try:
                    subj = Subject.from_path(item)
                    self._subjects[subj.id] = subj
                except Exception as e:
                    # Log but continue
                    print(f"Warning: Failed to load subject at {item}: {e}")
                    continue

    def get(self, subject_id: str) -> Subject:
        if subject_id not in self._subjects:
            raise KeyError(f"Subject '{subject_id}' not found in registry.")
        return self._subjects[subject_id]

    def all(self) -> List[Subject]:
        return sorted(list(self._subjects.values()), key=lambda x: x.id)

    def get_default_groups(self) -> Dict[str, List[str]]:
        """
        Grouping logic for the Analyzer.
        Clusters subjects by strategy and runtime environment.
        """
        groups: Dict[str, List[str]] = {}
        for s in self.all():
            # Group by Strategy + Runtime (e.g. "SSR-Node", "CSR-Nginx")
            group_key = f"{s.strategy.upper()}-{s.runtime.capitalize()}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(s.id)  # Use ID here for grouping logic
        return groups
