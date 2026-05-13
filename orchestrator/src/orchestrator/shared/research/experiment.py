import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .schema import Column, MetricName


class Experiment:
    """
    Represents a full research experiment dataset.
    Provides deep, research-ready DataFrames.
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        metrics: pd.DataFrame,
        wrk_results: Optional[pd.DataFrame] = None,
    ):
        self.metadata = metadata
        self.metrics = metrics
        self.wrk_results = wrk_results

    @property
    def test_type(self) -> str:
        return self.metadata.get("test_type", "unknown")


class ExperimentLoader:
    """
    Deep loader that transforms raw results into an Experiment object.
    """

    def __init__(self, groups_config: Optional[Dict[str, List[str]]] = None):
        self.groups_config = groups_config or {}
        self._tech_to_group = {
            tech.lower(): group for group, techs in self.groups_config.items() for tech in techs
        }

    def load(self, path: Path) -> Experiment:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result directory not found: {path}")

        # 1. Load Metadata
        metadata = self._load_metadata(path)

        # 2. Load Metrics (Resource utilization)
        metrics_df = self._load_metrics(path)

        # 3. Load Tool Results (Dirty wrk tests)
        wrk_df = self._load_wrk_results(path)

        return Experiment(metadata=metadata, metrics=metrics_df, wrk_results=wrk_df)

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        meta_path = path / "metadata.yaml"
        if not meta_path.exists():
            return {}
        with open(meta_path, "r") as f:
            return yaml.safe_load(f)

    def _load_metrics(self, path: Path) -> pd.DataFrame:
        metrics_dir = path / "metrics"
        if not metrics_dir.is_dir():
            return pd.DataFrame()

        all_files = glob.glob(str(metrics_dir / "*.csv"))
        if not all_files:
            return pd.DataFrame()

        all_long_dfs = []
        filename_regex = re.compile(r"^(\d+)_(.*)$")

        for f in all_files:
            p = Path(f)
            match = filename_regex.match(p.stem)
            if not match:
                continue
            run_num, tech_raw = match.groups()

            wide_df = pd.read_csv(f)
            long_df = pd.melt(
                wide_df,
                id_vars=[Column.TIMESTAMP],
                var_name=Column.METRIC,
                value_name=Column.VALUE,
            )
            long_df[Column.RUN_NUMBER] = int(run_num)
            tech = tech_raw.replace("_", "-")
            long_df[Column.SERVER_TYPE] = tech

            # Apply Taxonomy
            long_df[Column.GROUP] = (
                long_df[Column.SERVER_TYPE]
                .str.lower()
                .map(self._tech_to_group)
                .fillna("Uncategorized")
            )

            # Normalization (Research Standards)
            self._normalize_units(long_df)

            all_long_dfs.append(long_df)

        if not all_long_dfs:
            return pd.DataFrame()

        df = pd.concat(all_long_dfs, ignore_index=True)

        # Time Calculation
        df[Column.TIMESTAMP] = pd.to_datetime(df[Column.TIMESTAMP])
        df[Column.TIME_SEC] = df.groupby([Column.SERVER_TYPE, Column.RUN_NUMBER, Column.METRIC])[
            Column.TIMESTAMP
        ].transform(lambda x: (x - x.min()).dt.total_seconds())

        return df

    def _normalize_units(self, df: pd.DataFrame) -> None:
        """Applies academic normalization to metrics in-place."""
        # CPU: ratio -> percentage
        df.loc[df[Column.METRIC] == MetricName.CPU, Column.VALUE] *= 100
        # RAM: bytes -> MB
        df.loc[df[Column.METRIC] == MetricName.MEMORY, Column.VALUE] /= 1024 * 1024
        # Network: bytes -> MB
        df.loc[df[Column.METRIC] == MetricName.NETWORK_TX, Column.VALUE] /= 1024 * 1024
        df.loc[df[Column.METRIC] == MetricName.NETWORK_RX, Column.VALUE] /= 1024 * 1024

    def _load_wrk_results(self, path: Path) -> Optional[pd.DataFrame]:
        wrk_dir = path / "tool_results"
        all_files = glob.glob(str(wrk_dir / "*_wrk.json"))
        if not all_files:
            return None

        records = []
        for f in all_files:
            p = Path(f)
            parts = p.stem.split("_")
            run_num = int(parts[0])
            tech = "-".join(parts[1:-1])

            with open(f, "r") as jf:
                res = json.load(jf)

                # Parsing latency strings (e.g. "1.2ms", "500us")
                lat_str = str(res.get("latency_avg", "0ms"))
                lat_match = re.search(r"[\d.]+", lat_str)
                lat_val = float(lat_match.group()) if lat_match else 0.0
                if "us" in lat_str:
                    lat_val /= 1000
                elif "s" in lat_str and "ms" not in lat_str:
                    lat_val *= 1000

                records.append(
                    {
                        Column.RUN_NUMBER: run_num,
                        Column.SERVER_TYPE: tech,
                        Column.GROUP: self._tech_to_group.get(tech.lower(), "Uncategorized"),
                        "rps": float(res.get("rps", 0.0)),
                        "latency_ms": lat_val,
                    }
                )

        return pd.DataFrame(records) if records else None
