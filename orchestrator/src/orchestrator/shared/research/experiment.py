import json
import re
from typing import Any

import pandas as pd

from .artifact import ResearchArtifact
from .schema import Column, MetricName


class Experiment:
    """
    Represents a full research experiment dataset.
    Provides deep, research-ready DataFrames.
    """

    metadata: dict[str, Any]
    metrics: pd.DataFrame
    wrk_results: pd.DataFrame | None

    def __init__(
        self,
        metadata: dict[str, Any],
        metrics: pd.DataFrame,
        wrk_results: pd.DataFrame | None = None,
    ):
        self.metadata = metadata
        self.metrics = metrics
        self.wrk_results = wrk_results

    @property
    def test_type(self) -> str:
        return str(self.metadata.get("test_type", "unknown"))

    @property
    def application_metadata(self) -> dict[str, dict[str, Any]]:
        """Returns the structured metadata for all applications in this experiment."""
        from typing import cast

        return cast(dict[str, dict[str, Any]], self.metadata.get("applications", {}))


class ExperimentLoader:
    """
    Deep loader that transforms raw results into an Experiment object.
    """

    def __init__(self) -> None:
        pass

    def _get_group(self, server_type: str) -> str:
        """Determines the Rendering Strategy (Group) from the server type name prefix."""
        name = server_type.lower()
        if name.startswith("csr-"):
            return "CSR"
        if name.startswith("ssr-"):
            return "SSR"
        return "Uncategorized"

    def load(self, artifact: ResearchArtifact) -> Experiment:
        # 1. Use Metadata from artifact
        metadata = artifact.metadata

        # 2. Structured Metadata Guard (Fail-Fast)
        if "applications" not in metadata:
            raise ValueError("Missing structured application metadata in artifact.")

        # 3. Load Metrics (Resource utilization)
        metrics_df = self._load_metrics(artifact)

        # 3. Load Tool Results
        wrk_df = self._load_wrk_results(artifact)

        return Experiment(metadata=metadata, metrics=metrics_df, wrk_results=wrk_df)

    def _load_metrics(self, artifact: ResearchArtifact) -> pd.DataFrame:
        runs = artifact.get_repetitions()
        if not runs:
            return pd.DataFrame()

        all_long_dfs = []
        for run in runs:
            if not run.metrics_path:
                continue

            wide_df = pd.read_csv(run.metrics_path)
            long_df = pd.melt(
                wide_df,
                id_vars=[Column.TIMESTAMP],
                var_name=Column.METRIC,
                value_name=Column.VALUE,
            )
            long_df[Column.REPETITION_NUMBER] = run.repetition_id
            long_df[Column.SERVER_TYPE] = run.server_type

            # Apply Taxonomy
            long_df[Column.GROUP] = long_df[Column.SERVER_TYPE].apply(self._get_group)

            # Normalization (Research Standards)
            self._normalize_units(long_df)

            all_long_dfs.append(long_df)

        if not all_long_dfs:
            return pd.DataFrame()

        df = pd.concat(all_long_dfs, ignore_index=True)

        # Time Calculation
        df[Column.TIMESTAMP] = pd.to_datetime(df[Column.TIMESTAMP])
        df[Column.TIME_SEC] = df.groupby(
            [Column.SERVER_TYPE, Column.REPETITION_NUMBER, Column.METRIC]
        )[Column.TIMESTAMP].transform(lambda x: (x - x.min()).dt.total_seconds())

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

    def _load_wrk_results(self, artifact: ResearchArtifact) -> pd.DataFrame | None:
        runs = artifact.get_repetitions()
        if not runs:
            return None

        records = []
        for run in runs:
            if not run.results_path:
                continue

            with open(run.results_path, "r") as jf:
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
                        Column.REPETITION_NUMBER: run.repetition_id,
                        Column.SERVER_TYPE: run.server_type,
                        Column.GROUP: self._get_group(run.server_type),
                        "rps": float(res.get("rps", 0.0)),
                        "latency_ms": lat_val,
                    }
                )

        return pd.DataFrame(records) if records else None
