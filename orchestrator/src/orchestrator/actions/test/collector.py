from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from typing import TypedDict

import pandas as pd
from prometheus_api_client import PrometheusConnect
from rich.console import Console

console = Console()


class MetricDefinition(TypedDict):
    name: str
    query: str


METRIC_CONFIG: dict[str, MetricDefinition] = {
    "cpu": {
        "name": "CPU Utilization",
        "query": (
            "avg by (server) (1 - rate(node_cpu_seconds_total"
            '{{mode="idle", server="{server_label}"}}[15s]))'
        ),
    },
    "memory": {
        "name": "Memory Usage",
        "query": (
            "sum by (server) (node_memory_MemTotal_bytes"
            '{{server="{server_label}"}} - '
            'node_memory_MemAvailable_bytes{{server="{server_label}"}})'
        ),
    },
    "p99": {
        "name": "99. percentyl czasu odpowiedzi",
        "query": ('avg by (server) (k6_http_req_duration_p99{{server="{server_label}"}}) * 1000'),
    },
    "avg_latency": {
        "name": "Average Request Latency",
        "query": ('avg by (server) (k6_http_req_duration_avg{{server="{server_label}"}}) * 1000'),
    },
    "network_tx": {
        "name": "Network Transmit Rate",
        "query": (
            "sum by (server) (rate(node_network_transmit_bytes_total"
            '{{server="{server_label}"}}[15s]))'
        ),
    },
    "k6_successful_html_reqs_rate": {
        "name": "Successful HTML Requests Rate",
        "query": (
            "sum by (server) (rate(k6_http_reqs_total"
            '{{server="{server_label}", resource_type="html", status="200", error_code=""}}[15s]))'
        ),
    },
    "k6_total_html_reqs_rate": {
        "name": "Total HTML Requests Rate",
        "query": (
            "sum by (server) (rate(k6_http_reqs_total"
            '{{server="{server_label}", resource_type="html"}}[15s]))'
        ),
    },
}


def download_metric(
    prom: PrometheusConnect,
    metric_name: str,
    server_type: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame | None:
    """Downloads a single metric from Prometheus."""
    config = METRIC_CONFIG[metric_name]
    server_label = server_type.upper()
    promql = config["query"].format(server_label=server_label)

    try:
        metric_data = prom.custom_query_range(
            query=promql, start_time=start_time, end_time=end_time, step="1s"
        )

        if not metric_data:
            return None

        all_dfs = []
        for series in metric_data:
            if not series.get("values"):
                continue
            df = pd.DataFrame(series["values"])
            df.columns = ["timestamp", "metric_value"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
            all_dfs.append(df)

        if not all_dfs:
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True).dropna()
        combined_df = combined_df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="first"
        )

        if combined_df.empty:
            return None

        combined_df.set_index("timestamp", inplace=True)
        combined_df.rename(columns={"metric_value": metric_name}, inplace=True)
        return combined_df
    except Exception as e:
        console.print(f"[bold red]Error downloading {metric_name}:[/bold red] {e}")
        return None


def collect_metrics(
    prometheus_url: str,
    start_epoch: int,
    end_epoch: int,
    server_type: str,
    repetition_number: int,
    output_dir: Path,
) -> None:
    """Main entry point for metric collection."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    start_time = datetime.fromtimestamp(int(start_epoch), tz=timezone.utc)
    end_time = datetime.fromtimestamp(int(end_epoch), tz=timezone.utc)

    all_metric_dfs = []
    for metric_name in METRIC_CONFIG.keys():
        df = download_metric(prom, metric_name, server_type, start_time, end_time)
        if df is not None and not df.empty:
            all_metric_dfs.append(df)

    if not all_metric_dfs:
        console.print(f"[bold red]Failed to download any metrics for {server_type}[/bold red]")
        return

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), all_metric_dfs
    )

    ideal_index = pd.date_range(start=start_time, end=end_time, freq="s")
    merged_df = merged_df.reindex(ideal_index)

    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)
    merged_df.fillna(0, inplace=True)

    sanitized_server_type = server_type.lower().replace("-", "_")
    filename = f"{repetition_number:02d}_{sanitized_server_type}.csv"
    output_path = metrics_dir / filename
    merged_df.to_csv(output_path, index=True, index_label="timestamp")
    console.print(f"[green]Saved metrics to {output_path}[/green]")
