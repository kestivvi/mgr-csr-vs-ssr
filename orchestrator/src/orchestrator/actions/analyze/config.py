from typing import Dict, Union

METRIC_CONFIG: Dict[str, Dict[str, Dict[str, Union[str, bool]]]] = {
    "mean": {
        "cpu": {"name": "Mean CPU Usage (%)", "sort_ascending": True},
        "memory": {"name": "Mean Memory Usage (MB)", "sort_ascending": True},
        "latency": {"name": "Mean p95 Latency (ms)", "sort_ascending": True},
        "network_tx": {"name": "Mean Network Transmit Rate (MB/s)", "sort_ascending": True},
    },
    "std": {
        "cpu": {"name": "CPU Usage Stability (Std Dev)", "sort_ascending": True},
        "memory": {"name": "Memory Usage Stability (Std Dev)", "sort_ascending": True},
        "latency": {"name": "Latency Stability (Std Dev)", "sort_ascending": True},
        "network_tx": {"name": "Network Transmit Stability (Std Dev)", "sort_ascending": True},
    },
    "p95": {
        "cpu": {"name": "Peak CPU Usage (95th Percentile)", "sort_ascending": True},
        "memory": {"name": "Peak Memory Usage (95th Percentile)", "sort_ascending": True},
        "latency": {"name": "Peak Latency (95th Percentile)", "sort_ascending": True},
        "network_tx": {
            "name": "Peak Network Transmit Rate (95th Percentile) (MB/s)",
            "sort_ascending": True,
        },
    },
}

PLOT_PALETTE = {"CSR": "#1f77b4", "SSR": "#d62728", "Uncategorized": "gray"}
