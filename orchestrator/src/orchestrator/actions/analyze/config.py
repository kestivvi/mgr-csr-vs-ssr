from typing import Dict, Union

METRIC_CONFIG: Dict[str, Dict[str, Dict[str, Union[str, bool]]]] = {
    "mean": {
        "cpu": {"name": "Średnie zużycie CPU (%)", "sort_ascending": True},
        "memory": {"name": "Średnie zużycie pamięci (MB)", "sort_ascending": True},
        "p99": {"name": "Średnie opóźnienie p99 (ms)", "sort_ascending": True},
        "network_tx": {"name": "Średnia przepustowość wyjściowa (MB/s)", "sort_ascending": True},
    },
    "std": {
        "cpu": {
            "name": "Stabilność zużycia CPU (odchylenie stand.)",
            "sort_ascending": True,
        },
        "memory": {
            "name": "Stabilność zużycia pamięci (odchylenie stand.)",
            "sort_ascending": True,
        },
        "p99": {
            "name": "Stabilność opóźnienia p99 (odchylenie stand.)",
            "sort_ascending": True,
        },
        "network_tx": {
            "name": "Stabilność przepustowości wyjściowej (odchylenie stand.)",
            "sort_ascending": True,
        },
    },
    "p99": {
        "cpu": {"name": "Szczytowe zużycie CPU (99. percentyl)", "sort_ascending": True},
        "memory": {"name": "Szczytowe zużycie pamięci (99. percentyl)", "sort_ascending": True},
        "p99": {"name": "Szczytowe opóźnienie p99 (99. percentyl)", "sort_ascending": True},
        "network_tx": {
            "name": "Szczytowa przepustowość wyjściowa (99. percentyl) (MB/s)",
            "sort_ascending": True,
        },
    },
}

PLOT_PALETTE = {"CSR": "#1f77b4", "SSR": "#d62728", "Uncategorized": "gray"}
