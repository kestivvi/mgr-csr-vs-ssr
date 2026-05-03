from typing import Dict, Union

METRIC_CONFIG: Dict[str, Dict[str, Dict[str, Union[str, bool]]]] = {
    "mean": {
        "cpu": {"name": "Średnie zużycie CPU (%)", "sort_ascending": True},
        "memory": {"name": "Średnie zużycie pamięci (MB)", "sort_ascending": True},
        "latency": {"name": "Średnia latencja p95 (ms)", "sort_ascending": True},
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
        "latency": {
            "name": "Stabilność latencji (odchylenie stand.)",
            "sort_ascending": True,
        },
        "network_tx": {
            "name": "Stabilność przepustowości wyjściowej (odchylenie stand.)",
            "sort_ascending": True,
        },
    },
    "p95": {
        "cpu": {"name": "Szczytowe zużycie CPU (95. percentyl)", "sort_ascending": True},
        "memory": {"name": "Szczytowe zużycie pamięci (95. percentyl)", "sort_ascending": True},
        "latency": {"name": "Szczytowa latencja (95. percentyl)", "sort_ascending": True},
        "network_tx": {
            "name": "Szczytowa przepustowość wyjściowa (95. percentyl) (MB/s)",
            "sort_ascending": True,
        },
    },
}

PLOT_PALETTE = {"CSR": "#1f77b4", "SSR": "#d62728", "Uncategorized": "gray"}
