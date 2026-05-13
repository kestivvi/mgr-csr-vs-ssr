class MetricName:
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK_TX = "network_tx"
    NETWORK_RX = "network_rx"
    K6_SUCCESSFUL_RPS = "k6_successful_html_reqs_rate"
    K6_TOTAL_RPS = "k6_total_html_reqs_rate"


class Column:
    TIMESTAMP = "timestamp"
    METRIC = "metric"
    VALUE = "metric_value"
    SERVER_TYPE = "server_type"
    GROUP = "group"
    RUN_NUMBER = "run_number"
    TIME_SEC = "time_sec"


# Standard taxonomy for the Master's Thesis
DEFAULT_GROUPS = {
    "SSR-Node": ["ssr-nextjs-node", "ssr-remix-node", "ssr-astro-node"],
    "SSR-Bun": ["ssr-nextjs-bun", "ssr-remix-bun", "ssr-astro-bun"],
    "CSR-Nginx": ["csr-nextjs-nginx", "csr-remix-nginx", "csr-astro-nginx"],
}
