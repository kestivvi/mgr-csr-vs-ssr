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
# Mapping of Research Subject names (underscored) to logical analysis groups
DEFAULT_GROUPS = {
    "SSR-Node": [
        "ssr_analogjs",
        "ssr_angular",
        "ssr_astro",
        "ssr_lit",
        "ssr_nextjs",
        "ssr_nuxtjs",
        "ssr_qwik_city",
        "ssr_react_router",
        "ssr_solid_start",
        "ssr_sveltekit",
        "ssr_tanstack_start_react",
        "ssr_tanstack_start_solid",
    ],
    "SSR-Bun": [
        "ssr_astro_bun",
        "ssr_nextjs_bun",
        "ssr_nuxtjs_bun",
        "ssr_qwik_city_bun",
        "ssr_solid_start_bun",
        "ssr_sveltekit_bun",
        "ssr_tanstack_start_react_bun",
        "ssr_tanstack_start_solid_bun",
    ],
    "SSR-Deno": [
        "ssr_fresh",
    ],
    "CSR-Nginx": [
        "csr_angular",
        "csr_lit",
        "csr_react",
        "csr_solidjs",
        "csr_sveltekit_static",
        "csr_vanilla_nginx",
        "csr_vue",
    ],
    "CSR-Apache": [
        "csr_solidjs_apache",
        "csr_vanilla_apache",
    ],
}
