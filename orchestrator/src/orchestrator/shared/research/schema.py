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
    REPETITION_NUMBER = "repetition_number"
    TIME_SEC = "time_sec"


# Standard taxonomy for the Master's Thesis
# Mapping of Research Subject names to logical analysis groups
def get_default_groups() -> dict[str, list[str]]:
    """
    Returns the standard taxonomy for the Master's Thesis.
    Derived dynamically from the discovered apps.
    """
    from orchestrator.config import SUBJECTS_DIR
    from orchestrator.shared.research.subject import SubjectRegistry

    registry = SubjectRegistry(SUBJECTS_DIR)
    return registry.get_default_groups()
