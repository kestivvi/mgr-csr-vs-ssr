from orchestrator.actions.analyze.utils.group_summary import (
    MetricSpec,
    render_group_summary_section,
    render_per_family_group_summary_section,
)


def test_section_contains_header_and_metric_subheading() -> None:
    md = render_group_summary_section(
        per_app_values={
            "Utrzymany RPS": {
                "CSR": [30000.0, 35000.0, 36000.0],
                "SSR": [500.0, 1000.0, 2000.0],
            }
        },
        metric_specs=[
            MetricSpec(name="Utrzymany RPS", unit="RPS", decimals=0, higher_is="CSR"),
        ],
    )
    assert md.startswith("## Podsumowanie zbiorcze (CSR vs SSR)")
    assert "### Utrzymany RPS" in md
    assert "| CSR |" in md
    assert "| SSR |" in md


def test_precision_respects_decimals() -> None:
    md = render_group_summary_section(
        per_app_values={
            "CPU": {"CSR": [6.7, 6.8, 6.9], "SSR": [14.5, 14.6, 14.7]},
            "RPS": {"CSR": [30000.0, 35000.0], "SSR": [1000.0, 2000.0]},
        },
        metric_specs=[
            MetricSpec(name="CPU", unit="%", decimals=2, higher_is="SSR"),
            MetricSpec(name="RPS", unit="RPS", decimals=0, higher_is="CSR"),
        ],
    )
    assert "6.80" in md
    assert "14.60" in md
    assert "32 500" in md  # mean of 30000,35000 rendered as integer with thin space
    assert "32500.00" not in md  # no spurious decimals on int-precision metric


def test_ratio_line_uses_higher_is_direction() -> None:
    md = render_group_summary_section(
        per_app_values={
            "RPS": {"CSR": [30000.0, 35000.0], "SSR": [1000.0, 2000.0]},
            "CPU": {"CSR": [6.0, 7.0], "SSR": [14.0, 15.0]},
        },
        metric_specs=[
            MetricSpec(name="RPS", unit="RPS", decimals=0, higher_is="CSR"),
            MetricSpec(name="CPU", unit="%", decimals=2, higher_is="SSR"),
        ],
    )
    assert "Stosunek CSR/SSR (mediana):" in md
    assert "Stosunek SSR/CSR (mediana):" in md


def test_per_family_renders_section_for_family_with_both_strategies() -> None:
    md = render_per_family_group_summary_section(
        per_app_values_by_family={
            "react": {
                "Utrzymany RPS": {
                    "CSR": [30000.0, 35000.0],
                    "SSR": [1000.0, 2000.0],
                }
            }
        },
        metric_specs=[
            MetricSpec(name="Utrzymany RPS", unit="RPS", decimals=0, higher_is="CSR"),
        ],
    )
    assert md.startswith("## Podsumowanie zbiorcze według frameworku (CSR vs SSR)")
    assert "### Framework: React" in md
    assert "#### Utrzymany RPS" in md
    assert "| CSR |" in md
    assert "| SSR |" in md
    assert "Stosunek CSR/SSR (mediana):" in md


def test_per_family_omits_families_missing_one_strategy() -> None:
    md = render_per_family_group_summary_section(
        per_app_values_by_family={
            "vanilla": {"Utrzymany RPS": {"CSR": [30000.0, 35000.0]}},
            "qwik": {"Utrzymany RPS": {"SSR": [1000.0, 2000.0]}},
            "react": {"Utrzymany RPS": {"CSR": [30000.0], "SSR": [1000.0]}},
        },
        metric_specs=[
            MetricSpec(name="Utrzymany RPS", unit="RPS", decimals=0, higher_is="CSR"),
        ],
    )
    assert "### Framework: React" in md
    assert "### Framework: Vanilla" not in md
    assert "### Framework: Qwik" not in md


def test_per_family_orders_families_alphabetically() -> None:
    md = render_per_family_group_summary_section(
        per_app_values_by_family={
            "vue": {"RPS": {"CSR": [1.0], "SSR": [2.0]}},
            "angular": {"RPS": {"CSR": [1.0], "SSR": [2.0]}},
            "react": {"RPS": {"CSR": [1.0], "SSR": [2.0]}},
        },
        metric_specs=[MetricSpec(name="RPS", unit="RPS", decimals=0, higher_is="CSR")],
    )
    a = md.index("Framework: Angular")
    r = md.index("Framework: React")
    v = md.index("Framework: Vue")
    assert a < r < v


def test_per_family_empty_input_returns_empty_string() -> None:
    assert (
        render_per_family_group_summary_section(
            per_app_values_by_family={},
            metric_specs=[MetricSpec(name="RPS", unit="RPS", decimals=0, higher_is="CSR")],
        )
        == ""
    )
