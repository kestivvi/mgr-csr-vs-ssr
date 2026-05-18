import datetime
from pathlib import Path

import pytest
import yaml

from orchestrator.actions.analyze.engine import PerformanceAnalyzer


@pytest.fixture
def mock_aggregated_study_dir(tmp_path: Path) -> Path:
    """Creates a temporary aggregated study directory with simulated timeseries metrics."""
    study_dir = tmp_path / "agg_study"
    study_dir.mkdir()

    # Create subdirectories
    (study_dir / "metrics").mkdir()
    (study_dir / "tool_results").mkdir()
    (study_dir / "logs").mkdir()

    # We mock 2 repetitions for ssr-nextjs-node and 1 repetition for csr-vanilla-nginx
    # Each repetition needs at least 35 rows to test the rolling rolling window window (30 periods)
    base_ts = datetime.datetime(2026, 5, 17, 12, 0, 0)

    # 1. ssr-nextjs-node Run 1
    metrics_a1 = ["timestamp,k6_successful_html_reqs_rate,k6_total_html_reqs_rate,cpu,memory"]
    for idx in range(40):
        ts = (base_ts + datetime.timedelta(seconds=idx)).isoformat()
        # Simulated ramping RPS and metrics
        rps = 100 + idx * 10
        cpu = 0.2 + (idx * 0.01)
        mem = 100 * 1024 * 1024 + (idx * 1024 * 1024)
        metrics_a1.append(f"{ts},{rps},{rps},{cpu},{mem}")
    with open(study_dir / "metrics" / "01_ssr_nextjs_node.csv", "w") as f:
        f.write("\n".join(metrics_a1))
    with open(study_dir / "tool_results" / "01_ssr_nextjs_node_wrk.json", "w") as f:
        f.write('{"rps": 500, "latency_avg": "5ms"}')

    # 2. ssr-nextjs-node Run 2
    metrics_a2 = ["timestamp,k6_successful_html_reqs_rate,k6_total_html_reqs_rate,cpu,memory"]
    for idx in range(40):
        ts = (base_ts + datetime.timedelta(seconds=idx)).isoformat()
        rps = 110 + idx * 10
        cpu = 0.22 + (idx * 0.01)
        mem = 105 * 1024 * 1024 + (idx * 1024 * 1024)
        metrics_a2.append(f"{ts},{rps},{rps},{cpu},{mem}")
    with open(study_dir / "metrics" / "02_ssr_nextjs_node.csv", "w") as f:
        f.write("\n".join(metrics_a2))
    with open(study_dir / "tool_results" / "02_ssr_nextjs_node_wrk.json", "w") as f:
        f.write('{"rps": 510, "latency_avg": "5.1ms"}')

    # 3. csr-vanilla-nginx Run 1
    metrics_b1 = ["timestamp,k6_successful_html_reqs_rate,k6_total_html_reqs_rate,cpu,memory"]
    for idx in range(40):
        ts = (base_ts + datetime.timedelta(seconds=idx)).isoformat()
        rps = 500 + idx * 50
        cpu = 0.05 + (idx * 0.002)
        mem = 10 * 1024 * 1024 + (idx * 100 * 1024)
        metrics_b1.append(f"{ts},{rps},{rps},{cpu},{mem}")
    with open(study_dir / "metrics" / "03_csr_vanilla_nginx.csv", "w") as f:
        f.write("\n".join(metrics_b1))
    with open(study_dir / "tool_results" / "03_csr_vanilla_nginx_wrk.json", "w") as f:
        f.write('{"rps": 2500, "latency_avg": "0.5ms"}')

    # Aggregated Metadata matching the research contract
    meta = {
        "test_type": "capacity_k6",
        "parameters": {
            "test_type": "capacity_k6",
            "rate": 1000,
            "warmup_duration": "10s",
            "capacity_k6_options": {
                "peak_rate": 5000,
                "max_vus": 400,
                "ramp_up": "10m",
                "sustain": "1m",
            },
        },
        "applications": {
            "ssr-nextjs-node": {
                "family": "react",
                "meta_framework": "nextjs",
                "strategy": "ssr",
                "runtime": "node",
            },
            "csr-vanilla-nginx": {
                "family": "vanilla",
                "meta_framework": None,
                "strategy": "csr",
                "runtime": "nginx",
            },
        },
    }
    with open(study_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)

    return study_dir


@pytest.fixture
def mock_load_study_dir(tmp_path: Path) -> Path:
    """Creates a temporary aggregated Load Test study directory."""
    study_dir = tmp_path / "load_study"
    study_dir.mkdir()
    (study_dir / "metrics").mkdir()
    (study_dir / "tool_results").mkdir()
    (study_dir / "logs").mkdir()

    base_ts = datetime.datetime(2026, 5, 17, 12, 0, 0)

    # 2 Applications, 2 Repetitions each. Load Test holds a fixed rate.
    specs = {
        "ssr-nextjs-node": [(0.50, 200 * 1024 * 1024), (0.55, 210 * 1024 * 1024)],
        "csr-vanilla-nginx": [(0.08, 20 * 1024 * 1024), (0.09, 21 * 1024 * 1024)],
    }
    for tech, reps in specs.items():
        for rep_idx, (cpu, mem) in enumerate(reps, start=1):
            rows = ["timestamp,cpu,memory"]
            for idx in range(40):
                ts = (base_ts + datetime.timedelta(seconds=idx)).isoformat()
                rows.append(f"{ts},{cpu + idx * 0.001},{mem + idx * 1024 * 1024}")
            fname = f"{rep_idx:02d}_{tech.replace('-', '_')}.csv"
            with open(study_dir / "metrics" / fname, "w") as f:
                f.write("\n".join(rows))

    meta = {
        "test_type": "load",
        "parameters": {
            "test_type": "load",
            "rate": 1000,
            "warmup_duration": "10s",
        },
        "applications": {
            "ssr-nextjs-node": {
                "family": "react",
                "meta_framework": "nextjs",
                "strategy": "ssr",
                "runtime": "node",
            },
            "csr-vanilla-nginx": {
                "family": "vanilla",
                "meta_framework": None,
                "strategy": "csr",
                "runtime": "nginx",
            },
        },
    }
    with open(study_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)

    return study_dir


def test_performance_analyzer_load_generates_resource_bar_plots(
    mock_load_study_dir: Path,
) -> None:
    """`mgr analyze load` produces capacity-style CPU/RAM utilisation bar charts."""
    analyzer = PerformanceAnalyzer(
        input_dir=mock_load_study_dir,
        report_type="load",
        force=True,
    )
    analyzer.run()

    assert (analyzer.plots_dir / "load_cpu_comparison.png").exists()
    assert (analyzer.plots_dir / "load_ram_comparison.png").exists()

    report_content = analyzer.report_path.read_text()
    assert "### Porównanie zużycia zasobów" in report_content
    assert "load_cpu_comparison.png" in report_content
    assert "load_ram_comparison.png" in report_content


def test_performance_analyzer_capacity_k6_report_generation(
    mock_aggregated_study_dir: Path,
) -> None:
    """Verifies that PerformanceAnalyzer runs, draws plots, and saves markdown reports."""
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(
        input_dir=mock_aggregated_study_dir,
        report_type="capacity_k6",
        force=False,
    )
    analyzer.run()

    # 1. Assert timestamped output analysis directory was created
    assert analyzer.output_dir.exists()
    assert analyzer.plots_dir.exists()

    # 2. Check if Matplotlib PNG plots were generated successfully
    assert (analyzer.plots_dir / "successful_throughput_rps_timeseries.png").exists()
    assert (analyzer.plots_dir / "cpu_usage_timeseries.png").exists()
    assert (analyzer.plots_dir / "ram_usage_timeseries.png").exists()
    assert (analyzer.plots_dir / "capacity_rps_comparison.png").exists()
    assert (analyzer.plots_dir / "capacity_cpu_at_sustained_usage.png").exists()
    assert (analyzer.plots_dir / "capacity_ram_at_sustained_usage.png").exists()

    # 3. Check if academic markdown report was compiled
    assert analyzer.report_path.exists()

    with open(analyzer.report_path, "r") as f:
        report_content = f.read()

    assert f"# Capacity Report for `{mock_aggregated_study_dir.name}`" in report_content
    assert "Podsumowanie wyników zagregowanych" in report_content

    # The markdown table must contain computed values for the technologies
    assert "ssr-nextjs-node" in report_content
    assert "csr-vanilla-nginx" in report_content
