# statistics/collector.py

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from functools import reduce

# --- CONFIGURATION ---

# The PromQL queries for the metrics you want.
# The '{server_label}' placeholder will be filled in by the script.
METRIC_CONFIG = {
    'cpu': {
        'name': 'CPU Utilization',
        'query': 'avg by (server) (1 - rate(node_cpu_seconds_total{{mode="idle", server="{server_label}"}}[15s]))',
    },
    'memory': {
        'name': 'Memory Usage',
        'query': 'sum by (server) (node_memory_MemTotal_bytes{{server="{server_label}"}} - node_memory_MemAvailable_bytes{{server="{server_label}"}})',
    },
    'latency': {
        'name': 'p95 Request Latency',
        'query': 'histogram_quantile(0.95, sum by (le, server) (rate(k6_http_req_duration_seconds{{server="{server_label}"}}[15s]))) * 1000',
    },
    'network_tx': {
        'name': 'Network Transmit Rate',
        'query': 'sum by (server) (rate(node_network_transmit_bytes_total{{server="{server_label}"}}[15s]))',
    },
    'k6_successful_html_reqs_rate': {
        'name': 'Successful HTML Requests Rate',
        'query': 'sum by (server) (rate(k6_http_reqs_total{{server="{server_label}", resource_type="html", status="200", error_code=""}}[15s]))',
    },
    'k6_total_html_reqs_rate': {
        'name': 'Total HTML Requests Rate',
        'query': 'sum by (server) (rate(k6_http_reqs_total{{server="{server_label}", resource_type="html"}}[15s]))',
    }
}

def download_metric(prom, metric_name, server_type, start_time_dt, end_time_dt):
    """
    Downloads a single metric, processes it, and returns a DataFrame
    ready for merging (timestamp index, single metric column).
    """
    config = METRIC_CONFIG[metric_name]
    print(f"\n--- Downloading {config['name']} ---")
    
    server_label_for_query = server_type.upper()
    
    promql = config['query'].format(server_label=server_label_for_query)
    print(f"INFO: Querying Prometheus for: {promql}")
    
    try:
        from prometheus_api_client import PrometheusConnect
        prom = PrometheusConnect(url=prom.url, disable_ssl=True)
    except Exception as e:
        print(f"ERROR: Could not re-connect to Prometheus: {e}", file=sys.stderr)
        return None

    metric_data = prom.custom_query_range(
        query=promql,
        start_time=start_time_dt,
        end_time=end_time_dt,
        step="1s"
    )

    if not metric_data:
        print("WARNING: Prometheus returned no data for this query.")
        return None

    all_dfs = []
    for series in metric_data:
        if not series.get('values'):
            continue
        df = pd.DataFrame(series['values'])
        df.columns = ['timestamp', 'metric_value']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
        all_dfs.append(df)

    if not all_dfs:
        print("WARNING: Prometheus returned data, but all series had empty 'values'.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True).dropna()
    combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    
    if combined_df.empty:
        print("WARNING: DataFrame is empty after loading. No data to process.")
        return None

    # --- START: MODIFIED SECTION ---
    # Prepare DataFrame for merging: set index and rename metric column
    combined_df.set_index('timestamp', inplace=True)
    combined_df.rename(columns={'metric_value': metric_name}, inplace=True)
    # --- END: MODIFIED SECTION ---
    
    return combined_df


def main():
    """Main function to parse arguments and orchestrate the data download."""
    parser = argparse.ArgumentParser(description="Download and aggregate metrics from Prometheus for a specific test run.")
    parser.add_argument('--prometheus-url', required=True, help="URL of the Prometheus server (e.g., http://localhost:9090)")
    parser.add_argument('--start-epoch', required=True, type=int, help="Start time for the query as a Unix epoch timestamp.")
    parser.add_argument('--end-epoch', required=True, type=int, help="End time for the query as a Unix epoch timestamp.")
    parser.add_argument('--server-type', required=True, help="The type of server to query for (e.g., 'SolidJS-Nginx')")
    parser.add_argument('--run-number', required=True, type=int, help="The run number of the experiment (e.g., 5)")
    parser.add_argument('--output-dir', required=True, type=Path, help="Directory to save the output CSV files")
    args = parser.parse_args()

    # --- START: MODIFIED SECTION ---
    # Create a dedicated subdirectory for metric files
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # --- END: MODIFIED SECTION ---

    print(f"INFO: Connecting to Prometheus at {args.prometheus_url}...")
    try:
        from prometheus_api_client import PrometheusConnect
        prom = PrometheusConnect(url=args.prometheus_url, disable_ssl=True)
    except Exception as e:
        print(f"ERROR: Could not connect to Prometheus: {e}", file=sys.stderr)
        sys.exit(1)

    start_time = datetime.fromtimestamp(args.start_epoch, tz=timezone.utc)
    end_time = datetime.fromtimestamp(args.end_epoch, tz=timezone.utc)

    print(f"INFO: Querying data for synchronized window: {start_time.isoformat()} to {end_time.isoformat()}")

    # --- START: MODIFIED SECTION ---
    # Phase 1: Download all metrics into a list of DataFrames
    all_metric_dfs = []
    for metric_name in METRIC_CONFIG.keys():
        df = download_metric(prom, metric_name, args.server_type, start_time, end_time)
        if df is not None and not df.empty:
            all_metric_dfs.append(df)

    if not all_metric_dfs:
        print("ERROR: Failed to download any metrics. No output file will be generated.", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Merge all DataFrames into a single wide-format DataFrame
    # Using reduce for a robust merge of multiple dataframes
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), all_metric_dfs)

    # Phase 3: Post-processing
    # Create a full 1-second interval time range to ensure no gaps
    ideal_index = pd.date_range(start=start_time, end=end_time, freq='s')
    merged_df = merged_df.reindex(ideal_index)
    
    # Fill any gaps that might have occurred during merging or reindexing
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)
    merged_df.fillna(0, inplace=True) # Fill any remaining NaNs with 0

    # Phase 4: Save the aggregated file
    sanitized_server_type = args.server_type.lower().replace('-', '_')
    filename = f"{args.run_number:02d}_{sanitized_server_type}.csv"
    
    output_path = metrics_dir / filename
    merged_df.to_csv(output_path, index=True, index_label='timestamp')
    print(f"\nINFO: Successfully saved aggregated metrics to {output_path}")
    # --- END: MODIFIED SECTION ---


if __name__ == "__main__":
    main()
