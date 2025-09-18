import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from prometheus_api_client import PrometheusConnect

# --- CONFIGURATION ---

# The PromQL queries for the metrics you want.
# The '{server_label}' placeholder will be filled in by the script.
METRIC_CONFIG = {
    'cpu': {
        'name': 'CPU Utilization',
        'query': 'avg by (server) (1 - rate(node_cpu_seconds_total{{mode="idle", server="{server_label}"}}[15s]))',
    },
    'memory': {
        'name': 'Memory Utilization',
        'query': 'sum by (server) (1 - (node_memory_MemAvailable_bytes{{server="{server_label}"}} / node_memory_MemTotal_bytes{{server="{server_label}"}}))',
    },
    'latency': {
        'name': 'p95 Request Latency',
        'query': 'histogram_quantile(0.95, sum by (le, server) (rate(k6_http_req_duration_seconds{{server="{server_label}"}}[15s]))) * 1000',
    }
}

def download_metric(prom, metric_name, server_type, start_time_dt, end_time_dt):
    """
    Downloads a single metric for a given time range, correctly handling multi-series responses.
    """
    config = METRIC_CONFIG[metric_name]
    print(f"--- Downloading {config['name']} ---")
    
    # --- FIX: Always use the uppercase server label for all queries for consistency ---
    server_label_for_query = server_type.upper()
    
    promql = config['query'].format(server_label=server_label_for_query)
    print(f"INFO: Querying Prometheus for: {promql}")
    
    metric_data = prom.custom_query_range(
        query=promql,
        start_time=start_time_dt,
        end_time=end_time_dt,
        step="1s"
    )

    if not metric_data:
        print(f"WARNING: No data returned for query: {promql}")
        return pd.DataFrame()

    all_dfs = []
    for series in metric_data:
        # We always want to save the original server_type in the CSV for consistency
        series_server_label = server_type
        
        df = pd.DataFrame(series['values'])
        df.columns = ['timestamp', 'metric_value']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['server_label'] = series_server_label
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    
    return combined_df


def main():
    """Main function to parse arguments and orchestrate the data download."""
    parser = argparse.ArgumentParser(description="Download metrics from Prometheus for a specific test run.")
    parser.add_argument('--prometheus-url', required=True, help="URL of the Prometheus server (e.g., http://localhost:9090)")
    parser.add_argument('--start-epoch', required=True, type=int, help="Start time for the query as a Unix epoch timestamp.")
    parser.add_argument('--end-epoch', required=True, type=int, help="End time for the query as a Unix epoch timestamp.")
    parser.add_argument('--server-type', required=True, help="The type of server to query for (e.g., 'SolidJS-Nginx')")
    parser.add_argument('--run-number', required=True, type=int, help="The run number of the experiment (e.g., 5)")
    parser.add_argument('--output-dir', required=True, type=Path, help="Directory to save the output CSV files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"INFO: Connecting to Prometheus at {args.prometheus_url}...")
    try:
        prom = PrometheusConnect(url=args.prometheus_url, disable_ssl=True)
    except Exception as e:
        print(f"ERROR: Could not connect to Prometheus: {e}", file=sys.stderr)
        sys.exit(1)

    start_time = datetime.fromtimestamp(args.start_epoch, tz=timezone.utc)
    end_time = datetime.fromtimestamp(args.end_epoch, tz=timezone.utc)

    print(f"INFO: Querying data from {start_time.isoformat()} to {end_time.isoformat()}")

    for metric_name in METRIC_CONFIG.keys():
        df = download_metric(prom, metric_name, args.server_type, start_time, end_time)

        if not df.empty:
            sanitized_server_type = args.server_type.lower().replace('-', '_')
            filename = f"{args.run_number:02d}_{sanitized_server_type}_{metric_name}.csv"
            
            output_path = args.output_dir / filename
            df.to_csv(output_path, index=False)
            print(f"INFO: Successfully saved {metric_name} data to {output_path}")


if __name__ == "__main__":
    main()
