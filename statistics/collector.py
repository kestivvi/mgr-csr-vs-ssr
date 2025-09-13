import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from prometheus_api_client import PrometheusConnect

# --- CONFIGURATION ---

# The PromQL queries for the metrics you want.
# The '{server_type}' placeholder will be filled in by the script.
METRIC_CONFIG = {
    'cpu': {
        'name': 'CPU Utilization',
        'query': 'avg by (server) (1 - rate(node_cpu_seconds_total{{mode="idle", server=~"{server_type}"}}[1m]))'
    },
    'memory': {
        'name': 'Memory Utilization',
        'query': 'sum by (server) (1 - (node_memory_MemAvailable_bytes{{server=~"{server_type}"}} / node_memory_MemTotal_bytes{{server=~"{server_type}"}}))'
    },
    # --- NEW METRIC ADDED ---
    'latency': {
        'name': 'p95 Request Latency',
        # This query calculates the 95th percentile latency from the k6 histogram data.
        # It is converted to milliseconds by multiplying by 1000.
        'query': 'histogram_quantile(0.95, sum by (le, server) (rate(k6_http_req_duration_seconds{{server=~"{server_type}"}}[1m]))) * 1000'
    }
}

def download_metric(prom, metric_name, server_type, start_time, end_time):
    """Downloads a single metric for a given time range."""
    config = METRIC_CONFIG[metric_name]
    print(f"--- Downloading {config['name']} ---")
    
    # This regex will match the app server (e.g., 'NextJS-Bun') OR its load generator ('LG_NextJS-Bun').
    server_type_regex = f"{server_type}|LG_{server_type}"
    
    promql = config['query'].format(server_type=server_type_regex)
    print(f"INFO: Querying Prometheus for: {promql}")
    
    metric_data = prom.custom_query_range(
        query=promql,
        start_time=start_time,
        end_time=end_time,
        step="1s"
    )

    if not metric_data:
        print(f"WARNING: No data returned for query: {promql}")
        return pd.DataFrame()

    all_dfs = []
    for series in metric_data:
        series_server_label = series['metric']['server']
        
        df = pd.DataFrame(series['values'])
        df.columns = ['timestamp', 'metric_value']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['server_label'] = series_server_label
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def main():
    """Main function to parse arguments and orchestrate the data download."""
    parser = argparse.ArgumentParser(description="Download metrics from Prometheus for a specific test run.")
    parser.add_argument('--prometheus-url', required=True, help="URL of the Prometheus server (e.g., http://localhost:9090)")
    parser.add_argument('--start', required=True, help="Start time for the query in ISO 8601 format (e.g., 2023-10-28T10:00:00Z)")
    parser.add_argument('--end', required=True, help="End time for the query in ISO 8601 format")
    parser.add_argument('--server-type', required=True, help="The type of server to query for (e.g., 'NextJS-Bun')")
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

    start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(args.end.replace('Z', '+00:00'))

    for metric_name in METRIC_CONFIG.keys():
        df = download_metric(prom, metric_name, args.server_type, start_time, end_time)

        if not df.empty:
            # The filename now represents the entire scenario (e.g., nextjs-bun)
            filename = f"{args.server_type.lower()}_run_{args.run_number:02d}_{metric_name}.csv"
            output_path = args.output_dir / filename
            df.to_csv(output_path, index=False)
            print(f"INFO: Successfully saved {metric_name} data to {output_path}")


if __name__ == "__main__":
    main()