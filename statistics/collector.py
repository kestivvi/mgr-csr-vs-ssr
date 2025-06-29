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
    }
}

def download_metric(prom, metric_name, server_type, start_time, end_time):
    """Downloads a single metric for a given time range."""
    config = METRIC_CONFIG[metric_name]
    print(f"--- Downloading {config['name']} ---")
    
    promql = config['query'].format(server_type=server_type)
    print(f"INFO: Querying Prometheus for: {promql}")
    
    # Use custom_query_range for full PromQL queries
    metric_data = prom.custom_query_range(
        query=promql,
        start_time=start_time,
        end_time=end_time,
        step="1s" # Use a 1-second step for high resolution
    )

    if not metric_data:
        print(f"WARNING: No data returned for query: {promql}")
        return pd.DataFrame()

    # The result is a list of metrics. Even with one server, we get a list.
    # Convert the first (and only) result to a DataFrame.
    # The library now returns a DataFrame directly in the 'values' key.
    metric_df = pd.DataFrame(metric_data[0]['values'])
    metric_df.columns = ['timestamp', 'metric_value']
    metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], unit='s')
    return metric_df


def main():
    """Main function to parse arguments and orchestrate the data download."""
    parser = argparse.ArgumentParser(description="Download metrics from Prometheus for a specific test run.")
    parser.add_argument('--prometheus-url', required=True, help="URL of the Prometheus server (e.g., http://localhost:9090)")
    parser.add_argument('--start', required=True, help="Start time for the query in ISO 8601 format (e.g., 2023-10-28T10:00:00Z)")
    parser.add_argument('--end', required=True, help="End time for the query in ISO 8601 format")
    parser.add_argument('--server-type', required=True, choices=['CSR', 'SSR'], help="The type of server to query for (CSR or SSR)")
    parser.add_argument('--run-number', required=True, type=int, help="The run number of the experiment (e.g., 5)")
    parser.add_argument('--output-dir', required=True, type=Path, help="Directory to save the output CSV files")
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"INFO: Connecting to Prometheus at {args.prometheus_url}...")
    try:
        prom = PrometheusConnect(url=args.prometheus_url, disable_ssl=True)
    except Exception as e:
        print(f"ERROR: Could not connect to Prometheus: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert string times to datetime objects
    start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(args.end.replace('Z', '+00:00'))


    for metric_name in METRIC_CONFIG.keys():
        df = download_metric(prom, metric_name, args.server_type, start_time, end_time)

        # Construct filename and save
        if not df.empty:
            filename = f"{args.server_type.lower()}_run_{args.run_number:02d}_{metric_name}.csv"
            output_path = args.output_dir / filename
            df.to_csv(output_path, index=False)
            print(f"INFO: Successfully saved {metric_name} data to {output_path}")


if __name__ == "__main__":
    main() 