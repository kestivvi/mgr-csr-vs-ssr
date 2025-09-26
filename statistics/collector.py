import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

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
    }
}

def download_metric(prom, metric_name, server_type, start_time_dt, end_time_dt):
    """
    Downloads a single metric for a given time range and aligns it to the
    exact start/end times using a robust resampling strategy with debugging.
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
        return pd.DataFrame()

    metric_data = prom.custom_query_range(
        query=promql,
        start_time=start_time_dt,
        end_time=end_time_dt,
        step="1s"
    )

    if not metric_data:
        print("WARNING: Prometheus returned no data for this query.")
        # Create an empty DataFrame with the correct structure to avoid errors downstream
        df = pd.DataFrame(columns=['timestamp', 'metric_value', 'server_label'])
        return df

    # --- DATA PROCESSING PIPELINE WITH DEBUGGING ---

    # STEP 1: Load raw data from Prometheus into a DataFrame
    all_dfs = []
    for series in metric_data:
        if not series.get('values'):
            continue
        df = pd.DataFrame(series['values'])
        df.columns = ['timestamp', 'metric_value']
        # --- SOLUTION 1 APPLIED HERE ---
        # Convert to datetime and immediately make it timezone-aware (UTC).
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
        all_dfs.append(df)

    if not all_dfs:
        print("WARNING: Prometheus returned data, but all series had empty 'values'.")
        return pd.DataFrame(columns=['timestamp', 'metric_value', 'server_label'])

    combined_df = pd.concat(all_dfs, ignore_index=True).dropna()
    combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    
    print(f"--- DEBUG: STEP 1 (Raw Data) ---")
    print(f"Shape: {combined_df.shape}")
    print("Head:\n", combined_df.head(3))
    print("Tail:\n", combined_df.tail(3))
    
    if combined_df.empty:
        print("WARNING: DataFrame is empty after loading. No data to process.")
        return pd.DataFrame(columns=['timestamp', 'metric_value', 'server_label'])

    # STEP 2: Resample data to a 1-second frequency to handle irregular timestamps
    combined_df.set_index('timestamp', inplace=True)
    resampled_df = combined_df.resample('1s').mean()

    print(f"\n--- DEBUG: STEP 2 (Resampled Data) ---")
    print(f"Shape: {resampled_df.shape}")
    print("Head:\n", resampled_df.head(3))
    print("Tail:\n", resampled_df.tail(3))

    # STEP 3: Reindex to the ideal, synchronized window to create a complete grid
    ideal_index = pd.date_range(start=start_time_dt, end=end_time_dt, freq='s')
    aligned_df = resampled_df.reindex(ideal_index)

    print(f"\n--- DEBUG: STEP 3 (Aligned to Ideal Index) ---")
    print(f"Shape: {aligned_df.shape}")
    print("Head:\n", aligned_df.head(3))
    print("Tail:\n", aligned_df.tail(3))

    # STEP 4: Fill all gaps robustly
    aligned_df['metric_value'] = aligned_df['metric_value'].ffill().bfill()

    print(f"\n--- DEBUG: STEP 4 (Gaps Filled) ---")
    print(f"Shape: {aligned_df.shape}")
    print("Head:\n", aligned_df.head(3))
    print("Tail:\n", aligned_df.tail(3))

    # If the entire series is still NaN (i.e., no data at all), fill with 0.
    if aligned_df['metric_value'].isnull().all():
        print("WARNING: No valid data points found in the window. Filling with 0.")
        aligned_df['metric_value'].fillna(0, inplace=True)

    # STEP 5: Reconstruct the final DataFrame
    aligned_df['server_label'] = server_type
    aligned_df.reset_index(inplace=True)
    aligned_df.rename(columns={'index': 'timestamp'}, inplace=True)

    print(f"\n--- DEBUG: STEP 5 (Final DataFrame) ---")
    print(f"Shape: {aligned_df.shape}")
    print("Head:\n", aligned_df.head(3))
    print("Tail:\n", aligned_df.tail(3))
    
    return aligned_df


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
        from prometheus_api_client import PrometheusConnect
        prom = PrometheusConnect(url=args.prometheus_url, disable_ssl=True)
    except Exception as e:
        print(f"ERROR: Could not connect to Prometheus: {e}", file=sys.stderr)
        sys.exit(1)

    start_time = datetime.fromtimestamp(args.start_epoch, tz=timezone.utc)
    end_time = datetime.fromtimestamp(args.end_epoch, tz=timezone.utc)

    print(f"INFO: Querying data for synchronized window: {start_time.isoformat()} to {end_time.isoformat()}")

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
