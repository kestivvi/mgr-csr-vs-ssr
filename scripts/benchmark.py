import os
import subprocess
import time
import csv
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pathlib import Path
from threading import Lock

# Configuration
APPS_DIR = Path("../apps")
RESULTS_BASE_DIR = Path("../results")
WAIT_TIME = 12  # Time to wait for app to be ready

# wrk settings
WRK_THREADS = "2"
WRK_CONNECTIONS = "100"
WRK_WARMUP_DURATION = "30s"
WRK_MEASURE_DURATION = "30s"
WRK_MEASURE_RUNS = 5
WRK_URL = "https://localhost"

# Thread-safe printing
print_lock = Lock()

def log(message):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

def get_apps():
    apps = [d for d in os.listdir(APPS_DIR) if os.path.isdir(APPS_DIR / d)]
    return sorted(apps)

def parse_wrk_rps(output):
    """Extract Requests/sec from wrk output."""
    match = re.search(r"Requests/sec:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    return 0.0

def benchmark_app(app_name):
    app_path = APPS_DIR / app_name
    log(f"--- Benchmarking {app_name} ---")
    
    try:
        # Start the app
        log(f"Starting {app_name} with --force-recreate...")
        subprocess.run(
            ["docker-compose", "up", "-d", "--force-recreate"], 
            cwd=app_path, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        # Wait for the app to be ready
        log(f"Waiting {WAIT_TIME}s for app to stabilize...")
        time.sleep(WAIT_TIME)
        
        # 1. Warmup Phase
        log(f"Phase 1: Warmup ({WRK_WARMUP_DURATION})...")
        subprocess.run(
            ["wrk", "-t" + WRK_THREADS, "-c" + WRK_CONNECTIONS, "-d" + WRK_WARMUP_DURATION, WRK_URL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 2. Measurement Phase
        log(f"Phase 2: Measuring ({WRK_MEASURE_RUNS} runs of {WRK_MEASURE_DURATION})...")
        rps_scores = []
        for i in range(1, WRK_MEASURE_RUNS + 1):
            log(f"  Run {i}/{WRK_MEASURE_RUNS}: Executing wrk...")
            result = subprocess.run(
                ["wrk", "-t" + WRK_THREADS, "-c" + WRK_CONNECTIONS, "-d" + WRK_MEASURE_DURATION, WRK_URL],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                log(f"  ⚠️ wrk failed on run {i}: {result.stderr.strip()}")
                continue
                
            rps = parse_wrk_rps(result.stdout)
            log(f"  Result {i}: {rps:.2f} RPS")
            rps_scores.append(rps)
            
        if not rps_scores:
            return None
            
        # Statistical results
        stats = {
            'mean': np.mean(rps_scores),
            'std': np.std(rps_scores),
            'max': np.max(rps_scores),
            'min': np.min(rps_scores),
            'raw': rps_scores
        }
        log(f"📈 Result: Mean={stats['mean']:.2f} RPS, StdDev={stats['std']:.2f}")
        return stats
        
    except KeyboardInterrupt:
        log(f"\n⚠️ Benchmark interrupted by user.")
        raise
    except Exception as e:
        log(f"❌ Error benchmarking {app_name}: {str(e)}")
        return None
    finally:
        try:
            subprocess.run(["docker-compose", "down"], cwd=app_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

def generate_csv(results, output_path):
    log(f"Saving results to {output_path}...")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['app_name', 'mean_rps', 'std_dev', 'max_rps', 'min_rps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for app, stats in results.items():
            if stats:
                writer.writerow({
                    'app_name': app, 
                    'mean_rps': stats['mean'],
                    'std_dev': stats['std'],
                    'max_rps': stats['max'],
                    'min_rps': stats['min']
                })

def generate_chart(results, output_path):
    log(f"Generating chart: {output_path}...")
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        log("No results to plot.")
        return

    # Convert to DataFrame
    data = []
    for app, stats in valid_results.items():
        data.append({
            'app_name': app,
            'mean': stats['mean'],
            'std': stats['std'],
            'max': stats['max']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('mean', ascending=True)

    plt.figure(figsize=(14, 10))
    
    # Color logic
    colors = ['#3498db' if 'csr' in name.lower() else '#e74c3c' for name in df['app_name']]
    
    # Plot Mean with Error Bars (Standard Deviation)
    bars = plt.barh(df['app_name'], df['mean'], xerr=df['std'], color=colors, 
                    capsize=5, label='Mean RPS (with Std Dev)')
    
    # Mark Max RPS with a vertical line or point
    plt.scatter(df['max'], df['app_name'], color='black', marker='|', s=100, 
                zorder=10, label='Peak RPS (Max)')
    
    # Add mean values at the end of bars
    for i, row in df.iterrows():
        plt.text(row['mean'] + row['std'] + 50, i, f"{row['mean']:.1f}", 
                 va='center', fontsize=9, fontweight='bold')

    plt.xlabel('Requests Per Second (RPS)')
    plt.ylabel('Application')
    plt.title('Framework Performance Comparison (Capacity Baseline)\n30s Warmup + 5x 30s Measurement Runs')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path)
    log(f"Chart saved successfully.")

def save_metadata(output_dir):
    metadata = {
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "methodology": "Option 1: Standard Academic",
        "parameters": {
            "warmup": WRK_WARMUP_DURATION,
            "duration": WRK_MEASURE_DURATION,
            "runs": WRK_MEASURE_RUNS,
            "threads": WRK_THREADS,
            "connections": WRK_CONNECTIONS,
            "tool": "wrk"
        }
    }
    with open(output_dir / "metadata.yaml", 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

def main():
    apps = get_apps()
    log(f"Found {len(apps)} applications to benchmark.")
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = RESULTS_BASE_DIR / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Results will be saved in: {run_dir}")
    
    save_metadata(run_dir)
    
    results = {}
    for app in apps:
        stats = benchmark_app(app)
        results[app] = stats
        
    generate_csv(results, run_dir / "benchmark_results.csv")
    generate_chart(results, run_dir / "benchmark_rps_comparison.png")
    
    log(f"Benchmarking complete! Results in {run_dir}")

if __name__ == "__main__":
    main()
