import os
import subprocess
import concurrent.futures
import time
import requests
from threading import Lock

# Configuration
APPS_DIR = "../apps"
MAX_BUILD_WORKERS = 4  # Adjust based on your CPU/RAM
TEST_PORT = 443  # Standard port for these apps
CURL_TIMEOUT = 20

# Thread-safe printing
print_lock = Lock()

def log(message):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

def get_apps():
    return [d for d in os.listdir(APPS_DIR) if os.path.isdir(os.path.join(APPS_DIR, d))]

def build_app(app_name):
    app_path = os.path.join(APPS_DIR, app_name)
    log(f"Building {app_name}...")
    try:
        start_time = time.time()
        result = subprocess.run(
            ["docker-compose", "build"],
            cwd=app_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = time.time() - start_time
        if result.returncode == 0:
            log(f"✅ {app_name} built successfully in {duration:.1f}s")
            return app_name, True, ""
        else:
            log(f"❌ {app_name} build failed")
            return app_name, False, result.stderr
    except Exception as e:
        log(f"❌ {app_name} build error: {str(e)}")
        return app_name, False, str(e)

def test_app(app_name):
    app_path = os.path.join(APPS_DIR, app_name)
    log(f"Testing {app_name}...")
    try:
        # Start the app
        subprocess.run(["docker-compose", "up", "-d", "--force-recreate"], cwd=app_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Give it a moment to start
        time.sleep(10)
        
        # Test the endpoint (using curl -k for self-signed certs)
        test_result = subprocess.run(
            ["curl", "-k", "-s", "-o", "/dev/null", "-w", "%{http_code}", "https://localhost"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        http_code = test_result.stdout.strip()
        passed = http_code == "200"
        
        if passed:
            log(f"✅ {app_name} passed health check (HTTP 200)")
        else:
            log(f"❌ {app_name} failed health check (HTTP {http_code})")
            
        # Tear down
        subprocess.run(["docker-compose", "down"], cwd=app_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return passed, http_code
    except Exception as e:
        log(f"❌ {app_name} test error: {str(e)}")
        try:
            subprocess.run(["docker-compose", "down"], cwd=app_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
        return False, str(e)

def main():
    apps = get_apps()
    # Sort for deterministic order
    apps.sort()
    
    results = {}

    log(f"Starting verification for {len(apps)} applications...")

    # Step 1: Parallel Build
    log(f"Phase 1: Building apps in parallel (workers={MAX_BUILD_WORKERS})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_BUILD_WORKERS) as executor:
        build_futures = {executor.submit(build_app, app): app for app in apps}
        for future in concurrent.futures.as_completed(build_futures):
            app_name, success, error = future.result()
            results[app_name] = {"build": success, "test": False, "error": error}

    # Step 2: Sequential Test
    log("\nPhase 2: Testing apps sequentially...")
    for app_name in apps:
        if results[app_name]["build"]:
            passed, info = test_app(app_name)
            results[app_name]["test"] = passed
            results[app_name]["info"] = info
        else:
            results[app_name]["test"] = False
            results[app_name]["info"] = "Build failed"

    # Step 3: Summary
    log("\n" + "="*50)
    log("VERIFICATION SUMMARY")
    log("="*50)
    passed_count = 0
    for app_name in apps:
        status = "PASS" if results[app_name]["test"] else "FAIL"
        if status == "PASS": passed_count += 1
        log(f"{app_name:30} | {status} | Build: {'✅' if results[app_name]['build'] else '❌'} | Test: {'✅' if results[app_name]['test'] else '❌'}")
    
    log("="*50)
    log(f"Final Result: {passed_count}/{len(apps)} apps passed.")
    log("="*50)

if __name__ == "__main__":
    main()
