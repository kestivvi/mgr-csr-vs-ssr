`wrk` generates massive HTTP load. It uses few operating system threads and non-blocking I/O. This maximizes throughput.

**Workflow**
Set threads to match your CPU cores. Set connections higher than threads to simulate concurrency. Read the summary report.

**Common Commands**

**1. Basic Load Test**
This tests a single endpoint. It uses 4 threads and 100 open connections for 30 seconds.

```bash
wrk -t4 -c100 -d30s http://localhost:8080/api
```

In this codebase we specifically use:

```bash
wrk -t2 -c100 -d10s http://localhost:80
```

**2. Print Latency Distribution**
The `--latency` flag adds percentile statistics to the final report. It shows 50th, 75th, 90th, and 99th percentile response times.

```bash
wrk -t4 -c100 -d30s --latency http://localhost:8080/api
```

**3. Test with Timeout**
The `--timeout` flag enforces a strict response deadline. It records requests slower than 2 seconds as timeout errors.

```bash
wrk -t2 -c50 -d10s --timeout 2s http://localhost:8080/api
```

**4. Use Lua Script**
The `-s` flag loads a Lua script. Lua scripts customize the HTTP request or process the response data.

```bash
wrk -t4 -c100 -d30s -s post.lua http://localhost:8080/api
```

**Example Lua Script (`post.lua`)**
This changes the default GET request to a POST request. It injects a JSON body and custom headers.

```lua
wrk.method = "POST"
wrk.body   = '{"key": "value"}'
wrk.headers["Content-Type"] = "application/json"
```
