# Orchestrator

Python package behind the `mgr` CLI. See [mgr-code/README.md](../README.md) for the project overview and installation. This document is a reference for the experiment YAML schema.

## Experiment file schema

Custom YAML files are strictly validated by Pydantic.

```yaml
test_type: capacity_k6 # load | capacity_k6 | capacity_wrk
num_repetitions: 1

# --- test_type: load ---
load_options:
  rps: 100
  duration: 5m
  vus: 200
  path_type: dynamic # static | dynamic
  timeout: 0.4s

# --- test_type: capacity_k6 ---
capacity_k6_options:
  peak_rate: 1000
  ramp_up: 5m
  sustain: 1m
  ramp_down: 1m
  start_rate: 1
  warmup: 0s
  max_vus: 200
  path_type: dynamic
  timeout: 0.4s

# --- test_type: capacity_wrk ---
capacity_wrk_options:
  threads: 2
  connections: 10
  duration: 30s
  warmup: 30s
```

> [!IMPORTANT]
> The orchestrator auto-discovers provisioned application servers from the Terraform inventory. Do not specify hostnames or IPs in experiment files.

### Field reference

| Block            | Field             | Type  | Default      | Description                              |
| :--------------- | :---------------- | :---- | :----------- | :--------------------------------------- |
| **Root**         | `test_type`       | `str` | **Required** | `load`, `capacity_k6`, `capacity_wrk`    |
|                  | `num_repetitions` | `int` | `1`          | Number of times to repeat the experiment |
| **Load**         | `rps`             | `int` | `100`        | Target requests per second               |
|                  | `duration`        | `str` | `5m`         | Duration of the test                     |
|                  | `vus`             | `int` | `200`        | Number of virtual users                  |
|                  | `path_type`       | `str` | `dynamic`    | `static` or `dynamic` paths              |
|                  | `timeout`         | `str` | `0.4s`       | HTTP request timeout                     |
| **Capacity k6**  | `peak_rate`       | `int` | `1000`       | Target RPS at peak                       |
|                  | `ramp_up`         | `str` | `5m`         | Ramp-up duration                         |
|                  | `sustain`         | `str` | `1m`         | Sustain duration at peak                 |
|                  | `ramp_down`       | `str` | `1m`         | Ramp-down duration                       |
|                  | `start_rate`      | `int` | `1`          | Starting RPS                             |
|                  | `warmup`          | `str` | `0s`         | Initial stay duration at `start_rate`    |
|                  | `max_vus`         | `int` | `200`        | Max pre-allocated VUs                    |
|                  | `path_type`       | `str` | `dynamic`    | `static` or `dynamic` paths              |
|                  | `timeout`         | `str` | `0.4s`       | HTTP request timeout                     |
| **Capacity wrk** | `threads`         | `int` | `2`          | Number of threads                        |
|                  | `connections`     | `int` | `10`         | Number of connections                    |
|                  | `duration`        | `str` | `30s`        | Duration of the test                     |
|                  | `warmup`          | `str` | `30s`        | Warmup duration                          |

## Layout

- `src/orchestrator/actions/` — domain logic (setup, test, analyze, destroy)
- `src/orchestrator/shared/` — shared utilities (runner, logging)
- `src/orchestrator/config.py` — CWD-independent path resolution
- `src/orchestrator/main.py` — Typer CLI entry point
