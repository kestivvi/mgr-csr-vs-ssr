from typing import Any, TypedDict

from pydantic import BaseModel, Field


class ScenarioMetadata(TypedDict):
    name: str
    app_server_ip: str
    load_generator_group: str
    monitoring_public_ip: str
    monitoring_private_ip: str


class ScenarioResult(TypedDict, total=False):
    success: bool
    name: str
    timestamps: dict[str, float] | None
    wrk_results: dict[str, Any] | None
    scenario: ScenarioMetadata


class LoadOptions(BaseModel):
    rps: int = 100
    warmup: str = "30s"
    duration: str = "1m"
    after: str = "30s"
    vus: int = 200
    path_type: str = "dynamic"  # 'static' or 'dynamic'
    timeout: str = "0.4s"
    skip_assets: bool = False


class CapacityK6Options(BaseModel):
    # Rate-based (Thesis Standard)
    peak_rate: int = 1000
    ramp_up: str = "5m"
    peak_rate_2: int | None = None
    ramp_up_2: str | None = None
    sustain: str = "1m"
    ramp_down: str = "1m"
    start_rate: int = 1
    warmup: str = "0s"
    max_vus: int = 200
    path_type: str = "dynamic"
    timeout: str = "0.4s"
    skip_assets: bool = False


class CapacityWrkOptions(BaseModel):
    threads: int = 2
    connections: int = 10
    duration: str = "30s"
    warmup: str = "30s"


class ExperimentConfig(BaseModel):
    test_type: str = Field(..., description="load, capacity_k6, or capacity_wrk")
    num_runs: int = 1
    inter_run_delay: str = "1m"
    load_options: LoadOptions | None = None
    capacity_k6_options: CapacityK6Options | None = None
    capacity_wrk_options: CapacityWrkOptions | None = None
