from typing import List, Optional

from pydantic import BaseModel, Field


class LoadOptions(BaseModel):
    rps: int = 100
    warmup: str = "30s"
    duration: str = "1m"
    after: str = "30s"
    vus: int = 200
    path_type: str = "dynamic"  # 'static' or 'dynamic'
    timeout: str = "0.4s"


class CapacityK6Options(BaseModel):
    # Rate-based (Thesis Standard)
    peak_rate: int = 1000
    ramp_up: str = "5m"
    sustain: str = "1m"
    ramp_down: str = "1m"
    start_rate: int = 1
    warmup: str = "0s"
    max_vus: int = 200
    path_type: str = "dynamic"
    timeout: str = "0.4s"


class CapacityWrkOptions(BaseModel):
    threads: int = 2
    connections: int = 10
    duration: str = "30s"
    warmup: str = "30s"


class Scenario(BaseModel):
    name: str
    load_generator_group: str
    app_server_ip: str
    monitoring_private_ip: str
    monitoring_public_ip: Optional[str] = None


class ExperimentConfig(BaseModel):
    test_type: str = Field(..., description="load, capacity_k6, or capacity_wrk")
    num_runs: int = 1
    load_options: Optional[LoadOptions] = None
    capacity_k6_options: Optional[CapacityK6Options] = None
    capacity_wrk_options: Optional[CapacityWrkOptions] = None
    scenarios: Optional[List[Scenario]] = None
