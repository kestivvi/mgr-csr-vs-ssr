from typing import Any, TypedDict

from pydantic import BaseModel, Field


class ScenarioMetadata(TypedDict):
    name: str
    subject_server_ip: str
    load_generator_group: str
    monitoring_host_public_ip: str
    monitoring_host_private_ip: str


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

    def to_summary(self) -> dict[str, list[tuple[str, str]]]:
        return {
            "Workload": [
                ("Target RPS", str(self.rps)),
                ("Max VUs", str(self.vus)),
                ("Path Type", self.path_type),
                ("Timeout", self.timeout),
            ],
            "Timeline": [
                ("Warmup", f"{self.warmup} (0 ➔ {self.rps} RPS)"),
                ("Sustain", f"{self.duration} ({self.rps} RPS)"),
                ("Cooldown", f"{self.after} ({self.rps} ➔ 0 RPS)"),
            ],
        }


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

    def to_summary(self) -> dict[str, list[tuple[str, str]]]:
        workload = [
            ("Start Rate", str(self.start_rate)),
            ("Peak Rate", str(self.peak_rate)),
        ]
        if self.peak_rate_2:
            workload.append(("Peak Rate 2", str(self.peak_rate_2)))
        workload.extend(
            [
                ("Max VUs", str(self.max_vus)),
                ("Path Type", self.path_type),
                ("Timeout", self.timeout),
            ]
        )

        timeline = []
        if self.warmup != "0s":
            timeline.append(("Warmup", f"{self.warmup} ({self.start_rate} RPS)"))

        timeline.append(("Ramp Up", f"{self.ramp_up} ({self.start_rate} ➔ {self.peak_rate} RPS)"))
        timeline.append(("Sustain", f"{self.sustain} ({self.peak_rate} RPS)"))

        if self.ramp_up_2 and self.peak_rate_2:
            timeline.append(
                (
                    "Ramp Up 2",
                    f"{self.ramp_up_2} ({self.peak_rate} ➔ {self.peak_rate_2} RPS)",
                )
            )
            # Use same sustain duration for second peak
            timeline.append(("Sustain 2", f"{self.sustain} ({self.peak_rate_2} RPS)"))

        timeline.append(("Ramp Down", f"{self.ramp_down} ({self.peak_rate} ➔ 0 RPS)"))

        return {"Workload": workload, "Timeline": timeline}


class CapacityWrkOptions(BaseModel):
    threads: int = 2
    connections: int = 10
    duration: str = "30s"
    warmup: str = "30s"

    def to_summary(self) -> dict[str, list[tuple[str, str]]]:
        return {
            "Workload": [
                ("Threads", str(self.threads)),
                ("Connections", str(self.connections)),
            ],
            "Timeline": [
                ("Warmup", self.warmup),
                ("Sustain", self.duration),
            ],
        }


class ExperimentConfig(BaseModel):
    test_type: str = Field(..., description="load, capacity_k6, or capacity_wrk")
    num_repetitions: int = 1
    inter_repetition_delay: str = "1m"
    auto_approve: bool = False
    load_options: LoadOptions | None = None
    capacity_k6_options: CapacityK6Options | None = None
    capacity_wrk_options: CapacityWrkOptions | None = None

    def get_options(self) -> LoadOptions | CapacityK6Options | CapacityWrkOptions:
        if self.test_type == "load":
            return self.load_options or LoadOptions()
        if self.test_type == "capacity_k6":
            return self.capacity_k6_options or CapacityK6Options()
        if self.test_type == "capacity_wrk":
            return self.capacity_wrk_options or CapacityWrkOptions()
        raise ValueError(f"Unknown test type: {self.test_type}")
