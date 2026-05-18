from pathlib import Path
from typing import Any

import pytest

from orchestrator.shared.verifier import (
    CSR_PROFILE,
    SSR_PROFILE,
    HealthCheck,
    HealthProfile,
    SubjectVerifier,
)


def _ok_headers(status: str = "HTTP/1.1 200 OK") -> str:
    return f"{status}\r\ncontent-type: text/html\r\ncontent-encoding: gzip\r\n\r\n"


def _ssr_dynamic_app_body(id_: int = 42, row_count: int = 100) -> str:
    rows = "".join([f'<div class="row">r{i}</div>' for i in range(row_count)])
    return (
        f'<html><body><div class="dynamic-app"><h1>Items for #{id_}</h1>'
        f'<p class="summary">100 items</p>{rows}</div></body></html>'
    )


class _FakeVerifier(SubjectVerifier):
    """Verifier with curl mocked by a URL→output mapping."""

    def __init__(self, responses: dict[str, str]) -> None:
        super().__init__(workdir=Path("/tmp"))
        self._responses = responses
        self.calls: list[str] = []

    def _run(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        verbose: bool = False,
        error_type: Any = None,
    ) -> str:
        url = command[-1]
        self.calls.append(url)
        for needle, response in self._responses.items():
            if needle in url:
                return response
        return "HTTP/1.1 404 Not Found\r\n\r\n"


def test_ssr_profile_contains_dynamic_app_check() -> None:
    paths = [c.path for c in SSR_PROFILE.checks]
    assert "/dynamic-app/42" in paths


def test_csr_profile_contains_dynamic_app_check() -> None:
    paths = [c.path for c in CSR_PROFILE.checks]
    assert "/dynamic-app/42" in paths


def test_ssr_dynamic_app_passes_when_markers_present() -> None:
    body = _ssr_dynamic_app_body(42, 100)
    verifier = _FakeVerifier(
        {"/dynamic-app/42": _ok_headers() + body, "": _ok_headers() + "hello world"}
    )
    profile = HealthProfile(
        name="SSR",
        checks=[c for c in SSR_PROFILE.checks if c.path == "/dynamic-app/42"],
    )
    assert verifier.wait_until_healthy("http://localhost:80", profile, retries=1) is True


def test_ssr_dynamic_app_fails_when_h1_marker_missing() -> None:
    rows = "".join(['<div class="row">x</div>' for _ in range(100)])
    body = f"<html>{rows}</html>"  # missing "Items for #42"
    verifier = _FakeVerifier({"/dynamic-app/42": _ok_headers() + body})
    profile = HealthProfile(
        name="SSR",
        checks=[c for c in SSR_PROFILE.checks if c.path == "/dynamic-app/42"],
    )
    assert verifier.wait_until_healthy("http://localhost:80", profile, retries=1) is False


def test_ssr_dynamic_app_fails_when_row_count_wrong() -> None:
    body = _ssr_dynamic_app_body(42, 99)  # only 99 rows
    verifier = _FakeVerifier({"/dynamic-app/42": _ok_headers() + body})
    profile = HealthProfile(
        name="SSR",
        checks=[c for c in SSR_PROFILE.checks if c.path == "/dynamic-app/42"],
    )
    assert verifier.wait_until_healthy("http://localhost:80", profile, retries=1) is False


def test_csr_dynamic_app_passes_with_shell_only() -> None:
    # CSR shell has no rows and no H1 marker — should still pass (only 200 asserted).
    body = "<html><body><div id='root'></div><script src='/app.js'></script></body></html>"
    verifier = _FakeVerifier({"/dynamic-app/42": _ok_headers() + body})
    profile = HealthProfile(
        name="CSR",
        checks=[c for c in CSR_PROFILE.checks if c.path == "/dynamic-app/42"],
    )
    assert verifier.wait_until_healthy("http://localhost:80", profile, retries=1) is True


def test_dynamic_app_fails_on_404() -> None:
    verifier = _FakeVerifier({})  # everything 404
    profile = HealthProfile(
        name="SSR",
        checks=[c for c in SSR_PROFILE.checks if c.path == "/dynamic-app/42"],
    )
    logs: list[str] = []
    verifier.on_output = logs.append
    assert verifier.wait_until_healthy("http://localhost:80", profile, retries=1) is False


def test_health_check_body_contains_field_present() -> None:
    check = HealthCheck("/x", body_contains=["foo"])
    assert check.body_contains == ["foo"]


def test_health_check_body_count_field_present() -> None:
    check = HealthCheck("/x", body_count={"<row>": 100})
    assert check.body_count == {"<row>": 100}


@pytest.fixture
def _unused_tmp(tmp_path: Path) -> Path:
    return tmp_path
