import json
import re
from typing import Any, Dict, Optional, cast


def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def extract_marker(output: str, marker: str) -> Optional[Dict[str, Any]]:
    for line in output.splitlines():
        clean_line = strip_ansi(line)
        if marker in clean_line:
            try:
                # Use a more flexible regex to find the JSON block { ... }
                match = re.search(f"{re.escape(marker)}.*?({{.*}})", clean_line)
                if match:
                    json_str = match.group(1).replace('\\"', '"')
                    return cast(Dict[str, Any], json.loads(json_str))
            except Exception as e:
                print(f"Extraction error for {marker}: {e}")
                continue
    return None


# Test case 1: Ansible debug output with quotes and ANSI codes
test_output = """
 \x1b[0;32mok: [mgr-load-generator-CSR-Vanilla] => { \x1b[0m
 \x1b[0;32m    "msg": "ORCHESTRATOR_TIMESTAMPS::{\\"start\\": \\"1777234413\\", \\"end\\": \\"1777234444\\"}" \x1b[0m
 \x1b[0;32m} \x1b[0m
"""

marker = "ORCHESTRATOR_TIMESTAMPS::"
result = extract_marker(test_output, marker)
print(f"Result: {result}")
assert result == {"start": "1777234413", "end": "1777234444"}

# Test case 2: WRK results
wrk_output = """
 \x1b[0;32m    "msg": "WRK_RESULTS::{\\"rps\\": 29276.29, \\"latency_avg\\": \\"5.05ms\\", \\"transfer_per_sec\\": \\"15.16MB\\"}" \x1b[0m
"""
marker2 = "WRK_RESULTS::"
result2 = extract_marker(wrk_output, marker2)
print(f"Result 2: {result2}")
assert result2["rps"] == 29276.29
