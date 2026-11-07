"""Performance benchmark comparing Python FlowReactor against Native C++ FlowReactor."""

import json
import statistics
import subprocess
import time
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
audit_dir = repo_root / "audit"
tools_dir = repo_root / "tools"

def to_wsl_path(win_path: Path) -> str:
    path_str = str(win_path.resolve()).replace("\\", "/")
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        return f"/mnt/{drive}{path_str[2:]}"
    return path_str

def main():
    corpus_path = audit_dir / "corpus_flow_reactor.jsonl"
    native_script = tools_dir / "run_native_flow_reactor_oracle.py"
    native_out = audit_dir / "perf_flow_native_out.jsonl"
    python_out = audit_dir / "perf_flow_python_out.jsonl"

    wsl_oracle_python = "/home/nebula/cantera-oracle-env/bin/python3"
    wsl_corpus = to_wsl_path(corpus_path)
    wsl_native_script = to_wsl_path(native_script)
    wsl_native_out = to_wsl_path(native_out)

    # 1. Benchmark Native C++ Oracle
    native_cmd = [
        "wsl", "-d", "Ubuntu",
        wsl_oracle_python,
        wsl_native_script,
        wsl_corpus,
        wsl_native_out,
        "perf_flow_native",
    ]
    # Warmup
    subprocess.run(native_cmd, capture_output=True, check=True)

    native_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        subprocess.run(native_cmd, capture_output=True, check=True)
        t1 = time.perf_counter()
        native_times.append((t1 - t0) * 1000.0)

    native_median = statistics.median(native_times)

    # 2. Benchmark Python Implementation
    sys.path.insert(0, str(tools_dir))
    from run_python_flow_reactor_port import run_cases
    # Warmup
    run_cases(corpus_path, python_out, "perf_flow_py")

    py_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run_cases(corpus_path, python_out, "perf_flow_py")
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * 1000.0)

    py_median = statistics.median(py_times)
    ratio = py_median / native_median if native_median > 0 else 0.0

    print(f"Native C++ FlowReactor Median: {native_median:.2f} ms")
    print(f"Python FlowReactor Median: {py_median:.2f} ms")
    print(f"Speed Ratio (Py/Native): {ratio:.2f}x")

    bench_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": "16_flow_reactor_dae_cases_including_high_pressure_and_stoichiometry_sweeps",
        "native_median_ms": round(native_median, 3),
        "python_median_ms": round(py_median, 3),
        "speed_ratio": round(ratio, 2),
        "native_speed_parity": ratio < 5.0,
        "native_runs_ms": [round(x, 3) for x in native_times],
        "python_runs_ms": [round(x, 3) for x in py_times],
    }

    out_path = audit_dir / "flow_reactor_performance_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=2)

    print(f"Performance benchmark written to {out_path}")


if __name__ == "__main__":
    main()
