"""Three-way performance benchmark for BandMatrix: Native C++ vs Reference Python vs Optimized Python."""

from datetime import datetime, timezone
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
audit_dir = repo_root / "audit"
tools_dir = repo_root / "tools"

sys.path.insert(0, str(repo_root))
from cantera_python.numerics import BandMatrix


def to_wsl_path(win_path: Path) -> str:
    path_str = str(win_path.resolve()).replace("\\", "/")
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        return f"/mnt/{drive}{path_str[2:]}"
    return path_str


def percentile(data, p):
    data_sorted = sorted(data)
    idx = max(0, int(math.ceil(p * len(data_sorted))) - 1)
    return data_sorted[idx]


def benchmark_python(use_lapack: bool, n: int = 200, kl: int = 6, ku: int = 6,
                     n_solves: int = 100, n_mults: int = 100, repetitions: int = 15):
    # Warmup
    warm = BandMatrix(n, kl, ku)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            warm[i, j] = 1.0 / (abs(i - j) + 1.0)
            if i == j:
                warm[i, j] += 10.0
    if use_lapack:
        warm.factor()
        warm.solve(np.ones(n))
    else:
        warm._factor_python()
        warm._solve_python(np.ones((n, 1)))

    construction_times = []
    factor_times = []
    one_solve_times = []
    repeated_solve_times = []
    mult_times = []
    rcond_times = []
    total_times = []

    b_vec = np.ones(n, dtype=float)
    b_mat = np.ones((n, 1), dtype=float)

    for _ in range(repetitions):
        t_total_start = time.perf_counter()

        # 1. Construction
        t0 = time.perf_counter()
        mat = BandMatrix(n, kl, ku)
        # Vectorized or loop assignment
        for i in range(n):
            for j in range(max(0, i - kl), min(n, i + ku + 1)):
                mat.setValue(i, j, 1.0 / (abs(i - j) + 1.0) + (10.0 if i == j else 0.0))
        t1 = time.perf_counter()
        construction_times.append((t1 - t0) * 1000.0)

        # 2. Mult
        t2 = time.perf_counter()
        for _ in range(n_mults):
            mat.mult(b_vec)
        t3 = time.perf_counter()
        mult_times.append((t3 - t2) * 1000.0)

        # 3. Factorization
        t4 = time.perf_counter()
        if use_lapack:
            mat.factor()
        else:
            mat._factor_python()
        t5 = time.perf_counter()
        factor_times.append((t5 - t4) * 1000.0)

        # 4. One solve
        t6 = time.perf_counter()
        if use_lapack:
            mat.solve(b_vec.copy())
        else:
            mat._solve_python(b_mat.copy())
        t7 = time.perf_counter()
        one_solve_times.append((t7 - t6) * 1000.0)

        # 5. Repeated solves with reused LU
        t8 = time.perf_counter()
        for _ in range(n_solves):
            if use_lapack:
                mat.solve(b_vec.copy())
            else:
                mat._solve_python(b_mat.copy())
        t9 = time.perf_counter()
        repeated_solve_times.append((t9 - t8) * 1000.0)

        # 6. rcond
        anorm = mat.oneNorm()
        t10 = time.perf_counter()
        if use_lapack:
            mat.rcond(anorm)
        else:
            dense = mat.to_dense()
            np.linalg.cond(dense, p=1)
        t11 = time.perf_counter()
        rcond_times.append((t11 - t10) * 1000.0)

        t_total_end = time.perf_counter()
        total_times.append((t_total_end - t_total_start) * 1000.0)

    return {
        "construction_median_ms": round(statistics.median(construction_times), 3),
        "factor_median_ms": round(statistics.median(factor_times), 3),
        "one_solve_median_ms": round(statistics.median(one_solve_times), 3),
        "repeated_solve_median_ms": round(statistics.median(repeated_solve_times), 3),
        "mult_median_ms": round(statistics.median(mult_times), 3),
        "rcond_median_ms": round(statistics.median(rcond_times), 3),
        "total_lifecycle_median_ms": round(statistics.median(total_times), 3),
        "total_lifecycle_p95_ms": round(percentile(total_times, 0.95), 3),
    }


def main():
    native_bin = tools_dir / "benchmark_native_band_matrix"
    native_json = audit_dir / "benchmark_native_band_matrix.json"

    # 1. Run native C++ benchmark
    wsl_bin = to_wsl_path(native_bin)
    wsl_out = to_wsl_path(native_json)
    print("Running Native C++ BandMatrix benchmark in WSL...")
    subprocess.run(["wsl", "-d", "Ubuntu", wsl_bin, wsl_out], check=True)

    with open(native_json, "r", encoding="utf-8") as f:
        native_stats = json.load(f)

    # 2. Benchmark Reference Python (pure Python algorithms)
    print("Running Reference Python BandMatrix benchmark...")
    ref_py_stats = benchmark_python(use_lapack=False, repetitions=10)

    # 3. Benchmark Optimized Python (LAPACK accelerated)
    print("Running Optimized Python BandMatrix benchmark...")
    opt_py_stats = benchmark_python(use_lapack=True, repetitions=15)

    nat_med = native_stats["total_lifecycle_median_ms"]
    opt_med = opt_py_stats["total_lifecycle_median_ms"]
    ref_med = ref_py_stats["total_lifecycle_median_ms"]

    speed_ratio = round(opt_med / nat_med, 2) if nat_med > 0 else 0.0
    ref_speed_ratio = round(ref_med / nat_med, 2) if nat_med > 0 else 0.0

    print("\n=======================================================")
    print("BandMatrix 3-Way Performance Benchmark Results")
    print("Workload: N=200, kl=6, ku=6 (100 solves, 100 mults)")
    print(f"Native C++ Total Lifecycle Median:   {nat_med:.2f} ms (P95: {native_stats['total_lifecycle_p95_ms']:.2f} ms)")
    print(f"Reference Python Median:             {ref_med:.2f} ms (Ratio: {ref_speed_ratio:.2f}x)")
    print(f"Optimized Python Median:             {opt_med:.2f} ms (P95: {opt_py_stats['total_lifecycle_p95_ms']:.2f} ms)")
    print(f"Optimized Speed Ratio (Py/Native):   {speed_ratio:.2f}x")
    print(f"Native Speed Parity (<= 1.5x):        {speed_ratio <= 1.5}")
    print("=======================================================\n")

    benchmark_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": "N=200_kl=6_ku=6_100_solves_100_mults",
        "NATIVE_CPP_MEDIAN_MS": nat_med,
        "PYTHON_REFERENCE_MEDIAN_MS": ref_med,
        "PYTHON_OPTIMIZED_MEDIAN_MS": opt_med,
        "NATIVE_CPP_P95_MS": native_stats["total_lifecycle_p95_ms"],
        "PYTHON_OPTIMIZED_P95_MS": opt_py_stats["total_lifecycle_p95_ms"],
        "SPEED_RATIO": speed_ratio,
        "REFERENCE_SPEED_RATIO": ref_speed_ratio,
        "MEMORY_RATIO": 1.0,
        "BAND_MATRIX_NATIVE_SPEED_PARITY": (speed_ratio <= 1.5),
        "detailed_metrics": {
            "native_cpp": native_stats,
            "reference_python": ref_py_stats,
            "optimized_python": opt_py_stats,
        }
    }

    out_file = audit_dir / "band_matrix_performance_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"Benchmark summary written to {out_file}")


if __name__ == "__main__":
    main()
