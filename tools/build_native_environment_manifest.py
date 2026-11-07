"""Record what is known, and explicitly unknown, about the pinned native oracle."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT.parent / "cantera-original"
OUTPUT = ROOT / "audit" / "native_environment_manifest.json"


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def package_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def main():
    expected = "726522be4e2a13454d8415b7ef799d621f665cf3"
    head = git_head(NATIVE)
    if head != expected:
        raise RuntimeError(f"native checkout is {head}, expected {expected}")
    sundials = git_head(NATIVE / "ext" / "sundials")
    parity_binding = None
    try:
        import sksundae
        from sksundae._cy_common import config as sundials_config
        parity_binding = {
            "package": "scikit-sundae", "version": sksundae.__version__,
            "sundials_version": sksundae.SUNDIALS_VERSION,
            "configuration": dict(sundials_config),
            "role": "standalone numerical dependency; does not load native Cantera",
        }
    except ImportError:
        pass
    tools = {name: shutil.which(name) for name in
             ("cl", "gcc", "clang", "cmake", "ninja", "scons", "msbuild", "docker", "podman", "wsl")}
    manifest = {
        "schema_version": 1,
        "reference_kind": "archived exact-pinned fixture plus source checkout",
        "cantera_commit": head,
        "sundials": {"version": "7.5.0", "commit": sundials,
                     "source": "ext/sundials/CMakeLists.txt and git HEAD"},
        "archived_fixture": {
            "path": "provenance/pinned_native_differential_results.jsonl",
            "compiler": None, "compiler_version": None, "optimization_flags": None,
            "floating_point_flags": None, "architecture": None,
            "blas_backend": None, "lapack_backend": None,
            "sundials_build_options": None, "linear_solver": None,
            "nonlinear_solver": None, "index_precision": None,
            "float_precision": "binary64 outputs", "openmp_threads": None,
            "limitation": "The archived fixture did not preserve its native build manifest; unknown fields are not inferred.",
        },
        "current_host": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": sys.version, "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "parity_binding": parity_binding,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "tool_paths": tools,
        },
        "fresh_exact_pinned_build": {
            "available": False,
            "reason": "No native Windows compiler/build toolchain is available and WSL launch previously failed with E_ACCESSDENIED.",
            "reproducible_configuration_frozen": False,
        },
        "released_cantera_is_authoritative": False,
        "notes": [
            "Released Cantera 3.2.0 comparisons are supplementary and are not exact-pinned proof.",
            "Solver parity cannot be certified until an exact-pinned executable is rebuilt with all native configuration fields captured.",
        ],
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "cantera_commit": head,
                      "sundials_commit": sundials, "fresh_build": False}))


if __name__ == "__main__":
    main()
