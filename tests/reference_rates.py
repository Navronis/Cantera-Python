"""Optional native-reference validation; production code has no native dependency.

Run with a separate interpreter where Cantera is installed::

    python tests/reference_rates.py

Prints a deterministic JSON report and exits nonzero on a comparison failure.
Cantera 3.2.0 is an older release than the pinned 4.0-development source;
this verifies only formulas common to the two versions. The newer plasma
bg/Tinv terms are checked by independent tests/test_rates.py identities.
"""

import json
import math
from pathlib import Path
import random
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cantera as native
from cantera_python import rates as port


SEED = 726522
SAMPLES_PER_FAMILY = 500
RELATIVE_TOLERANCE = 2.0e-11


def run():
    rng = random.Random(SEED)
    checks = {}
    failures = []

    def compare(family, actual, expected, context):
        row = checks.setdefault(family, {"checks": 0, "max_relative_error": 0.0})
        row["checks"] += 1
        denominator = max(abs(actual), abs(expected), 1e-300)
        error = abs(actual - expected) / denominator
        row["max_relative_error"] = max(row["max_relative_error"], error)
        if not math.isclose(actual, expected, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-300):
            failures.append({"family": family, "actual": actual, "expected": expected,
                             "relative_error": error, "context": context})

    def arrhenius_pair():
        args = (10.0**rng.uniform(-3, 20), rng.uniform(-2, 2), rng.uniform(-1e7, 1e8))
        return port.ArrheniusRate(*args), native.ArrheniusRate(*args)

    for _ in range(SAMPLES_PER_FAMILY):
        T = rng.uniform(200.0, 6000.0)
        py, cpp = arrhenius_pair()
        compare("Arrhenius", py(T), cpp(T), {"T": T})

        args = (10.0**rng.uniform(-3, 15), rng.uniform(-2, 2),
                rng.uniform(-1e7, 1e8), rng.uniform(-1e7, 1e8))
        Te = rng.uniform(200.0, 30000.0)
        py, cpp = port.TwoTempPlasmaRate(*args), native.TwoTempPlasmaRate(*args)
        compare("TwoTempPlasma_common_4_parameters", py(T, Te), cpp(T, Te), {"T": T, "Te": Te})

        E0 = 10.0**rng.uniform(5, 8)
        args = (10.0**rng.uniform(-3, 15), rng.uniform(-2, 2), E0,
                E0 * rng.uniform(2.1, 1000.0))
        dh = E0 * rng.uniform(-6.0, 6.0)
        py, cpp = port.BlowersMaselRate(*args), native.BlowersMaselRate(*args)
        cpp.delta_enthalpy = dh
        compare("BlowersMasel_barrier", py.activation_energy(dh), cpp.activation_energy,
                {"T": T, "delta_enthalpy": dh})
        compare("BlowersMasel_rate", py(T, dh), cpp(T), {"T": T, "delta_enthalpy": dh})

    # Exercise exact branch points separately from continuous random samples.
    for dh in (-5e8, -4e8, 0.0, 4e8, 5e8):
        py = port.BlowersMaselRate(3.0, 0.0, 1e8, 1e9)
        cpp = native.BlowersMaselRate(3.0, 0.0, 1e8, 1e9)
        cpp.delta_enthalpy = dh
        compare("BlowersMasel_boundary", py.activation_energy(dh), cpp.activation_energy,
                {"delta_enthalpy": dh})

    falloff_variants = [
        ("Lindemann", port.LindemannRate, native.LindemannRate, ()),
        ("Troe3", port.TroeRate, native.TroeRate, (0.7346, 94.0, 1756.0)),
        ("Troe4", port.TroeRate, native.TroeRate, (0.7346, 94.0, 1756.0, 5182.0)),
        ("Troe_zero_T1_T3", port.TroeRate, native.TroeRate, (0.5, 0.0, 0.0, 900.0)),
        ("SRI3", port.SriRate, native.SriRate, (1.1, 700.0, 1234.0)),
        ("SRI5", port.SriRate, native.SriRate, (1.1, 700.0, 1234.0, 0.9, 0.2)),
        ("SRI_zero_c", port.SriRate, native.SriRate, (1.1, 700.0, 0.0)),
        ("Tsang1", port.TsangRate, native.TsangRate, (0.7,)),
        ("Tsang2", port.TsangRate, native.TsangRate, (0.95, -0.0001)),
    ]
    for name, py_type, cpp_type, coeffs in falloff_variants:
        for _ in range(SAMPLES_PER_FAMILY):
            T = rng.uniform(200.0, 6000.0)
            M = 10.0**rng.uniform(-12.0, 5.0)
            py_low, cpp_low = arrhenius_pair()
            py_high, cpp_high = arrhenius_pair()
            py = (py_type(py_low, py_high, coeffs) if coeffs
                  else py_type(py_low, py_high))
            cpp = cpp_type(low=cpp_low, high=cpp_high, falloff_coeffs=coeffs)
            compare(name + "_falloff", py(T, M), cpp(T, M), {"T": T, "M": M})
            # Native falloff_function takes M; the port's corresponding API
            # is falloff_coefficient. Port falloff_function itself takes Pr.
            compare(name + "_F", py.falloff_coefficient(T, M), cpp.falloff_function(T, M),
                    {"T": T, "M": M})
            py.chemically_activated = True
            cpp.chemically_activated = True
            compare(name + "_activated", py(T, M), cpp(T, M), {"T": T, "M": M})

    for _ in range(SAMPLES_PER_FAMILY):
        T, P = rng.uniform(200.0, 6000.0), 10.0**rng.uniform(2.0, 9.0)
        py_rates, cpp_rates = [], []
        for reference_P in (1e4, 1e5, 1e5, 1e6, 1e7):
            py_rate, cpp_rate = arrhenius_pair()
            py_rates.append((reference_P, py_rate))
            cpp_rates.append((reference_P, cpp_rate))
        py, cpp = port.PlogRate(py_rates), native.PlogRate(cpp_rates)
        compare("Plog", py(T, P), cpp(T, P), {"T": T, "P": P})

        nT, nP = rng.randrange(1, 8), rng.randrange(1, 7)
        coeffs = [[rng.uniform(-0.5, 0.5) for _ in range(nP)] for _ in range(nT)]
        coeffs[0][0] += 8.0
        py = port.ChebyshevRate(200.0, 6000.0, 1e2, 1e9, coeffs)
        cpp = native.ChebyshevRate(temperature_range=(200.0, 6000.0),
                                  pressure_range=(1e2, 1e9), data=coeffs)
        compare("Chebyshev", py(T, P), cpp(T, P), {"T": T, "P": P, "nT": nT, "nP": nP})

    report = {
        "status": "passed" if not failures else "failed",
        "reference_version": native.__version__,
        "reference_is_pinned_source_build": False,
        "pinned_source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "seed": SEED,
        "samples_per_family": SAMPLES_PER_FAMILY,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "total_checks": sum(row["checks"] for row in checks.values()),
        "families": checks,
        "failure_count": len(failures),
        "first_failures": failures[:20],
        "limitations": [
            "Native reference is a released Cantera version, not a build of the pinned development commit.",
            "Two-temperature plasma native comparison covers A,b,Ea,EE only; newer bg,Tinv terms have independent identity and derivative tests.",
            "Randomized comparisons cover sampled parameter/state ranges, not all mechanisms, all floating-point inputs, or complete equation coverage.",
            "Derivative comparisons are independent finite differences in test_rates.py, not native derivative API checks."
        ],
    }
    return report


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="BlowersMaselData::update:.*", category=UserWarning)
        result = run()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)
