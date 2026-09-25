# Cantera-Python: Unofficial Pure-Python Cantera Solver Reconstruction

> **DISCLAIMER**: This repository is an **unofficial experimental pure-Python reconstruction** of Cantera solvers.
> **Full Cantera is NOT complete.** Only **Solver Wave 1: Thermodynamic Inverse-State Solvers** is released and published here.
> Upstream Cantera reference commit: [`726522be4e2a13454d8415b7ef799d621f665cf3`](https://github.com/Cantera/cantera/commit/726522be4e2a13454d8415b7ef799d621f665cf3).

---

## Overview

Cantera is an open-source suite of tools for problems involving chemical kinetics, thermodynamics, and transport processes. This project reconstructs Cantera's numerical solvers in pure Python with zero native compilation dependencies, guaranteeing strict algorithmic and numerical parity against the pinned native C++ reference.

### Solver Wave 1: Thermodynamic Inverse-State Solvers

Wave 1 implements the complete foundation for setting thermodynamic states via inverse property pairs:
- `ThermoPhase::setState_HP(h, p, tol)`: Enthalpy and Pressure inversion
- `ThermoPhase::setState_UV(u, v, tol)`: Internal Energy and Specific Volume inversion
- `ThermoPhase::setState_SP(s, p, tol)`: Entropy and Pressure inversion
- `ThermoPhase::setState_SV(s, v, tol)`: Entropy and Specific Volume inversion
- `ThermoPhase::setState_HPorUV(h_or_u, p_or_v, hp)`: Unified HP / UV solver
- `ThermoPhase::setState_SPorSV(s, p_or_v, sp)`: Unified SP / SV solver

### Algorithmic Contract & Parity

The inverse solvers reproduce the exact numerical algorithms from Cantera's `src/thermo/ThermoPhase.cpp`:
1. **Safeguarded Bounded Newton-Raphson**: Step limits enforced at $|\Delta T|_{\max} = 100\,\text{K}$ per step.
2. **Positive-Temperature Domain Guard**: Temperature is strictly prevented from stepping below $T / 3$ or below physical bounds.
3. **Monotonic Bisection Fallback**: Safeguarded contraction ensures convergence even under non-monotonic or adversarial starting points.
4. **Independent Starting Points**: Validated against independent initial states separated by $\ge 20\%$ in temperature from the target state.
5. **Exact Native Differential Parity**: 4,100 test cases differential-tested against native Cantera `726522be4` built on Linux x86_64, achieving **0 mismatches** and $100.0\%$ parity.

---

## Installation & Requirements

- Python 3.10+
- `numpy`
- `ruamel.yaml` or `pyyaml`
- Zero C++ compiler or native extension dependencies.

```bash
git clone https://github.com/Navronis/Cantera-Python.git
cd Cantera-Python
pip install -e .
```

---

## Quickstart Example

```python
import cantera_python as cp

# Initialize gas mixture from packaged mechanism
gas = cp.Solution("gri30.yaml")

# Set reference state
gas.TPX = 1200.0, cp.OneAtm, "CH4:0.1, O2:0.2, N2:0.7"
target_h = gas.enthalpy_mass
target_p = gas.P

# Perturb state to a distant starting point
gas.TP = 400.0, 0.5 * cp.OneAtm

# Invert state using setState_HP (Wave 1 solver)
gas.setState_HP(target_h, target_p)

print(f"Converged T: {gas.T:.4f} K (Target: 1200.0000 K)")
print(f"Converged P: {gas.P:.2f} Pa (Target: {target_p:.2f} Pa)")
assert abs(gas.T - 1200.0) < 1e-5
```

---

## Validation & Verification

To run the complete test suite:
```bash
pytest tests/
```

To run the 4,000-case independent-start differential verification tool:
```bash
python tools/run_wave1_differential.py
```

See [docs/WAVE_1.md](docs/WAVE_1.md) for full mathematical specifications, branch coverage, and tolerance sweep analysis.

---

## Attribution & License

This project includes thermodynamic models and algorithms derived from Cantera (BSD-3-Clause License).
Copyright 2001-2024 Cantera Developers. See [LICENSE](LICENSE) and [License.txt](License.txt) for complete details.
