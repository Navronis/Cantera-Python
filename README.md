# Cantera-Python: Pure-Python Algorithmic Reconstruction of Cantera

[![CI](https://github.com/Navronis/Cantera-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/Navronis/Cantera-Python/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Canonical%20Closure-47.91%25-orange.svg)](PORT_STATUS.md)

> [!IMPORTANT]
> **GOVERNING ARCHITECTURAL SPECIFICATION**:
> This repository is a pure-Python, zero-compiler-dependency algorithmic reconstruction of [Cantera](https://github.com/Cantera/cantera), verified against pinned native Cantera commit [`726522be4e2a13454d8415b7ef799d621f665cf3`](https://github.com/Cantera/cantera/commit/726522be4e2a13454d8415b7ef799d621f665cf3).
>
> All tracking strictly adheres to an immutable denominator of **5,784** canonical C++ source symbols.

---

## Current Certified Baseline

| Metric | Certified Count | Percentage |
| :--- | :---: | :---: |
| **Immutable Canonical Denominator** | **5,784** | **100.00%** |
| **Fully Dispositioned Symbols** | **2,771** | **47.91%** |
| ↳ Pinned-Native Validated (`PINNED_NATIVE_VALIDATED`) | 2,012 | 34.79% |
| ↳ Proven Non-Runtime C++-Only (`NON_RUNTIME_CPP_ONLY_PROVEN`) | 759 | 13.12% |
| **Unresolved Symbols** | **3,013** | **52.09%** |
| ↳ `NOT_PORTED` | 2,630 | 45.47% |
| ↳ `PARTIAL` | 383 | 6.62% |

### Subsystem Completion Overview

```
Transport Subsystem        [████████████████████████████████████████] 100.00% (229/229)
Kinetics Subsystem         [████████████████████████████████████████] 100.00% (749/749)
Thermodynamics Subsystem   [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░]  34.50% (573/1,661)
1D Flow & Flame Solvers    [████████████████████████████████████████] 100.00% Verified
0D Reactor Networks        [████████████████████████████████████████] 100.00% Verified
VCS Multiphase Equil       [████████████████████████████████████████] 100.00% Verified
```

- **Transport (100% Closed)**: Mixture-averaged, Multicomponent (Stefan–Maxwell system inversion), Dusty Gas porous transport, High-Pressure dense gas corrections, and Ion Gas Transport (including mobility/diffusion differential precision to $10^{-13}$).
- **Kinetics (100% Closed)**: All 5 kinetics closure families validated: reaction rates, Arrhenius, P-log, Chebyshev, third-body efficiencies, falloff parameterizations (Lindemann, Troe, SRI), Blowers-Masel, and surface/interface reactions.
- **Thermodynamics (Families 1–3 Closed)**:
  - **Family 1 (`Phase` Core)**: 119/119 symbols. Species, elements, state coordinates, composition mappings.
  - **Family 2 (`SpeciesThermo`)**: 137/137 symbols. All 8 polynomial models (`NasaPoly1`, `NasaPoly2`, `Nasa9Poly1`, `Nasa9PolyMultiTempRegion`, `ShomatePoly`, `ShomatePoly2`, `ConstCpPoly`, `Mu0Poly`) validated to $0.00\text{e}+00$ relative error against native Cantera, plus $H_{f,298}$ modification/reset.
  - **Family 3 (`ThermoPhase` Base)**: 201/201 symbols. State-pair setters (`setState_TP`, `setState_TD`, `setState_DP`, `setState_HP`, `setState_UV`, `setState_SP`, `setState_SV`), partial molar properties, chemical potentials, activities, sound speed, and temperature limit enforcement validated across 90 differential test cases to machine precision ($\le 9.09 \times 10^{-15}$).
- **1D Reacting Flow & Flame Solvers**: `Domain1D`, `Boundary1D`, `Flow1D`, `Sim1D`, `IonFlow`, adaptive grid `Refiner`, damped `MultiNewton` solver, block `BandMatrix` linear solvers, and Sundials-equivalent CVODES/IDAS integrators.
- **0D Reactor Networks**: Homogeneous batch, constant volume, surface, and flow reactors coupled into multi-reactor networks with energy equations and mass conservation.
- **VCS Multiphase Chemical Equilibrium**: Complete non-ideal Gibbs energy minimization solver.

---

## Installation

Cantera-Python requires **Python 3.10+** and pure-Python numerical libraries. It requires **no C/C++ compiler or native build toolchain**.

```bash
git clone https://github.com/Navronis/Cantera-Python.git
cd Cantera-Python
pip install -e .
```

Dependencies:
- `numpy`
- `scipy`
- `ruamel.yaml` or `pyyaml`
- `pytest` (optional, for running tests)

---

## Quickstart Examples

### 1. Thermodynamic State Inversion (`setState_HP`)

```python
import cantera_python as cp

# Load GRI-30 mechanism
gas = cp.Solution("gri30.yaml")

# Set reference high-temperature state
gas.TPX = 1500.0, cp.OneAtm, "CH4:1, O2:2, N2:7.52"
target_enthalpy = gas.enthalpy_mass
target_pressure = gas.P

# Perturb to a cold starting state
gas.TP = 300.0, 0.5 * cp.OneAtm

# Perform bounded Newton state inversion matching native Cantera's algorithm
gas.setState_HP(target_enthalpy, target_pressure)

print(f"Recovered Temperature: {gas.T:.4f} K (Target: 1500.0000 K)")
print(f"Density: {gas.density:.4f} kg/m^3")
```

### 2. Chemical Kinetics & Reaction Rates

```python
import cantera_python as cp

gas = cp.Solution("gri30.yaml")
gas.TPX = 1200.0, cp.OneAtm, "CH4:1, O2:2, N2:7.52"

# Evaluate net production rates of species (kmol/m^3/s)
wdot = gas.net_production_rates
print(f"CH4 consumption rate: {wdot[gas.species_index('CH4')]:.4e} kmol/m^3/s")
print(f"CO2 production rate:  {wdot[gas.species_index('CO2')]:.4e} kmol/m^3/s")
```

### 3. 0D Reactor Network Ignition Delay

```python
import cantera_python as cp

gas = cp.Solution("gri30.yaml")
gas.TPX = 1000.0, cp.OneAtm, "CH4:1, O2:2, N2:7.52"

# Instantiate ideal-gas constant-volume reactor and network
r = cp.IdealGasReactor(gas)
sim = cp.ReactorNet([r])

# Advance simulation in time
time = 0.0
while time < 0.1:
    time = sim.step()
    if r.T > 1500.0:
        print(f"Ignition reached at t = {time * 1e3:.2f} ms, T = {r.T:.1f} K")
        break
```

---

## Verification & Testing

The test suite validates algorithmic behavior, numerical accuracy, and exception handling against native Cantera oracles:

```bash
# Run the entire test suite
python -m pytest tests/ -q
```

**Test Suite Health**:
- **776 passed**
- **3 skipped**
- **0 failures, 0 errors**
- **698 subtests passed**

### Key Differential Audits & Frozen Manifests

- **Canonical Symbol Inventory**: [`provenance/canonical_5784_symbol_inventory.jsonl`](provenance/canonical_5784_symbol_inventory.jsonl)
- **Symbol Closure Dashboard**: [`provenance/symbol_closure_dashboard.json`](provenance/symbol_closure_dashboard.json)
- **Port Status Overview**: [`PORT_STATUS.md`](PORT_STATUS.md) and [`PORT_STATUS.json`](PORT_STATUS.json)
- **Kinetics Closure Manifests**: `audit/kinetics_family1_frozen_manifest.json` through `family5`
- **Transport Verification**: `audit/mixture_transport_differential_summary.json`, `audit/multicomponent_transport_differential_summary.json`
- **Thermodynamics Closure Manifests**: `audit/thermo_family1_frozen_manifest.json` through `family3`
- **Thermodynamics Differential Validation**: `audit/thermo_family3_differential_summary.json` (90/90 passing cases, max error $< 10^{-14}$)

---

## Repository Structure

```
Cantera-Python/
├── cantera_python/             # Pure Python implementation
│   ├── ideal_gas.py            # IdealGasPhase and mixtures
│   ├── phases.py               # Phase, ThermoPhase, and specialized phases
│   ├── thermo.py               # SpeciesThermo polynomials (NASA, Shomate, ConstCp, Mu0)
│   ├── kinetics.py             # GasKinetics, ReactionPath, InterfaceKinetics
│   ├── rates.py                # Reaction rate evaluators (Arrhenius, P-log, Chebyshev)
│   ├── transport.py            # Mixture, Multicomponent, DustyGas, HighPressure transport
│   ├── oned.py                 # 1D flow domains, flame solvers, boundaries
│   ├── reactor.py              # 0D reactors and reactor networks
│   ├── vcs.py                  # VCS non-ideal multiphase chemical equilibrium
│   ├── numerics.py             # Damped Newton, BandMatrix, CVODES/IDAS integrators
│   └── solution.py             # Unified Solution interface
├── tests/                      # 776+ comprehensive unit and differential tests
├── provenance/                 # Canonical 5,784-symbol inventory and dashboard
├── audit/                      # Numerical differential summaries and frozen manifests
└── tools/                      # Benchmark, oracle generation, and verification scripts
```

---

## Attribution & License

Cantera-Python contains thermodynamic models, transport formulations, and kinetics algorithms derived from [Cantera](https://cantera.org) under the **BSD-3-Clause License**.
Copyright 2001–2026 Cantera Developers. See [`LICENSE`](LICENSE) for complete terms.
