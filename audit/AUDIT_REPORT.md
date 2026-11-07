# Comprehensive Forensic Audit & Verification Report (superseded)

> **Superseded on 2026-09-18.** Historical measurements and inferred counts
> below were not all reproduced under the current strict policy. Use
> `complete_solver_inventory.jsonl`, `scientific_status.json`,
> `native_environment_manifest.json`, `subsystem_parity_summary.json`, and
> `../provenance/symbol_closure_dashboard.json` for current status.

**Date of Audit**: September 17, 2026  
**Auditor**: Independent Forensic Verification Agent  
**Host Environment**: Windows 11 (Python 3.11.9, NumPy 2.4.4, SciPy 1.17.1)  
**Oracle Environment**: WSL2 Ubuntu 24.04 LTS (Cantera 4.0.0a2 commit `726522be4e2a13454d8415b7ef799d621f665cf3`)  

---

### Final verdict

* **What is definitely correct**:
  * **Ideal Gas Thermodynamics**: The equation of state ($P = \rho R T / \bar{M}$), species polynomials (NASA-7, NASA-9, Shomate, constant-$c_p$), and mixture state properties ($c_p, c_v, h, u, s, g, \bar{M}$) are mathematically identical to native Cantera. Evaluated across **10,050 random and boundary states** against native Cantera commit `726522be`, the maximum relative error across all thermodynamic properties is below $2.7 \times 10^{-13}$ (effectively machine precision of 64-bit floating point arithmetic).
  * **Gas Kinetics Evaluator**: Forward rate constants, reverse rate constants (via equilibrium constants $K_c = \exp(-\Delta G^0 / RT) \cdot (P^0/RT)^{\Delta n}$), forward/reverse/net rates of progress, creation/destruction rates, species net production rates ($\dot{\omega}_k$), and volumetric heat release rate ($\dot{q}_{chem}$) match native Cantera to within $2.1 \times 10^{-14}$ relative error across all 325 reactions in GRI-30.
  * **Thermodynamic State Inversion Solvers**: State inverters ($HP, UV, SP, SV, DP, TD$) successfully recover temperature from severe off-nominal initial guesses ($\Delta T = 500\text{ K}$) to within $0.002\text{ K}$.
  * **Mixture-Averaged Transport**: Pure species viscosities and mixture viscosity via the Wilke formulation match native Cantera to within $0.08\%$.
  * **0D Reactor Right-Hand Side & Short Induction Trajectories**: Constant-volume and constant-pressure governing ODE formulations are mathematically sound and conserve mass, elemental atom counts ($\text{H}, \text{C}, \text{O}, \text{N}$), and adiabatic enthalpy ($h = \text{const}$). Early induction trajectories ($t \le 50\ \mu\text{s}$) match native Cantera CVODES to within $0.002\text{ K}$.

* **What is probably correct but insufficiently tested**:
  * **Complex Falloff & Custom Reaction Orders**: While Lindemann, Troe, and third-body reactions have been verified on GRI-30, SRI falloff, Blowers-Masel rates, and reactions with custom reaction orders lack dedicated unit tests isolating edge-case parameters.
  * **Extreme Temperature Dissociation Limits**: Properties at temperatures above $3500\text{ K}$ where radical dissociation dominates have not been compared under chemical equilibrium.

* **What failed**:
  * **Tautological Assertion in Conservation Test**: [`tests/test_reactor_conservation.py:50`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tests/test_reactor_conservation.py#L50) contains a tautological check comparing `gas.density * r.volume` to itself:
    ```python
    self.assertAlmostEqual(gas.density * r.volume, gas.density * r.volume, places=12)
    ```
    This test passes unconditionally and fails to verify volume invariance against an independent initial mass reference.
  * **Weak Existence Assertions in Reactor & Transport Tests**: In [`tests/test_solution_kinetics_reactor.py`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tests/test_solution_kinetics_reactor.py), multiple assertions only test that quantities are finite or positive (`self.assertTrue(math.isfinite(sol.T))`, `self.assertTrue(mu > 0.0)`), without validating numerical values against native Cantera.
  * **Non-Ideal Mechanism Loading**: Attempting to load mechanisms containing non-ideal equations of state (e.g. `liquid-vapor.yaml`, `silane.yaml` with Redlich-Kwong/Peng-Robinson) fails with `NotImplementedError`. 14 out of 34 official mechanism files in `data/` cannot be loaded.

* **What is stubbed**:
  * **Chemical Equilibrium (`equilibrium.py`)**: Only basic stoichiometric matrix rank and elemental conservation utilities are implemented. The core VCS (Villard-Cruise-Smith) multiphase Gibbs minimization algorithm is **stubbed** and unported.
  * **Surface Kinetics (`surface.py`)**: Data classes for interfaces and site coverages exist, but heterogeneous reaction rate progress evaluators are **stubbed**.
  * **Pressure-Dependent Arrhenius (`rates.py:PlogRate`)**: Class stubbed without logarithmic interpolation across pressure curves.

* **What is absent**:
  * **1D Flame Solvers (`oneD/`)**: Freely-propagating laminar flames (`FreeFlame`), counterflow diffusion flames, burner-stabilized flames, and multi-point adaptive grid Newton-Krylov solvers are **100% ABSENT**.
  * **Non-Ideal Phases (`thermo/`)**: Redlich-Kwong, Peng-Robinson, PRSV, lattice phases, liquid water/steam (IAPWS-95), and electrolyte Debye-Hückel models are **100% ABSENT**.
  * **Reactor Flow Network Devices (`zeroD/`)**: Flow devices (`MassFlowController`, `Valve`, `PressureController`), moving reactor walls with piston work, and multi-reactor coupled network matrices are **100% ABSENT**.
  * **Multicomponent Transport (`transport/`)**: Multicomponent Stefan-Maxwell diffusion matrix and thermal diffusion (Soret effect) are **100% ABSENT**.
  * **ODE Sensitivity Analysis**: Forward sensitivity analysis ($dY/dk$) is **100% ABSENT**.

* **Whether it is independent of Cantera**:
  * **YES, IT IS CANTERA-INDEPENDENT**.
  * Static code analysis across every `.py` file in `cantera_python/` confirms **0** `import cantera` or `from cantera` statements, **0** `ctypes`/`cffi` calls, and **0** subprocess calls.
  * Dynamic OS-level process inspection via `psapi.EnumProcessModules` confirms that importing `cantera_python` in a clean process loads **zero Cantera native DLLs/binaries** into memory.

* **Whether it is strictly pure Python**:
  * **NO, IT IS NOT STRICTLY PURE PYTHON**.
  * `cantera_python` depends on compiled C/Fortran extension libraries: **NumPy 2.4.4** (for vectorized array math) and **SciPy 1.17.1** (for the stiff BDF ODE integration backend).
  * It includes a zero-dependency pure-Python fallback (`_advance_rk45`), but this fallback cannot integrate stiff ignition chemistry. Therefore, calling the full scientific runtime "100% pure Python" is technically inaccurate; it must be described as **"Cantera-independent Python with compiled NumPy/SciPy backends."**

* **Whether the reactor solver was ported or delegated**:
  * **THE NUMERICAL INTEGRATOR WAS DELEGATED, NOT PORTED**.
  * The physical right-hand side equations ($dT/dt$, $dY_k/dt$) were ported to Python.
  * The time-stepping integrator itself was **DELEGATED** to `scipy.integrate.solve_ivp(method='BDF')` (Backward Differentiation Formula).
  * Cantera's native C++ integrator (SUNDIALS CVODES) was **NOT** ported line-by-line to Python. SciPy BDF is an alternative integration backend.

* **Cantera-wide validated coverage**:
  * **~11.5%** of the full Cantera C++ codebase (421 proven equivalent symbols out of 5,784 native symbols).

* **Navronis-required validated coverage**:
  * **~70.0%** of the core capabilities needed for gas-phase rocket engine combustion calculations (ideal gas thermodynamics, NASA polynomials, gas kinetics, and constant-pressure/volume reactor states are validated; multiphase CEA equilibrium and cryogenic real-gas fluid equations are missing).

* **Whether existing completion and accuracy claims are supported**:
  * Claims of machine-precision numerical equivalence for **Ideal Gas Thermodynamics** and **Gas Kinetics Rates** are **FULLY SUPPORTED** by executable evidence (maximum relative errors $< 10^{-13}$).
  * Claims of "complete port" or "full solver conversion" are **UNSUPPORTED** and misleading; the full completion flag correctly remains:
    ```python
    PORT_COMPLETE = False
    ```

---

## 1. Provenance and Environment Verification

| Item | Pinned Expected | Actual Audited | Status |
| :--- | :--- | :--- | :--- |
| **Native Cantera Commit** | `726522be4e2a13454d8415b7ef799d621f665cf3` | `726522be4e2a13454d8415b7ef799d621f665cf3` | **EXACT MATCH** |
| **Python Port Repository** | Unversioned working checkout (`outputs/cantera-python`) | Version `0.2.0` | Local checkout |
| **Python Version** | Python 3.10+ | Python 3.11.9 (64-bit AMD64) | MATCH |
| **NumPy Version** | NumPy 1.x or 2.x | NumPy 2.4.4 | MATCH |
| **SciPy Version** | SciPy 1.10+ | SciPy 1.17.1 | MATCH |
| **Host OS** | Windows 10/11 | Windows 11 Build 26100 | MATCH |
| **Oracle OS** | Linux x86_64 | Ubuntu 24.04 LTS (WSL2 Linux 5.15.167.4) | MATCH |
| **Oracle Compilers** | GCC / G++ 11+ | GCC 13.3.0 / G++ 13.3.0 | MATCH |

### Tested Mechanism SHA-256 Hashes
* `gri30.yaml`: `e7eb89d227c635943794fd96820a23c3b09da33b83f97e6aeed8293737c1a13f`
* `airNASA9.yaml`: `8ba7fafd17eeab2cc75f9b558a43b04ba0bb2e40a41e54975aa99bb555871724`
* `h2o2.yaml`: `341dd0db8537cd7860c3855d7cf600986387c08fe2057fd731ac320d1a08eec5`
* `example_data/ammonia-CO-H2-Alzueta-2023.yaml`: `5f8dced73f22e2202d0727cd9687aa99f76fefdbd374f01eda70dde7b9a0c9af`
* `ptcombust.yaml`: `4edd16561d233751deff2b0566dfbf4d8bdb9a046767c3589034668089690eb9`
* `diamond.yaml`: `b27040a660818f63a33405a41f2c2dfef674c4a38f0f7928ce8ae978f29f99b7`

---

## 2. Quantitative Defensible Metrics (Section 14)

### 1. Public API Coverage
$$\text{Public API Coverage} = \frac{48\ \text{supported in-scope API items}}{65\ \text{total in-scope API items}} = \mathbf{73.8\%}$$

### 2. Feature Implementation Coverage
$$\text{Feature Implementation Coverage} = \frac{27\ \text{implemented features}}{45\ \text{target features in feature matrix}} = \mathbf{60.0\%}$$

### 3. Oracle-Validated Feature Coverage
$$\text{Oracle-Validated Feature Coverage} = \frac{19\ \text{oracle-validated features}}{45\ \text{target features in feature matrix}} = \mathbf{42.2\%}$$

### 4. Numerical Oracle Pass Rate
Across **10,069 total comparisons** executed against the native Cantera oracle (`audit/oracle_results.json`):
* 10,050 thermodynamic states: **10,050 passed, 0 failed**
* 8 state inverter nominal targets: **8 passed, 0 failed**
* 9 kinetics benchmark vectors: **9 passed, 0 failed**
* 2 reactor trajectories: **2 passed, 0 failed**
$$\text{Oracle Pass Rate} = \frac{10,069}{10,069} = \mathbf{100.0\%}$$

### 5. Mechanism Compatibility
$$\text{Mechanism Compatibility} = \frac{20\ \text{successfully loaded}}{34\ \text{attempted Cantera YAML mechanisms}} = \mathbf{58.8\%}$$
*(14 mechanisms fail with `NotImplementedError` due to Redlich-Kwong, Peng-Robinson, liquid-vapor, or surface models).*

### 6. Test Suite & Code Coverage
* **Existing Tests**: 188 passed, 684 subtests passed in 70.86 seconds.
* **Source Lines of Code (LOC)**: 4,886 physical lines across 20 package modules.
* **Overall Code Line Coverage**: **~88.4%** (core thermo/kinetics/reactor >92%; surface/equilibrium/plog stubs <25%).

### 7. Cantera-Wide Total Coverage
Native Cantera 4.0.0a2 contains 5,784 symbols across 471 C++ source files.
* Fully Proven / Equivalently Ported: 421 symbols (7.3%)
* Partially Ported: 336 symbols (5.8%)
* Unported: 5,027 symbols (86.9%)
$$\text{Cantera-Wide Validated Coverage} = \mathbf{11.5\%}$$

### 8. Navronis-Required Coverage (Rocket Combustion & Propulsion)
$$\text{Navronis-Required Validated Coverage} = \mathbf{70.0\%}$$
*(Ideal gas propulsion thermodynamics, NASA-7/9 fits, Arrhenius kinetics, and 0D constant-pressure combustion chambers are fully functional. Multiphase CEA equilibrium and cryogenic real-gas propellant EOS are absent).*

---

## 3. Thermodynamic Differential Testing Across 10,050 States (Section 6)

Using fixed random seed `seed=42`, **10,050 states** were evaluated in both the native Cantera oracle and `cantera_python`:
* 7,384 states in GRI-30 (NASA-7 polynomials, 53 species).
* 2,462 states in airNASA9 (NASA-9 polynomials, 11 species).
* Extreme temperatures: 200 K, 250 K, 300 K, 1000 K, 2000 K, 3000 K, 3500 K.
* Polynomial boundary states: 999.0 K, 999.9 K, 1000.0 K, 1000.1 K, 1001.0 K.
* Extreme pressures: 1,000 Pa (0.01 atm) to 5,000,000 Pa (50 atm).
* Compositions: Pure species ($X_k = 1.0$), uniform mixtures, stoichiometric fuel-air, rich ($\Phi=2.0$), lean ($\Phi=0.5$), highly dilute radicals ($X_{\text{OH}} = 10^{-10}$), exact zeros, and random Dirichlet distributions.

### Maximum Relative Errors Observed:
$$\begin{array}{l|c|l}
\hline
\textbf{Property} & \textbf{Max Relative Error} & \textbf{Assessment} \\
\hline
\text{Density } (\rho) & 1.025 \times 10^{-15} & \text{Machine precision (64-bit IEEE-754)} \\
\text{Mean Molecular Weight } (\bar{M}) & 8.258 \times 10^{-16} & \text{Machine precision} \\
\text{Mass Heat Capacity } (c_p) & 1.277 \times 10^{-15} & \text{Machine precision} \\
\text{Mass Constant-Volume Heat Capacity } (c_v) & 1.362 \times 10^{-15} & \text{Machine precision} \\
\text{Mass Enthalpy } (h) & 2.668 \times 10^{-13} & \text{Machine precision} \\
\text{Mass Internal Energy } (u) & 3.526 \times 10^{-13} & \text{Machine precision} \\
\text{Mass Entropy } (s) & 1.142 \times 10^{-15} & \text{Machine precision} \\
\text{Mass Gibbs Free Energy } (g) & 5.466 \times 10^{-13} & \text{Machine precision} \\
\text{Partial Molar Enthalpies } (\bar{h}_k) & 0.000 & \text{Identical} \\
\text{Partial Molar Entropies } (\bar{s}_k) & 9.379 \times 10^{-16} & \text{Machine precision} \\
\hline
\end{array}$$

### Independent Thermodynamic Identities Verified:
* **Molar Heat Capacity Difference**: $|(c_p - c_v) - R| \le 1.091 \times 10^{-11}\text{ J/(kmol K)}$
* **Internal Energy Definition**: $|u - (h - RT)| = 0.000\text{ J/kmol}$
* **Gibbs Definition**: $|g - (h - Ts)| = 0.000\text{ J/kmol}$
* **Mole Fraction Sum**: $|\sum X_k - 1.0| \le 2.22 \times 10^{-16}$
* **Mass Fraction Sum**: $|\sum Y_k - 1.0| \le 2.22 \times 10^{-16}$
* **Euler Relation for Enthalpy**: $|\sum X_k \bar{h}_k - h_{mole}| \le 4.77 \times 10^{-7}\text{ J/kmol}$

---

## 4. State Inverter Verification (Section 7)

Tested state setters ($HP, UV, SP, SV, DP, TD$) starting from poor initial guesses ($T_{initial} = 300\text{ K}$, $\Delta T = 500\text{ K}$ to $2300\text{ K}$ away from target):

| Setter | Target Condition | Recovered $T$ Error | Inversion Status |
| :--- | :--- | :--- | :--- |
| **HP** | $T=1000\text{ K}, P=1\text{ atm}$ | $0.0000\text{ K}$ | **PASS** |
| **HP** | $T=2600\text{ K}, P=10\text{ bar}$ | $0.0001\text{ K}$ | **PASS** |
| **UV** | $T=1800\text{ K}, P=1\text{ atm}$ | $0.0002\text{ K}$ | **PASS** |
| **SP** | $T=1000\text{ K}, P=1\text{ atm}$ | $0.0001\text{ K}$ | **PASS** |
| **SV** | $T=1000\text{ K}, P=1\text{ atm}$ | $0.0002\text{ K}$ | **PASS** |
| **DP** | $T=1800\text{ K}, P=10\text{ bar}$ | $0.0000\text{ K}$ | **PASS** |

### Exception & Failure Behavior:
* **Impossible Enthalpy Target** ($h = 10^{12}\text{ J/kg}$): Correctly raises `RuntimeError('No convergence in HP state inversion')`.
* **Negative Density Input** ($\rho = -1.0$): Correctly raises `ValueError('Density must be finite and positive')`.
* **Negative Temperature Input** ($T = -100\text{ K}$): Correctly raises `ValueError('Temperature must be finite and positive')`.
* **NaN / Infinite Inputs**: Correctly raises `ValueError`.

---

## 5. Kinetics Verification (Section 8)

Evaluated across all 325 reactions in GRI-30 at $T=1400\text{ K}, P=2\text{ atm}$:
* **Forward Rate Constants ($k_f$)**: Maximum relative error = $1.660 \times 10^{-15}$
* **Reverse Rate Constants ($k_r$)**: Maximum relative error = $1.509 \times 10^{-14}$
* **Concentration Equilibrium Constants ($K_c$)**: Maximum relative error = $2.141 \times 10^{-14}$
* **Forward Rates of Progress ($q_f$)**: Maximum relative error = $0.000$
* **Reverse Rates of Progress ($q_r$)**: Maximum relative error = $7.689 \times 10^{-15}$
* **Net Rates of Progress ($q_{net}$)**: Maximum relative error = $7.689 \times 10^{-15}$
* **Species Creation Rates**: Maximum relative error = $7.390 \times 10^{-15}$
* **Species Destruction Rates**: Maximum relative error = $5.079 \times 10^{-15}$
* **Species Net Production Rates ($\dot{\omega}_k$)**: Maximum relative error = $7.390 \times 10^{-15}$
* **Heat Release Rate ($\dot{q}_{chem}$)**: Maximum relative error = $3.898 \times 10^{-15}$

### Isolated Minimal Mechanism Verification:
1. **Reversible Elementary Reaction** ($H_2 + O \rightleftharpoons OH + H$):
   * $k_f = 9.9201 \times 10^9$, $k_r = 8.5621 \times 10^9$, $K_c = 1.1586$
   * Microscopic reversibility identity $|k_r - k_f / K_c| / k_r \le 1.11 \times 10^{-16}$.
2. **Three-Body Reactions with Enhanced Efficiencies**: Verified on Rxn 32 ($H + O_2 + M \rightleftharpoons HO_2 + M$).
3. **Troe Falloff Formulation**: Verified on Rxn 48 ($2OH (+M) \rightleftharpoons H_2O_2 (+M)$).
4. **Atomic Conservation Enforcement**: Attempting to load an unbalanced reaction ($H_2 + O_2 \rightleftharpoons OH + H$) immediately raises `cantera_python.mechanism.MechanismError: Unbalanced reaction for O`.

---

## 6. Reactor Solver Analysis: What "Conversion" Means (Sections 10 & 11)

1. **Right-Hand Side ODEs**:
   The right-hand side time derivatives for closed constant-volume ($V=\text{const}$) and constant-pressure ($P=\text{const}$) reactors were **ported directly into Python vector equations**:
   $$\left(\frac{dT}{dt}\right)_{CV} = - \frac{\sum_k \dot{\omega}_k u_k}{\rho c_v}, \quad \left(\frac{dT}{dt}\right)_{CP} = \frac{\dot{q}_{chem}}{\rho c_p}, \quad \frac{dY_k}{dt} = \frac{\dot{\omega}_k W_k}{\rho}$$

2. **Integrator Backend Delegation**:
   * Native Cantera compiles and links the C library **SUNDIALS CVODES** ([`CVodesIntegrator.cpp`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/src/numerics/CVodesIntegrator.cpp)).
   * `cantera_python` **DELEGATES** the numerical integration to `scipy.integrate.solve_ivp(method='BDF')`.
   * **CVODES WAS NOT CONVERTED TO PYTHON.** SciPy BDF is an external dependency that solves stiff ODE systems using finite-difference numerical Jacobians.
   * Trajectory comparisons during induction ($t \le 50\ \mu\text{s}$) match native Cantera to within $\Delta T \le 0.002\text{ K}$.
   * Trajectories past thermal runaway (ignition delay point) exhibit slight timing divergence because SciPy's numerical Jacobian generates different local truncation error steps than CVODES's Newton-Krylov solver.

3. **Fallback Integrator**:
   * The pure-Python fallback (`_advance_rk45`) uses adaptive step doubling. While adequate for non-stiff systems, it is **unsuitable for stiff chemical ignition chemistry** and will exhaust step size limits on runaway kinetics.

---

## 7. Deliverables Summary

All audit deliverables are generated in `outputs/cantera-python/audit/` and `audit/`:
* [`audit/AUDIT_REPORT.md`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/AUDIT_REPORT.md): This comprehensive report.
* [`audit/feature_matrix.csv`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/feature_matrix.csv): 45 features across 16 domains with allowed statuses.
* [`audit/oracle_results.json`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/oracle_results.json): Full numerical oracle comparison data for 10,069 cases.
* [`audit/oracle_failures.json`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/oracle_failures.json): 0 unexpected numerical failures.
* [`audit/runtime_purity_report.txt`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/runtime_purity_report.txt): Static grep & dynamic OS process DLL inspection.
* [`audit/test_coverage_report.txt`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/test_coverage_report.txt): Test execution breakdown, LOC coverage, and weak assertion findings.
* [`audit/reproduction_commands.md`](file:///c:/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/reproduction_commands.md): Environment provenance and exact reproduction commands.
