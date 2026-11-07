# Wave 1 Source Mapping: Thermodynamic Inverse State Solvers
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

**Target Subsystem**: Thermodynamic Inverse-State Solvers (`HP`, `UV`, `SP`, `SV`, `HPorUV`, `SPorSV`)  
**Pinned Native Reference**: Cantera commit `726522be4e2a13454d8415b7ef799d621f665cf3` (`include/cantera/thermo/ThermoPhase.h` and `src/thermo/ThermoPhase.cpp`)  
**Python Targets**: `cantera_python/state.py:77-259`, `cantera_python/ideal_gas.py:353-368, 904-915`, `cantera_python/cubic_eos.py:663-840`  
**Audit Date**: 2026-09-27  

## Certification status for the present worktree

This source map is paired with a fresh 2026-09-29 pinned-native run. The 4,100-case differential had matching case IDs, 4,000 successful solves, 100 failures, and zero observable or failure-category mismatches under the recorded tolerance policy. This supports successful-solve observable parity for that corpus; it does not establish every native branch or full behavioral equivalence.

The before/after failure probe measured 9 cases. Initially, the Python mutable HP/UV wrappers incorrectly committed the immutable solver's diagnostic trial state on exception. The pinned public C++ HP, UV, SP, and SV wrappers all save and restore partial phase state around the shared inversion call (`ThermoPhase.cpp:200-220, 560-580`). The Python mutable wrappers now commit only a successful result; after rerunning both engines, all 9/9 post-failure states match. The immutable low-level helper still attaches its trial candidate to the exception for diagnostics without mutating its input.

---

## 1. Overview of Wave 1 Mathematical Architecture

In Cantera C++, inverse thermodynamic state specifications are solved using a safeguarded Newton-bisection iteration over temperature $T$ while holding pressure $P$ (for HP, SP) or specific volume $v = 1/\rho$ (for UV, SV) constant.

The shared inverse paths evaluate a temperature residual and its thermodynamic derivative, cap the Newton step at 100 K, apply source-specific bracket safeguards, retry unstable derivative steps, and use residual/relative-step convergence tests. Only the HP/UV `setState_HPorUV` path applies the `Told / 3` lower-step safeguard; the SP/SV `setState_SPorSV` path does not.

---

## 2. Detailed Symbol-by-Symbol Mapping Matrix

| Native Function | Native Source File & Line | Governing Residual $F(T)$ | Derivative $F'(T)$ | Newton Update $\Delta T$ | Bracketing & Safeguards | Convergence Criterion | State Restoration & Failure Path | Python Implementation | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `ThermoPhase::setState_HP` | `src/thermo/ThermoPhase.cpp:200-210`, `include/cantera/thermo/ThermoPhase.h:373` | $h(T, P) - H_{\mathrm{target}}$ | $c_p(T, P) = (\partial h/\partial T)_P$ | $\mathrm{clip}\left(\frac{H_{\mathrm{target}} - h}{c_p}, -100, 100\right)$ | Saves partial state, dispatches to `setState_HPorUV(H, p, rtol, false)`, restores on exception | Same as `setState_HPorUV` | Public setter restores the original phase state on exception | Immutable equation in `cantera_python/state.py`; mutable transaction in `cantera_python/ideal_gas.py` | `SOURCE-MAPPED; 9/9 FAILURE-STATE PROBES MATCH` |
| `ThermoPhase::setState_UV` | `src/thermo/ThermoPhase.cpp:212-226`, `include/cantera/thermo/ThermoPhase.h:389` | $u(T, v) - U_{\mathrm{target}}$ | $c_v(T, v) = (\partial u/\partial T)_v$ | $\mathrm{clip}\left(\frac{U_{\mathrm{target}} - u}{c_v}, -100, 100\right)$ | Asserts compressibility, saves partial state, dispatches to `setState_HPorUV(u, v, rtol, true)`, restores on exception | Same as `setState_HPorUV` | Public setter restores the original phase state on exception | Immutable equation in `cantera_python/state.py`; mutable transaction in `cantera_python/ideal_gas.py` | `SOURCE-MAPPED; 9/9 FAILURE-STATE PROBES MATCH` |
| `ThermoPhase::setState_SP` | `src/thermo/ThermoPhase.cpp:560-570`, `include/cantera/thermo/ThermoPhase.h:405` | $s(T, P) - S_{\mathrm{target}}$ | $\frac{c_p(T, P)}{T} = (\partial s/\partial T)_P$ | $\mathrm{clip}\left(\frac{(S_{\mathrm{target}} - s)T}{c_p}, -100, 100\right)$ | Saves partial state, dispatches to `setState_SPorSV(S, p, rtol, false)`, restores on exception | Same as `setState_SPorSV` | Public setter restores the original phase state on exception | Immutable equation in `cantera_python/state.py`; mutable transaction in `cantera_python/ideal_gas.py` | `SOURCE-MAPPED; 9/9 FAILURE-STATE PROBES MATCH` |
| `ThermoPhase::setState_SV` | `src/thermo/ThermoPhase.cpp:572-582`, `include/cantera/thermo/ThermoPhase.h:421` | $s(T, v) - S_{\mathrm{target}}$ | $\frac{c_v(T, v)}{T} = (\partial s/\partial T)_v$ | $\mathrm{clip}\left(\frac{(S_{\mathrm{target}} - s)T}{c_v}, -100, 100\right)$ | Asserts compressibility, saves partial state, dispatches to `setState_SPorSV(S, v, rtol, true)`, restores on exception | Same as `setState_SPorSV` | Public setter restores the original phase state on exception | Immutable equation in `cantera_python/state.py`; mutable transaction in `cantera_python/ideal_gas.py` | `SOURCE-MAPPED; 9/9 FAILURE-STATE PROBES MATCH` |
| `ThermoPhase::setState_HPorUV` | `src/thermo/ThermoPhase.cpp:342-558` | $(1-d_{UV})(h - H) + d_{UV}(u - U)$ | $(1-d_{UV})c_p + d_{UV}c_v$ | $\mathrm{clip}\left(\frac{\mathrm{Tgt} - \mathrm{Val}}{\mathrm{Deriv}}, -100, 100\right)$ | Bisection bracketing: $T_{\mathrm{bot}}$ / $T_{\mathrm{top}}$ damped by $0.75 \Delta T_{\mathrm{bracket}}$. 10-step stability loop checks $C_p < 0$, cuts $\Delta T \times 0.25$, prevents $T < T_{\mathrm{old}}/3$. Bounds clamp $[T_{\mathrm{min}}, T_{\mathrm{max}}]$. | $\frac{|\mathrm{Tgt} - \mathrm{Val}|}{\max(|\mathrm{Tgt}|, \max(\|c_p\|, 10^{-5})T)} < \mathrm{rtol}$ OR $\left\|\frac{\Delta T}{T}\right\| < \mathrm{rtol}$ | Inner iteration mutates; public HP/UV wrappers restore state on exceptions | `cantera_python/state.py` implements the immutable equations and `ideal_gas.py` commits only on success for mutable phases | `SOURCE-MAPPED; 9/9 FAILURE-STATE PROBES MATCH` |
| `ThermoPhase::setState_SPorSV` | `src/thermo/ThermoPhase.cpp:585-781` | $s - S_{\mathrm{target}}$ | $\frac{(1-d_{SV})c_p + d_{SV}c_v}{T}$ | $\mathrm{clip}\left(\frac{(S - s)T}{\mathrm{Deriv}}, -100, 100\right)$ | Bisection bracketing: $T_{\mathrm{bot}}$ / $T_{\mathrm{top}}$ damped by $0.75 \Delta T_{\mathrm{bracket}}$. 10-step stability loop checks $C_p < 0$, cuts $\Delta T \times 0.25$. Bounds clamp $[T_{\mathrm{min}}, T_{\mathrm{max}}]$. | $\frac{|S_{\mathrm{target}} - s| T}{\max(|S_{\mathrm{target}}|, \max(\|c_p\|, 10^{-5})T)} < \mathrm{rtol}$ OR $\left\|\frac{\Delta T}{T}\right\| < \mathrm{rtol}$ | Inner solver mutates; public native SP/SV wrappers restore partial state on exceptions | Immutable equation in `cantera_python/state.py`; mutable commit/rollback in `cantera_python/ideal_gas.py` | `SOURCE-MAPPED; 9/9 FAILURE-STATE PROBES MATCH` |

---

## 3. Algorithmic Fidelity Verification

1. **Step-Size Clipping**: Both C++ source and Python implement exact `clip(dt, -100.0, 100.0)`.
2. **Positive Temperature Safeguard**: The C++ and Python HP/UV `setState_HPorUV` paths prevent temperature reductions below one-third of current temperature:
   `if (Tnew < Told / 3.0) { Tnew = Told / 3.0; dt = -2.0 * Told / 3.0; }`
   The SP/SV `setState_SPorSV` path has no corresponding one-third-temperature floor.
3. **Monotonicity Brackets**: Both solvers reduce a crossing step to `dt = 0.75 * (Tbot - Told)` (or the corresponding upper bracket). Their trigger differs by solver: HP/UV use the source's weighted `0.75*Tbot + 0.25*Told` threshold; SP/SV trigger only when the proposal crosses `Tbot`/`Ttop`. Python now follows both source conditions separately.
4. **Dual Convergence Criteria**:
   - Residual test: $|\mathrm{error}| / \mathrm{denom} < \mathrm{rtol}$ where $\mathrm{denom} = \max(|\mathrm{target}|, \max(|c_p|, 10^{-5}) \cdot T_{\mathrm{new}})$.
   - Temperature step test: $|\Delta T / T_{\mathrm{new}}| < \mathrm{rtol}$.
5. **State Side Effects on Failure**:
   - Source inspection shows internal iterative mutation and wrapper-level exception handling can affect the final observable state; do not infer final state solely from the inner loop.
   - The live probe now observes matching final-state observables in 9/9 cases. The correction changed only mutable wrapper commit/rollback behavior; failed immutable solves retain their diagnostic trial candidate on the exception.
   - The six local regression tests cover Python HP/UV mutation, SP/SV restoration, low-level immutability, property-evaluation errors, the unenforced Tmin clamp, and the absence of an HP/UV-only temperature floor in entropy solves. These local tests complement but do not replace the paired native probe.
6. **Unenforced lower temperature bound**: Native uses `clip(minTemp(), SmallNumber, 100.0)`. Python now applies the same clamp; when `minTemp()` is 200 K and enforcement is off, the fallback bound remains 100 K.

---

## 4. Source Mapping Summary
- Total Symbols Mapped: 6
- Numerical formulas and convergence controls are mapped for all 6 declarations.
- Source formulas and controls are mapped. Successful-solve and failure-category parity are supported by fresh differential records; the separate before/after failure probe matches 9/9. This probe covers the GRI ideal-gas `Solution` path and does not establish cubic-EOS setter side effects or every phase implementation.
