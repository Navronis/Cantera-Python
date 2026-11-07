# Wave 3 `IMPLEMENTED_DIFFERENTLY` Declarations Disposition Report
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

## 1. Overview & Forensic Accounting

The pinned Cantera source code for the TPX Pure Fluid Solver Stack contains 395 total declarations across:
- `include/cantera/tpx/Sub.h`
- `src/tpx/Sub.cpp`
- `include/cantera/tpx/*.h` (8 pure fluid specializations: Water, Nitrogen, Oxygen, Hydrogen, Methane, CarbonDioxide, Heptane, HFC134a)
- `src/tpx/*.cpp`

The complete forensic inventory breakdown is:
- **`IMPLEMENTED_EXACTLY`**: 286 declarations
- **`IMPLEMENTED_DIFFERENTLY`**: 21 declarations
- **`NON_RUNTIME`**: 88 declarations (enums, typedefs, inline debug hooks, destructors)
- **Algorithmic disposition**: All 21 declarations have explicit dispositions. Ten core methods (and their 10 header declarations) are classified as semantically equivalent to pinned C++ Cantera. `Substance::set_xy` is mapped to the pinned direct determinant-division algorithm: Python computes the same two Newton numerators and uses `_ieee_divide` to preserve C++ IEEE-754 infinity/NaN behavior for zero determinants instead of raising Python's `ZeroDivisionError`. The former Python gradient-regularization branch is absent. The current 4,180-case targeted campaign did not activate a fallback because no Python-only fallback exists; this finite corpus does not prove outcomes for every degenerate thermodynamic state.

---

## 2. Exhaustive Roster of the 21 `IMPLEMENTED_DIFFERENTLY` Declarations

The 21 declarations consist of 11 core thermodynamic and solver routines implemented in `src/tpx/Sub.cpp` and their 10 corresponding virtual/member function declarations in `include/cantera/tpx/Sub.h`.

| ID | C++ Symbol & Location | Python Implementation | Classification | Observable Impact | Disposition Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Substance::cv()`<br>`Sub.cpp:44` | `cantera_python.tpx.Substance.cv` | `SEMANTICALLY_EQUIVALENT` | NONE | Exact port of $T (\Delta s / \Delta T)_v$ with boundary checking and NaN inside 2-phase dome. |
| 2 | `Substance::cp()`<br>`Sub.cpp:82` | `cantera_python.tpx.Substance.cp` | `SEMANTICALLY_EQUIVALENT` | NONE | Exact port with $T_{\mathrm{sat}}$ clipping, phase branch selection, and $+ \infty$ return in 2-phase dome. |
| 3 | `Substance::thermalExpansionCoeff()`<br>`Sub.cpp:134` | `cantera_python.tpx.Substance.thermalExpansionCoeff` | `SEMANTICALLY_EQUIVALENT` | NONE | Exact port of $2 (v_2 - v_1) / ((v_2 + v_1) \Delta T)$ with $T_{\mathrm{sat}}$ perturbation bounding. |
| 4 | `Substance::isothermalCompressibility()`<br>`Sub.cpp:187` | `cantera_python.tpx.Substance.isothermalCompressibility` | `SEMANTICALLY_EQUIVALENT` | NONE | Exact port of isothermal compressibility with $+ \infty$ return in 2-phase region. |
| 5 | `Substance::internalPressure()`<br>`Sub.cpp:235` | `cantera_python.tpx.Substance.internalPressure` | `SEMANTICALLY_EQUIVALENT` | NONE | Direct translation of $T (\partial P/\partial T)_v - P$ via central finite differences. |
| 6 | `Substance::dPsdT()`<br>`Sub.cpp:254` | `cantera_python.tpx.Substance.dPsdT` | `SEMANTICALLY_EQUIVALENT` | NONE | Direct translation of finite-difference $d P_{\mathrm{sat}} / d T$ with $T_{\mathrm{crit}}$ safeguarding. |
| 7 | `Substance::Tsat(double)`<br>`Sub.cpp:301` | `cantera_python.tpx.Substance.Tsat` | `SEMANTICALLY_EQUIVALENT` | NONE | Direct translation of Newton iteration for $T_{\mathrm{sat}}(P)$ with $0.1 T$ step limit and $[T_{\mathrm{min}}, T_{\mathrm{crit}}]$ clamping. |
| 8 | `Substance::Set(XY, x0, y0)`<br>`Sub.cpp:352` | `cantera_python.tpx.Substance.Set` | `SEMANTICALLY_EQUIVALENT` | NONE | Direct translation of 14-pair dispatch (TV, HP, SP, PV, TP, UV, ST, SV, UP, VH, TH, SH, PX, TX), two-phase Lever rule, and iterative fallbacks. |
| 9 | `Substance::set_xy(...)`<br>`Sub.cpp:639` | `cantera_python.tpx.Substance._set_xy_source` | `SOURCE_EQUIVALENT` | No Python-only regularization branch remains | Direct determinant division and both Newton numerators are source-mapped to pinned `Sub.cpp:701-704`; `_ieee_divide` models C++ IEEE results when Python float division would raise. The 4,180 targeted records are empirical coverage only, not proof of every degenerate-state outcome. |
| 10 | `Substance::BracketSlope(p)`<br>`Sub.cpp:768` | `cantera_python.tpx.Substance.BracketSlope` | `SEMANTICALLY_EQUIVALENT` | NONE | Direct translation of secant slope update between $[V_{\mathrm{min}}, P_{\mathrm{min}}]$ and $[V_{\mathrm{max}}, P_{\mathrm{max}}]$ with bracket damping. |
| 11 | `Substance::set_TPp(T, P)`<br>`Sub.cpp:787` | `cantera_python.tpx.Substance.set_TPp` | `SEMANTICALLY_EQUIVALENT` | NONE | Direct translation of 1D Newton-Raphson volume inversion at $(T, P)$ with liquid/vapor trial roots and exact C++ step limiting. |
| 12 | `virtual double cv()`<br>`Sub.h:123` | `cantera_python.tpx.Substance.cv` | `SEMANTICALLY_EQUIVALENT` | NONE | Virtual header declaration mapped to Python instance method. |
| 13 | `virtual double cp()`<br>`Sub.h:126` | `cantera_python.tpx.Substance.cp` | `SEMANTICALLY_EQUIVALENT` | NONE | Virtual header declaration mapped to Python instance method. |
| 14 | `virtual double thermalExpansionCoeff()`<br>`Sub.h:128` | `cantera_python.tpx.Substance.thermalExpansionCoeff` | `SEMANTICALLY_EQUIVALENT` | NONE | Virtual header declaration mapped to Python instance method. |
| 15 | `virtual double isothermalCompressibility()`<br>`Sub.h:130` | `cantera_python.tpx.Substance.isothermalCompressibility` | `SEMANTICALLY_EQUIVALENT` | NONE | Virtual header declaration mapped to Python instance method. |
| 16 | `virtual double internalPressure()`<br>`Sub.h:134` | `cantera_python.tpx.Substance.internalPressure` | `SEMANTICALLY_EQUIVALENT` | NONE | Virtual header declaration mapped to Python instance method. |
| 17 | `virtual double dPsdT()`<br>`Sub.h:143` | `cantera_python.tpx.Substance.dPsdT` | `SEMANTICALLY_EQUIVALENT` | NONE | Virtual header declaration mapped to Python instance method. |
| 18 | `double Tsat(double p)`<br>`Sub.h:146` | `cantera_python.tpx.Substance.Tsat` | `SEMANTICALLY_EQUIVALENT` | NONE | Header declaration mapped to Python instance method. |
| 19 | `void set_TPp(t0, p0)`<br>`Sub.h:173` | `cantera_python.tpx.Substance.set_TPp` | `SEMANTICALLY_EQUIVALENT` | NONE | Header declaration mapped to Python instance method. |
| 20 | `void Set(XY, x0, y0)`<br>`Sub.h:178` | `cantera_python.tpx.Substance.Set` | `SEMANTICALLY_EQUIVALENT` | NONE | Header declaration mapped to Python instance method. |
| 21 | `void BracketSlope(p)`<br>`Sub.h:222` | `cantera_python.tpx.Substance.BracketSlope` | `SEMANTICALLY_EQUIVALENT` | NONE | Header declaration mapped to Python instance method. |

---

## 3. Forensic Finding
1. **Disposition status**: All 21 declarations have explicit disposition records; the `set_xy` record now matches the current direct-division implementation and is not labeled as a numerical stabilization.
2. **Accounting**: `DIFFERENT_IMPLEMENTATIONS_TOTAL = 21`, `JUSTIFIED = 21`, `UNRESOLVED = 0`, `UNJUSTIFIED = 0`.
3. **Observable parity scope**: In the refreshed 9,608-case comparison across all 8 fluids, all 9,494 jointly successful cases pass the configured observable tolerances. Both engines fail on 114 cases, and all 114 normalized failure categories now match after the NaN volume predicate was aligned with native `set_v`. The five previously mismatched failure cases are documented in `wave3_metric_reconciliation.md`.
4. **Audit limitation**: The differential corpus and source mapping support the listed dispositions within their tested scope. They do not prove all possible inputs or every degenerate determinant outcome.
