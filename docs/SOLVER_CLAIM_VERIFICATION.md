# Solver and conversion claim verification — 2026-09-18

> Updated checkpoint: 294 tests pass. Reactor parity now uses standalone
> SUNDIALS 7.5.0 CVODE through scikit-SUNDAE 1.1.3. Forward sensitivities use
> one augmented solve with local finite-difference Jacobians. Exact-pinned
> trajectory and sensitivity parity remain unproven. The authoritative status
> is `../audit/scientific_status.json`.

## Conclusion

The attached explanation is **partly correct about the architecture and materially overstates solver completeness, validation, and conversion percentages**.

Production Python source does not import native Cantera in the two checks run: the fresh-process test inspects `sys.modules`, and a source scan checks the package's `.py` files. That establishes runtime separation for those checks. It does not establish absence of every possible native binary-loading path.

The project should be described as a **Cantera-independent Python implementation using NumPy, SciPy, and an optional standalone SUNDIALS backend**. These are compiled numerical dependencies. Therefore “pure Python” is inaccurate as a description of the complete runtime/dependency stack.

## What the Python solver code does

The physical reactor right-hand sides are handwritten in [reactor.py](../cantera_python/reactor.py), including constant-volume and constant-pressure species, mass, temperature, and volume derivatives. `ReactorNet` has an operational standalone SUNDIALS 7.5 CVODE parity path. The legacy path uses SciPy BDF/LSODA and an adaptive RK fallback. Same-solver-family availability does not by itself prove exact-pinned trajectory parity.

The attached ODE description is not fully exact to the pinned source. Native `IdealGasReactor.cpp` includes internal-pressure, intrinsic-heating, and surface-production/mass-flux terms. The Python ideal-gas RHS does not implement those terms. For the narrower supported ideal-gas, no-surface case, several governing balances are represented, but this is a partial reactor port.

The Python equilibrium solver now has code paths for TP, HP, UV, SP, and SV in [equilibrium.py](../cantera_python/equilibrium.py). They use constrained ideal-gas Gibbs minimization with elemental constraints, followed by scalar temperature/pressure iteration for the other modes. Eleven focused mode, equilibrium, and runtime-purity tests passed. This is evidence that code exists and selected cases run; it does not confirm the claimed 25 pinned-native equilibrium cases. No matching equilibrium oracle data was found among the current differential artifacts.

Surface coverage advancement has a SciPy Radau path and an RK4 fallback in [surface.py](../cantera_python/surface.py). The explicit fallback now rejects non-finite or nonphysical trial coverages, reduces its step size, and reports a clear error if it exhausts its step budget. It is still unsuitable as a general stiff-surface solver. With the declared SciPy dependency installed, the Radau path passes the Pt-combustion coverage test and the complete suite.

Reaction multipliers are applied by the Python kinetics manager. The legacy sensitivity path now integrates state and all registered forward-sensitivity columns together using `dS/dt = Jy*S + Jp`. It computes `Jy` and `Jp` by local finite differences, so it remains approximate, but it no longer performs one complete perturbed trajectory per parameter. The standalone binding does not expose CVODES sensitivity APIs; parity-mode sensitivity therefore fails explicitly. Exact-pinned sensitivity trajectories remain unavailable.

## Current test evidence

The current tree discovers 294 unittest cases. On 2026-09-18, `python -m unittest discover -s tests` completed **294 tests in 95.761 seconds with zero failures and zero errors**. This establishes that the current regression suite is green in that environment. It does not establish equivalence to the pinned native solver or completeness outside the tested scope.

## Conversion percentage

There is no defensible single percent for conversion of the whole library. The latest auditable source-symbol snapshot records 421 equivalent-proven, 336 partial, and 5,027 not ported among 5,784 inventoried symbols. That is **7.28% equivalent-proven symbols in that audit snapshot**; 13.09% are equivalent-proven plus partial, and the partial portion is not validated. The audit predates some newer Python code, so these ratios are the latest auditable snapshot, not an exact current-source count. Symbol ratios, lines of code, test pass rate, and module count do not measure functionality reproduced.

The attached 782/5,784 (13.5%) equivalent figure conflicts with `provenance/forensic_audit_summary.json` and `provenance/original_cantera_symbols.jsonl`, both of which record 421 legacy equivalent-proven symbols. Under the stricter current audit, zero scientific symbols have a final validated disposition; 644 are partial, 4,381 are not ported, and 759 are proven inventory artifacts rather than runtime symbols. Older attached test counts and the 782-symbol figure are stale or unsupported.

The attached claims of 100% completion for selected thermo, kinetics, equilibrium, reactor, surface, and transport subsystems are not supported by the current evidence. Cubic EOS and multicomponent transport code exists but remains partial; TPX/IAPWS, VCS, DustyGas, high-pressure transport, IDAS FlowReactor, CVODES sensitivities, and other major families remain open.

Machine-readable results: [solver_claim_verification_2026-09-17.json](../provenance/solver_claim_verification_2026-09-17.json).
