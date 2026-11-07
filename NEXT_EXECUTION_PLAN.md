# Implementation roadmap from the current repository state

Pinned reference: Cantera `726522be4e2a13454d8415b7ef799d621f665cf3`.

This roadmap is based on the current code tree and audit artifacts, not the older 124-test snapshot. The port is a partial prototype. No end-to-end subsystem is yet demonstrated equivalent to the pinned library, and `FULL_PORT_COMPLETE` must remain false.

## Current evidence and immediate correction

- The lexical/body-audit artifacts list 5,784 source symbols: 421 `EQUIVALENT_PROVEN`, 336 `PARTIALLY_PORTED`, and 5,027 `NOT_PORTED`. Those labels apply only to the audited symbol universe; they are not a whole-library completion percentage.
- The math-candidate matrix lists 126 validated and 1,789 audited-unported candidates.
- Runtime YAML requirements are now declared and documented; the reference environment contains `ruamel.yaml` while SciPy remains absent.
- The authoritative 2026-09-17 run passes 192 tests. Reactor tests use the adaptive pure-Python fallback.
- The earlier GRI collider mismatch was a one-based-comment versus zero-based-index test defect; adjacent reactions now have regression coverage.
- `VALIDATION.md` and the run manifest describe the same current test snapshot. A hash-verified exact-pinned fixture is available and replayed; native regeneration remains blocked by host WSL `E_ACCESSDENIED`.

## Wave 0 — Make the baseline trustworthy (P0)

### 0.1 Reproducible environments

- Decide and document runtime dependency policy. Full YAML parsing needs a declared YAML parser. Reactor solver backend needs a declared SciPy dependency or a fully exercised, documented pure-Python fallback. Keep Cantera/native libraries out of runtime dependencies.
- Add clean-environment commands for core tests, parser/mechanism tests, and reactor tests; run each in a fresh venv so missing optional packages cannot silently change results.
- Keep native-reference dependencies in a separate environment from runtime tests.
- Update `README.md`, `pyproject.toml`, `DEPENDENCIES.md`, and `VALIDATION.md` only with results from the same run manifest.

### 0.2 Fix observed regression failures before broadening features

- Investigate GRI-30 reaction parsing around the reaction whose expected collider efficiency is H2 but whose `third_body.efficiencies` currently reports H2O. Trace YAML -> reaction parser -> `ThirdBody` and compare with the exact pinned GRI input and Reaction/ThirdBody source. Add a minimal regression fixture.
- Diagnose reactor integration overflow at `reactor.py` integration stages and Arrhenius evaluation. Determine whether invalid intermediate states should cause a rejected step, smaller step, or a source-compatible exception. Do not clip rates/temperature just to get tests green.
- Re-run tests after each correction in both clean runtime and reference-oracle environments.

### 0.3 Freeze source and oracle evidence

- Verify original commit and source status; preserve SUNDIALS long-path fixture evidence.
- Build an isolated exact pinned native reference. Record compiler, configuration, libraries, and commit. Do not expose it to `cantera_python` imports.
- Until built, label all Cantera 3.2 comparisons supplementary, never pinned validation.

**Wave 0 exit:** repeatable clean-environment run with no unexplained test failures; exact source commit verified; native oracle build either reproducibly available or blocker documented with concrete logs.

## Wave 1 — Runtime foundations and real mechanism construction (P1)

1. Finish core data objects: Element/Species/Reaction/Phase, units, aliases, compositions, errors, serialization, state ownership, and cache invalidation.
2. Replace mapping-only mechanism loading with schema-aware YAML loading for the common bulk gas case, retaining unsupported model diagnostics rather than silently ignoring input.
3. Add explicit factories that construct species thermo, reaction rates, phase kinetics, and transport from parsed data. Validate every registered model and alias from the pinned factories.
4. Complete mutable `Solution` semantics and per-phase state versioning; verify state changes invalidate thermo, kinetics, and transport caches.

Tests: upstream YAML schema cases; reaction equations including third bodies/orders/duplicates; alias and unit conversions; state/cache mutation sequences; GRI-30 parse counts and selected properties against the pinned oracle.

**Exit:** supported mechanism schemas produce actual Python model objects with explicit errors for unsupported schemas; no raw dictionaries mistaken for executable mechanisms.

## Wave 2 — Thermodynamic core and standard states (P1)

1. Audit every registered thermo and species-thermo class plus inline/header formulas.
2. Complete implemented NASA7/NASA9/Shomate/ConstCp/Mu0 behavior against source: boundaries, reference pressure, extrapolation, derivatives, packing/order, and all error paths.
3. Complete ideal-gas mutable state and thermodynamic property surface; test TP, TD, HP, SP, UV, SV, DP and composition/state combinations against exact source behavior.
4. Port ideal-solution and remaining incompressible/stoichiometric models, then nonideal activity models and electrolyte phases.
5. Port cubic EOS, TPX fluids, saturation, water and fugacity only after core standard states and inversion infrastructure are verified.

Tests: analytical identities, finite-difference derivatives, trace/pure compositions, pressure/temperature boundaries, state round trips, pinned differential checks.

**Exit:** each registered thermo model has implementation, parser/factory path, focused tests, and pinned comparisons; unknown models remain explicitly unsupported.

## Wave 3 — Complete gas and interface kinetics (P1)

1. Finish reaction objects, validation, orders, duplicate semantics, explicit collider rules and unit conversion.
2. Complete rate-family boundary semantics and derivatives: Arrhenius, Blowers-Masel, plasma/electron, falloff, PLOG, Chebyshev, Linear-Burke, custom and every additional pinned registration.
3. Complete equilibrium-derived reverse rates and detailed balance for all supported reaction orders and standard concentration conventions.
4. Implement kinetics-manager state updates, phase/species/reaction indexing, cached forward/reverse constants, ROP, creation/destruction/production rates, and invalidation.
5. Implement surface/interface rates, sticking, coverage dependence, Motz-Wise, electrochemical corrections, and multiphase behavior.
6. Implement every exposed derivative API and test with finite differences/complex step when valid.

Tests: pinned reaction-rate tests, randomized valid states, low/high-pressure limits, zero/trace concentrations, detailed balance, element/charge conservation, cache mutation tests, derivative comparisons.

**Exit:** every pinned kinetics/rate registry row is constructed through its factory and has tested calculation and derivative behavior where the upstream API provides it.

## Wave 4 — Transport (P1)

1. Port transport data, collision integral coefficients/fits, and exact source tables.
2. Implement mixture-averaged viscosity, conductivity, binary/mixture diffusion and thermal diffusion.
3. Implement multicomponent, ionized/high-pressure, liquid and DustyGas models that are actually registered in the pinned commit.
4. Add transport factory/API integration and state cache tests.

Tests: collision-integral table hash checks, known dilute limits, mixture rules, species-order invariance, randomized pinned comparisons, transport units.

**Exit:** all pinned transport models and registries accounted for; no model represented only by a name or placeholder.

## Wave 5 — Numerical foundation and equilibrium (P1)

1. Inventory generic numerical operations versus Cantera-specific solver behavior.
2. Reproduce residual construction, scaling, Jacobians, damping/line-search, convergence and failure semantics. Use SciPy only for generic kernels with the Cantera-specific behavior visible in Python.
3. Port equilibrium routes individually: ChemEquil/element potentials, VCS, multiphase/Gibbs, component selection, charge/element constraints, fallback behavior.
4. Cross-check equilibrium state and conservation against pinned oracle over mechanisms and difficult initial states.

Tests: matrix/root kernels, Jacobians, convergence/failure paths, elemental/charge balance, multiple solver routes and fallback branches.

**Exit:** all pinned equilibrium entry points have real solvers, source-mapped residuals and constraints, and reproducible pinned comparisons.

## Wave 6 — Zero-dimensional reactors and sensitivities (P1)

1. Implement reactor residuals/balances for each pinned reactor type; keep energy, mass, volume, pressure, wall and surface equations separately auditable.
2. Add ReactorNet integration with error control, rejected steps, tolerances, state synchronization, advance limits and failure behavior.
3. Implement walls, reservoirs, flow devices, surfaces, steady-state and sensitivities/preconditioners.
4. Validate mass, elements, charge, energy and analytic simple cases.

Tests: stiff and nonstiff mechanisms, constant pressure/volume, heat transfer, moving boundaries, coupled surfaces, sensitivities, restart and expected failure.

**Exit:** each supported reactor class actually integrates source-derived governing equations and passes conservation and pinned-reference tests.

## Wave 7 — One-dimensional reacting flow / flames (P1, highest risk)

1. Implement domain state layout, boundary conditions, residual assembly, fluxes and Jacobians.
2. Implement flow and flame classes from source equations, chemistry/transport coupling and electric effects where applicable.
3. Port nonlinear solve, pseudo-time fallback, continuation, grid refinement/pruning, save/restore and diagnostics.
4. Validate canonical freely propagating, burner and counterflow cases against pinned builds and upstream tests.

**Exit:** representative flame problems converge using the Python equations/solver; name-compatible classes alone do not count.

## Wave 8 — TPX, public API, auxiliary behavior, closure (P2)

1. Finish TPX fluids/water and phase-boundary/saturation behavior.
2. Complete public API parity for Solution/Interface/Quantity/Mixture/reactors/flames/transport/reaction paths.
3. Run mechanism and example corpora; classify every relevant upstream test (the third-party vendor tests may be excluded only with reasons).
4. Regenerate symbol, equation, data-table, registry, mechanism and test matrices from reproducible tools.
5. Have an independent adversarial review compare source bodies, headers, inline functions, factories, tables, tests and Python behavior.
6. Clean-environment full test, no-native scan and exact-oracle differential run.

**Exit:** all required source rows validated; zero unported scientific features, unexplained failures or runtime stubs; full completion gate independently checked.

## Workstream ownership

- Thermo/equilibrium: Wave 2 and Wave 5, plus TPX/water in Wave 8.
- Kinetics/transport: Wave 3 and Wave 4.
- Numerics/reactors/oneD: numerical core in Wave 5, reactors in Wave 6, flames in Wave 7.
- Parser/API/audit: Wave 0, Wave 1, and Wave 8; maintain traceability and cross-stream integration.

## Release/checkpoint discipline

For each small model group: source audit → implementation → equation tests → pinned differential validation → provenance update → independent gap review → full regression suite. Never promote a whole subsystem based on partial model tests. Keep exact test counts and environment metadata in one generated run record; keep completion false until the full gate passes.
