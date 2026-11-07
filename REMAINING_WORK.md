# Remaining work, based on the current implementation tree

This is an actionable gap list, not a claim that every listed native symbol has already received body-level review. The source-symbol and equation matrices remain the detailed indexes.

## P0 — stabilize and prove the baseline

- [x] Declare and document Python requirements used by actual runtime features: YAML parsing and reactor integration. Clean-venv automation remains to be added.
- [x] Reconcile the current test count and `VALIDATION.md`; the 2026-09-17 run records 192/192 passing in the YAML-enabled environment.
- [x] Resolve the reported H2/H2O collider discrepancy: the implementation was correct and the test confused one-based YAML comments with zero-based Python indices. Adjacent reactions are now guarded.
- [x] Replace the overflow-prone fixed-step fallback with adaptive RK4 step doubling and explicit rejected-step behavior. Exact ReactorNet/SUNDIALS equivalence remains unimplemented.
- [ ] Regenerate the exact pinned native oracle in isolation and record build configuration. A hash-verified pinned fixture is replayed successfully, but the current host denied WSL launch with `E_ACCESSDENIED`.
- [x] Add import-level and subprocess-level tests proving the package runtime does not import native Cantera.

## P1 — foundational object graph and input path

- [ ] Complete the YAML parser/schema and units conversion path. Phase selection, external imports, reaction rules, gas/surface initial state, adjacent-phase metadata, serialization, and strict unsupported-model diagnostics are implemented; interface coupling, extensions, and full AnyMap behavior remain.
- [ ] Wire all source-registered thermo, rate, kinetics and transport names/aliases through Python factories; test canonical names and aliases.
- [ ] Finish Species/Reaction/Phase/Solution objects and state mutation/cache invalidation. Cover composition changes, temperature, pressure, density, coverage and phase-potential updates.
- [x] Complete the current bulk duplicate reaction behavior, including reaction-order opt-ins, declarations, explicit collider parsing, all explicit-third-body policies, and undeclared-efficiency handling. Interface-specific duplicate coverage remains part of interface kinetics work.
- [x] Add a construction census for every bundled ideal-gas phase with exact failure reasons in `provenance/ideal_gas_construction_census_2026-09-17.json`. Pinned-oracle property comparison remains blocked pending the exact build.

## P1 — thermodynamics and kinetics

- [ ] Audit all species thermo manager variants and phase classes, including header-only branches and property derivatives.
- [ ] Complete nonideal solutions, activity/fugacity models, electrolytes, cubic EOS, TPX fluids, saturation, water and all state setters.
- [ ] Complete rate-family branches, derivatives, pressure/temperature bounds, units, limiting behavior and error semantics.
- [ ] Complete bulk/interface kinetics managers, reverse-rate state dependence, rate caching, forward/reverse/net rates, creation/destruction/production, derivatives and multiphase indexing.
- [ ] Implement surface kinetics, sticking, coverage dependencies, Motz-Wise, electrochemical corrections and interface phase coupling. Initial ideal-surface thermodynamics is implemented and matches the released diamond reference state.

## P1 — transport, numerics, equilibrium, reactors

- [ ] Port all transport data tables and models: collision integrals, viscosity, conductivity, binary/mixture/multicomponent diffusion, thermal diffusion, high-pressure/ionic/liquid/DustyGas where registered.
- [ ] Reproduce Cantera-specific residual/Jacobian scaling, Newton/damping/convergence rules, integrator error control, preconditioning and sensitivity behavior.
- [ ] Complete all equilibrium routes and fallback behavior. Ideal-gas TP/HP constrained Gibbs minimization is implemented with elemental and enthalpy checks; UV/SV/SP, charge constraints, multiphase VCS, and nonideal activities remain.
- [ ] Finish each zeroD reactor balance, walls, flow devices, surfaces, ReactorNet, advance/restart/error paths and sensitivities.
- [ ] Add stiffness/convergence and conservation tests against analytical cases and the pinned oracle.

## P1 — mandatory one-dimensional systems

- [ ] Port Domain1D/Sim1D/Flow1D state layout, boundary conditions, residuals, fluxes and Jacobians.
- [ ] Implement free/burner/counterflow/stagnation flames, transport/chemistry coupling, electric effects where applicable, solver fallback, refinement, continuation and restart.
- [ ] Demonstrate real convergence on representative cases and compare solutions to the pinned oracle.

## P2 — API and long-tail closure

- [ ] Complete public Python APIs, including Interface, Quantity, Mixture, transport, reaction paths, reactor/flame classes and serialization.
- [ ] Account for all relevant upstream tests and examples, with justified exclusions.
- [ ] Independently check every scientific data table and registration against pinned source.
- [ ] Run full clean-environment suite, exact-native differential suite, mechanism corpus, no-native scan, and adversarial forensic audit.

## Evidence currently available

- The audit summary lists 5,784 source symbols and 1,915 code-level math candidates, but name/body status still requires review and is not a completion fraction.
- The current source tree includes partial solution, transport, reactor and state-revision modules; presence of those files does not establish full subsystem behavior.
- The complete runtime suite passes, but passing tests cover only the implemented subset. Use the feature, mechanism, symbol, and oracle matrices to judge scope.
