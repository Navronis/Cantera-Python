# Wave 1 implementation checkpoint — 2026-09-17

Pinned source: `726522be4e2a13454d8415b7ef799d621f665cf3`.

Implemented in this checkpoint:

- phase-aware mechanism loading and required phase-name selection for multi-phase files;
- scoped species and reaction construction, including external file sections;
- Cantera-style `all` and `declared-species` reaction-section rules;
- preservation of species transport data and phase thermo/kinetics/transport/state metadata;
- initial ideal-gas `T`, `P`, `X`, and `Y` state application with unit-bearing scalar support;
- explicit rejection of unsupported phase and reaction models instead of silent ideal-gas/Arrhenius substitution;
- duplicate species/phase and malformed state/section diagnostics;
- P-log pressure parsing and live-pressure evaluation in scalar and vectorized kinetics;
- Chebyshev reaction construction, rate-unit offset, pressure parsing, and live-pressure evaluation;
- reaction-level `negative-A: true` support;
- Linear-Burke construction for P-log reference and explicit colliders;
- ionized-gas transport with n64 ion-neutral collisions, mobilities, electron diffusion, and electrical conductivity;
- electron pseudo-element charge inference;
- reaction-order opt-ins and duplicate-declaration validation;
- all pinned explicit-third-body duplicate policies (`warn`, `error`, `mark-duplicate`, and `modify-efficiency`) plus undeclared-efficiency handling;
- pinned thermo, kinetics, and transport alias canonicalization for the implemented factories;
- adjacent-phase and site-density metadata preservation;
- initial ideal-surface thermodynamics: multi-site coverage transforms, activities, concentrations, and chemical potentials;
- unit-bearing constant-cp species thermo fields;
- general length, quantity, time, pressure, and temperature unit-system conversion;
- fresh-process and source-scan runtime-purity tests;
- YAML-compatible mechanism serialization and round-trip loading;
- corrected NASA9 multi-region factory construction.
- ideal-gas TP constrained Gibbs minimization and HP temperature solve with conservation and reference tests.

The bundled-data inventory contains 20 ideal-gas phases. The regenerated census constructs all 20. This construction result proves parser/factory reachability only; it does not prove numerical equivalence.

The full runtime suite passes 192 tests with zero failures and zero errors. SciPy is absent, so the reactor tests use the adaptive pure-Python fallback.

The next closure work is randomized exact-pinned TP/HP equilibrium validation followed by UV/SV/SP, interface kinetics and adjacent-phase coupling, broader serialization parity, and remaining factory families. Waves 2–8 remain open, including nonideal thermo, complete transport, multiphase equilibrium, numerical solvers, reactors, one-dimensional flames, and API closure.
