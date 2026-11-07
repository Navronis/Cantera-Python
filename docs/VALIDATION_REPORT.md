# Validation report — 2026-09-18

> Current checkpoint: 294 tests pass in 95.761 seconds. The strict inventory has
> 759 final non-runtime rows, 644 partial rows, and 4,381 not-ported rows.
> Standalone CVODE 7.5 is operational for reactor parity mode. `../VALIDATION.md`
> and `../audit/scientific_status.json` supersede stale counts below.

Pinned reference: `726522be4e2a13454d8415b7ef799d621f665cf3`.

The runtime is a Python implementation. `cantera_python` does not import native Cantera or load its shared libraries. Cantera 3.2.0 exists only in the isolated validation environment for supplementary comparisons.

## Reproduced evidence

- The current recorded full regression command is `python -m unittest discover -s tests`. The suite contains 294 tests; all passed in 95.761 seconds on 2026-09-18 with Python 3.12, SciPy 1.18.1, and optional scikit-SUNDAE 1.1.3.
- The hash-verified pinned fixture contains 10,050 ideal-gas thermo states with seed 42 and 2,707,860 scalar/vector components. All components pass the property-specific absolute tolerance plus `1e-10` relative tolerance. The largest relative component error is `5.6098368711565595e-12` for a near-zero chemical potential; the largest composition-normalization residual is `2.220446049250313e-16`. The fixture spans 252.33–3498.01 K and 1 kPa–5 MPa, so it does not satisfy the requested 200–6000 K envelope.
- The exact-pinned scalar comparison utility reports 144 cases, zero failures, and maximum relative error `1.4411324199237575e-14`. Its fixture hash and abbreviated commit were verified against the pinned checkout before replay. Native fixture regeneration was not possible in this run because WSL launch returned `E_ACCESSDENIED`; the archived fixture therefore does not provide a reproducible native-build manifest.
- TP equilibrium for stoichiometric H2-air at 2500 K and 1 atm matches released Cantera 3.2.0 for major and dissociated species within the test tolerances. HP equilibrium gives `2380.806278093 K` at 1 atm versus `2380.806278078 K`; the temperature error is `1.4912e-8 K` and the maximum selected-species absolute error is `5.3454e-12`. At 30 atm, the temperature error is `1.1369e-10 K` and the species error is `2.4758e-14`. These are supplementary released-version results, not exact-pinned equilibrium evidence.
- The mechanism report contains all 34 bundled YAML files: 31 pass parser loading and 3 fail. This is parser reachability, not numerical certification.
- The legacy symbol inventory snapshot has 5,784 rows: 421 marked equivalent-proven, 336 partial, and 5,027 not ported. Equivalent-proven rows are 7.278%, but independent revalidation retained none of those 421 as a final validated scientific disposition. The current strict inventory has 759 proven non-runtime artifacts, 644 partial symbols, and 4,381 not ported; 5,025 symbols still lack a final disposition.
- The reproducible scoped coverage run passed 180 selected tests with 12 reactor-selected tests deselected. Runtime modules covered 4,026 of 5,313 statements (75.776%); branch-inclusive total coverage is 76.932% when test files are included. All 192 tests passed separately without instrumentation.

## Interpretation

The strong evidence is concentrated in ideal-gas thermodynamics, selected rate formulas, and the newly added TP/HP path. It does not justify a whole-library equivalence claim. Coverage, API counts, and selected-feature counts measure the current Python code or hand-selected matrices; none is a valid percent-complete denominator for Cantera.

Detailed per-property statistics are in `provenance/oracle_statistics.csv`. Claim reconciliation is in `provenance/baseline_reconciliation.csv`, and missing capabilities are listed in `docs/KNOWN_LIMITATIONS.md`.
