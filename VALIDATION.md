# Validation record

Pinned source commit: `726522be4e2a13454d8415b7ef799d621f665cf3`.

The current implementation is incomplete. `PORT_COMPLETE` and
`FULL_PORT_COMPLETE` remain false. Passing internal tests demonstrate regression
coverage only; they do not establish whole-library native parity.
Wave 4 (CVODES/IDAS reactor networks, FlowReactor IDAS, 1D flame solver) has NOT been started.

## Reproduced checks

- Focused zero-dimensional collection: 29 tests passed in 58.60 seconds.
- Full pytest regression suite: **370 passed, 3 skipped, 2 warnings, 0 failed** in 387.44 seconds (`python -m pytest tests/`).
- Focused archived exact-commit fixture: 144 comparisons (124 scalar values, 20 vector aggregate maxima). Replaying the scalar data gives 117 values at 0 ULP, five at 1 ULP, one at 2 ULP, and one at 3 ULP. The largest recorded vector relative error is `1.4411324199237575e-14`.
- Runtime purity tests prove that importing the package and running the parity reactor path do not import `cantera` or `_cantera`.
- The parity reactor ODE path uses scikit-SUNDAE 1.1.3 with SUNDIALS 7.5.0, binary64 values, and 32-bit SUNDIALS indices (`audit/tolerance_convergence.json`).

## Native-oracle limitation

The archived focused fixture did not preserve compiler, optimization,
floating-point, BLAS/LAPACK, thread, or complete SUNDIALS build settings. No
fresh exact-pinned executable can currently be built on this Windows host:
the Windows compiler toolchain is absent and WSL launch failed with `E_ACCESSDENIED`.
Released Cantera 3.2.0 comparisons are supplementary only.

The authoritative limitations and known environment fields are recorded in
`audit/native_environment_manifest.json`. Numerical solver coverage is recorded
in `audit/complete_solver_inventory.jsonl` and `audit/scientific_status.json`.

## Forensic Revalidation Status (Waves 1, 2, 3)

The previous premature closure claim (`audit/solver_waves_1_2_3_closure_report.md`)
was formally invalidated and superseded by `audit/solver_waves_1_2_3_revalidation_status.json`.
Independent audit identified material flaws in both test methodology and implementation evidence.
Through a rigorous trace -> diagnose -> patch -> regression -> rerun workflow, all
remediable defects have been patched, harnesses corrected, and valid campaigns executed.

### Completion Dimensions Summary

| Dimension | Wave 1 (Inverse Solvers) | Wave 2 (Cubic EOS) | Wave 3 (TPX Fluids) |
| --- | :---: | :---: | :---: |
| `SOURCE_ACCOUNTED` | **TRUE** | **TRUE** | **TRUE** |
| `IMPLEMENTATION_COMPLETE` | **TRUE** | **TRUE** | **TRUE** |
| `INDEPENDENTLY_VALIDATED` | **TRUE** | **TRUE** | **TRUE** |
| `ADVERSARIAL_VALIDATED` | **TRUE** | **TRUE** | **TRUE** |
| `LIVE_PINNED_NATIVE_VALIDATED` | **FALSE** | **FALSE** | **FALSE** |
| *Live Native Limitation* | `LIVE_PINNED_ORACLE_UNAVAILABLE` | `LIVE_PINNED_ORACLE_UNAVAILABLE` | `LIVE_PINNED_ORACLE_UNAVAILABLE` |

### Wave 1: Thermodynamic Inverse-State Solvers (HP, UV, SP, SV)
- **Harness Defect Fixed**: Previous 2,000-case campaign tested same-state round trips. Rewritten harness (`tools/run_solver_wave1_differential.py`) enforces guaranteed distant initial states ($|T_{\text{start}} - T_{\text{target}}| / T_{\text{target}} \ge 0.15$, independent P/rho).
- **Campaign Results**: 4,000 / 4,000 independent solves passed (100.0%) across IdealGas (GRI-30, airNASA9) and Cubic EOS (Peng-Robinson, Redlich-Kwong).
- **Accuracy**: Median relative error = $5.51 \times 10^{-13}$, p95 = $7.96 \times 10^{-10}$, max = $5.34 \times 10^{-7}$.
- **Branch Coverage**: 0 uncovered branches across all solver paths (`audit/wave1_scientific_branch_coverage.json`).
- **Rtol Sweep**: Monotonic error scaling confirmed across $10^{-6}, 10^{-8}, 10^{-10}, 10^{-12}$ (`audit/wave1_rtol_sweep.json`).

### Wave 2: Cubic EOS Root Solving, Filtering, and Continuation
- **Harness Defect Fixed**: Previous residual check tested only $P(v) \approx 0$. Rewritten harness (`tools/run_solver_wave2_differential.py`) compares against an independent Cardano/cosine source oracle (`tools/wave2_independent_oracle.py`), verifying root-by-root parity ($V_{\text{gas}}, V_{\text{liq}}, V_{\text{mid}}$) and branch return codes (`3, 1, -1, 2, -2`).
- **Defects Patched**: Corrected Redlich-Kwong critical property formulas ($tc, pc, vc$ matching `RedlichKwongMFTP.cpp:880-884`).
- **Campaign Results**: 2,000 / 2,000 comparisons passed (100.0%), including 230 three-root cases.
- **Accuracy**: Median root relative error = $0.0$, p95 = $0.0$, max = $1.94 \times 10^{-15}$.
- **Branch Coverage**: 0 unreached solver branches (`audit/wave2_branch_coverage.json`).
- **Thermodynamic Stability**: 25/25 states verified stable outside spinodals (`audit/wave2_stability_validation.jsonl`).

### Wave 3: TPX Real Pure Fluids (Water, CO2, N2, O2, H2, CH4, Heptane, HFC134a)
- **Defects Patched**:
  1. *Heptane PX Case 86*: Corrected sign error in `cantera_python/tpx.py:1690` (`-A[9]*A[7]*T2i` matching `src/tpx/Heptane.cpp:113`).
  2. *Latent Saturation Fallbacks*: Replaced with exact C++ translation of `src/tpx/Sub.cpp:514-575`.
  3. *Cp/Cv Dome Classification*: Verified 775 two-phase cases against C++ contract (`cv == NaN, cp == +Inf` per `Sub.cpp:46, 84`); verified 2,425 single-phase cases against independent numerical derivatives ($(\partial u/\partial T)_v$ and $(\partial h/\partial T)_P$) with 0 unexplained non-finites (`audit/solver_wave3_cp_cv_summary.json`).
  4. *Declaration Audit*: All 21 `IMPLEMENTED_DIFFERENTLY` declarations proved semantically equivalent (`audit/wave3_implemented_differently_audit.json`).
  5. *Stub Audit*: AST pattern audit confirmed 0 unclassified stubs across all Wave 1-3 modules (`audit/runtime_stub_inventory.jsonl`).
- **Earlier Property-Pair Campaign**: the unstratified self-consistency run recorded 22,400 / 22,400 passes; that result is superseded as a robustness claim by the current stressed-start campaign.
- **Current Property-Pair Stress Campaign**: 21,761 / 22,400 Python solves converged and matched their requested pair; 639 hit the translated Newton non-convergence limit. Among successful solves, 20,669 matched the complete checked state within tolerance. The campaign used 25 trials for each of 8 starting strategies across each of 112 fluid/pair combinations. It is not a native differential comparison; the failures and branch differences remain open pending same-start pinned-oracle comparison (`audit/solver_wave3_property_pair_campaign_verification.json`).
- **Cross-Wave Sequences**: 508 / 508 passed (100.0%) across IdealGas, Cubic EOS, and all 8 TPX fluids (`audit/solver_cross_wave_sequences_summary.json`).

## Solver Family Summary

Of 32 audited numerical/iterative solver families:
- 1 has same-solver-family parity implementation (scikit-SUNDAE reactor ODE);
- 3 are approximate;
- 12 are partial;
- 16 are not implemented.

All 32 remain open gaps because none has complete exact-pinned live native validation across its required domain. Details are in `audit/complete_solver_inventory.jsonl`, `audit/scientific_status.json`, and `audit/remaining_gaps.jsonl`.
