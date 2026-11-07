# Wave 3 metric reconciliation: measured `cp` / `cv` differences
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

Pinned native source: Cantera `726522be4e2a13454d8415b7ef799d621f665cf3`.

## Source-supported formulas

Both native `Substance::cv()` and Python `Substance.cv()` use `dt = 1e-4 * T` and compute `T * (s2-s1)/(T2-T1)`. The perturbation holds specific volume fixed, clips to temperature limits, applies a one-sided endpoint rule when a pure-phase quality crosses into the two-phase region, and restores the original temperature. Native source is `src/tpx/Sub.cpp:44-80`; Python is `cantera_python/tpx.py:301-324`.

Both native `Substance::cp()` and Python `Substance.cp()` use the same nominal `dt` and entropy difference, with pressure held fixed through `(T,P)` states and saturation-aware `(T,x)` endpoints. They restore the saved `(T, density)` state. Native source is `src/tpx/Sub.cpp:82-132`; Python is `cantera_python/tpx.py:326-364`. Full branch detail is in [wave3_cp_cv_formula_audit.md](wave3_cp_cv_formula_audit.md).

## Fresh Wave 3 production comparison

The differential was rerun on 2026-09-29 against pinned native commit `726522be4e2a13454d8415b7ef799d621f665cf3` and the current Python source. Corpus, native, and Python case-ID sets are equal at 9,608 records. Both engines succeeded on 9,494 records and failed on 114. All 9,494 jointly successful cases passed the configured observable tolerances, and all 114 paired failures have matching normalized categories after the NaN-volume setter correction. The fresh comparator reports zero Wave 3 mismatches.

Before the correction, five `TH` inversion failures (`W3_PAIR_NITROGEN_TH_0390`, `W3_PAIR_OXYGEN_TH_0530`, `W3_PAIR_HYDROGEN_TH_0668`, `W3_PAIR_METHANE_TH_0804`, and `W3_PAIR_HEPTANE_TH_0948`) differed because C++ `set_v` rejects NaN via `if (v0 > 0)`, while Python's `if v <= 0` accepted NaN. Python now rejects any value for which `v > 0` is false, matching the source predicate; both sign variants of NaN have a regression test. After rerunning all Python records, all five now match the native failure category. The case-level evidence is in `native_wave3_results.jsonl`; the fresh aggregate is in `native_wave3_summary.json` and `native_parity_summary.json`.

## Perturbation-size study and causal conclusion

A 120-row cross-engine measurement sweep was run at `dt/T = 1e-3, 5e-4, 1e-4, 5e-5, 1e-5` across eight fluids and representative liquid, supercritical, and gas states. Its numeric distributions are in `wave3_cp_cv_perturbation_results.json` and row-level data is in `wave3_cp_cv_step_sweep.jsonl`.

This sweep's native path manually evaluates centered entropy differences. The production native `Sub::cp` / `Sub::cv` methods use temperature clipping and saturation-aware endpoints, including one-sided cases. The sweep therefore does not run a parameterized version of the production method and cannot isolate the cause of production `cp` / `cv` discrepancies. The earlier causal explanation based on `O(dt^2)` and near-saturation roundoff is withdrawn. The actual cause remains **`CAUSE_NOT_FULLY_ESTABLISHED`**; the fresh 9,608-case production parity result remains an independent empirical result.
