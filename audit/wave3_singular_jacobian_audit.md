# Wave 3 Jacobian and near-critical validation
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

Pinned native reference: Cantera `726522be4e2a13454d8415b7ef799d621f665cf3`.

## Solver path and activation result

The current Python `Substance._set_xy_source` at `cantera_python/tpx.py:774-825` computes the same two finite-difference Jacobian numerators and divides each directly by the determinant through `_ieee_divide`. The pinned C++ implementation does the raw determinant division in `src/tpx/Sub.cpp:701-704`. The current Python method has no determinant threshold, gradient projection, or Python-only fallback branch. The earlier report describing such a branch is stale and is superseded by this source mapping.

The test-only activation instrumentation was refreshed against current source. It recorded **4,180 targeted attempts and zero fallback activations**. Each attempt includes determinant observations and the requested event fields; activation threshold is null because no fallback threshold exists in the implementation. This is evidence that the removed fallback did not run; it cannot validate the behavior of a nonexistent branch. Therefore `WAVE_3_FALLBACK_ACTIVATION_VALIDATED = FALSE`. The mapped direct-division path is classified `SOURCE_EQUIVALENT` at the equation level, with finite-input coverage limits.

## Pinned-native singular corpus

The 88-case corpus was rerun on both engines. All 88 status results match: 85 successes and 3 failures on each side. For the 85 jointly successful cases, all common saved observables passed their stated tolerances. Maximum relative differences were:

| Observable | Maximum relative difference |
|---|---:|
| Temperature | `0` |
| Pressure | `9.096e-9` |
| Specific volume | `8.007e-6` |
| Density | `8.007e-6` |
| Enthalpy | `2.568e-6` |
| Entropy | `1.011e-6` |

The native records do not contain internal energy or quality, so those observables are unavailable for native comparison. Error distributions and residual details are in `wave3_singular_jacobian_comparison.json`; raw records are `raw_native_wave3_singular.jsonl` and `raw_python_wave3_singular.jsonl`.

Failure status matches do not mean identical exception types or text. The comparison did not certify message equality. This corpus establishes empirical parity for the listed cases and observables only.

## Whole-corpus context

The refreshed 9,608-case Wave 3 comparison covers all eight fluids. Both engines succeeded on 9,494 cases and failed on 114. All successful observables pass the configured tolerances; 109 paired failures have matching normalized categories, while five do not. The five category mismatches are listed in `wave3_metric_reconciliation.md`. Consequently, successful-solve observable parity is supported for the corpus, but complete Wave 3 failure-behavior parity is not.

