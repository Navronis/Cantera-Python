# Wave 3 `cp` / `cv` Step Sweep: Measurement Scope
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

Pinned source commit: `726522be4e2a13454d8415b7ef799d621f665cf3`. Records: 120.

This campaign manually evaluates entropy differences at several step ratios. Its native evaluator uses centered `T-dT` / `T+dT` samples; the pinned `Sub.cpp` production methods use clipped and saturation-aware endpoints. The campaign therefore measures cross-engine property evaluation only and does not reproduce the native production `Sub::cp` / `Sub::cv` algorithm.

The source formula audit separately maps native `Sub.cpp` and Python `tpx.py`, including nominal `dt = 1e-4*T`, entropy differences, held constraints, clipping, and restoration. This sweep does not make the production methods identical.


## Measured relative differences by step size

| `dt/T` | property | p50 | p95 | p99 | max | records |
|---:|:---|---:|---:|---:|---:|---:|
| `1.0e-03` | `cv` | 1.510e-12 | 1.397e-09 | 2.141e-09 | 2.298e-09 | 24 |
| `1.0e-03` | `cp` | 3.326e-12 | 8.710e-09 | 1.286e-08 | 1.368e-08 | 24 |
| `5.0e-04` | `cv` | 9.959e-13 | 1.398e-09 | 2.140e-09 | 2.296e-09 | 24 |
| `5.0e-04` | `cp` | 2.768e-12 | 8.709e-09 | 1.285e-08 | 1.367e-08 | 24 |
| `1.0e-04` | `cv` | 5.241e-12 | 1.400e-09 | 2.147e-09 | 2.305e-09 | 24 |
| `1.0e-04` | `cp` | 6.434e-12 | 8.706e-09 | 1.285e-08 | 1.367e-08 | 24 |
| `5.0e-05` | `cv` | 3.496e-12 | 1.409e-09 | 2.130e-09 | 2.282e-09 | 24 |
| `5.0e-05` | `cp` | 1.049e-11 | 8.788e-09 | 1.286e-08 | 1.368e-08 | 24 |
| `1.0e-05` | `cv` | 2.576e-11 | 2.149e-09 | 7.197e-09 | 8.679e-09 | 24 |
| `1.0e-05` | `cp` | 4.451e-11 | 9.559e-09 | 1.283e-08 | 1.363e-08 | 24 |

## Causal conclusion

These data do not isolate the cause of the saved `cp` / `cv` differences in the separate 9,608-case production comparison. Finite-difference truncation, roundoff, phase-boundary sensitivity, state-reset behavior, and implementation-path differences remain possible contributors; this experiment does not distinguish them.

**Disposition: `CAUSE_NOT_FULLY_ESTABLISHED`.**

