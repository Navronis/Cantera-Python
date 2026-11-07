# Performance methodology

Correctness gates precede timing. Each benchmark must record the mechanism,
state, backend, tolerances, checkpoint times, package versions, warm-up policy,
number of repetitions, wall time, and solver counters when exposed. Native,
parity, and native-Python results must use the same physical inputs and output
checkpoints.

Timing results are invalid if their outputs fail the associated absolute,
mixed-tolerance, conservation, or native differential checks. Python process
startup and mechanism parsing are reported separately from repeated numerical
evaluation. Median and dispersion are reported rather than a single best run.

The current parity binding exposes RHS and Jacobian evaluation counts. The
legacy augmented sensitivity path also records local RHS evaluations. Exact
native timings and broad benchmark cases remain open until a reproducible
pinned native build is available.
