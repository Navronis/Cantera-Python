# Current checkout validation: Waves 1–4

This record binds fresh tests and exact-pinned comparisons to the current working tree. It does not certify Cantera as a whole.

## Results

- Python regression suite: **483 passed, 3 skipped**, 486 collected, 698 subtests passed, 133.09 seconds. Command: `python -m pytest tests/`.
- Fresh exact-pinned Cantera comparisons: **15,708 cases, zero mismatches** under the repository's recorded property and failure tolerances:
  - Wave 1 inverse-state cases: 4,100 (4,000 successful states, 100 expected failures).
  - Wave 2 cubic EOS cases: 2,000 (1,962 successful states, 38 expected failures).
  - Wave 3 TPX cases: 9,608 (9,494 successful states, 114 expected failures), including 1,120 sampled property-pair replays.
- The native checkout reports commit `726522be4e2a13454d8415b7ef799d621f665cf3`; the extension and static-library hashes match `audit/native_oracle_identity.json`.
- The rerun used current `cantera_python/*.py`, `tests/*.py`, runner scripts, tolerance policy, and corpora. Their hashes and every raw result hash are in [current_tree_parity_manifest.json](current_tree_parity_manifest.json).

## What this proves

For these corpora, the present Python implementations produced the same compared properties and success/failure outcomes as the pinned native oracle. The Python side calls the repository's Python implementations. The native library is used only by the separate oracle process; it is not imported by the Python runtime package.

This does not establish full-domain equivalence for those models, post-failure state equivalence, dynamic ODE/DAE integration parity, or coverage of other Cantera subsystems. It does not validate the full 5,784-symbol inventory or the remaining solver families. The generated top-level closure gate has not yet been taught to consume this isolated evidence and should not be represented as passed.

## Later follow-up, 2026-09-29

After the parity snapshot above, the VCS API was corrected to reject true multi-phase equilibrium explicitly instead of returning a gas-only result with heuristic condensed-phase amounts. The single-phase convenience path preserves the initial mass basis as composition changes. `MultiPhase` now aggregates species and element amounts, phase charges, and the phase-local composition snapshot; one gas-plus-graphite case matches the pinned native container for 54 species and 5 elements. Focused multiphase/VCS tests pass **10/10**. A fresh full-suite run passes **488 tests**, skips 3, and passes 698 subtests; the log and file hashes are recorded in [vcs_followup_20260929.json](vcs_followup_20260929.json).

The new full-suite run does not add pinned-native VCS evidence. VCS remains `NOT_IMPLEMENTED`; the earlier Wave 1–4 parity figures remain bound only to the source hashes in the original manifest and were not rebound to the follow-up tree.

## Wave 4 trace-heavy MultiTransport diagnostic

A fresh probe was built from the current C++ probe source and linked to the exact pinned Cantera shared library. The same GRI-3.0 trace-heavy composition was evaluated by the current Python transport implementation at 1200 K and 100 kPa.

- The 159-by-159 matrix differs by at most `4.48e-13`; the right-hand side differs by at most `4.24e-22`.
- The matrix condition number is about `1.93e21`. The native and Python internal solution vectors differ substantially, but each has an infinity-norm residual near `3.3e-16`.
- The pinned source's public thermal-conductivity formula gives `0.15898332455490657 W/m/K`; Python returns `0.1589833245549065 W/m/K`.
- All 53 thermal-diffusion coefficients agree within `1.6e-20` absolute error in this state. A regression test now checks these public outputs against [the pinned reference fixture](../../tests/fixtures/pinned_multitransport_trace_heavy.json).

This one state shows that the large internal-vector difference does not carry through to these public properties. It does not close MultiTransport: broader composition, temperature, pressure, flux, and diffusion-coefficient comparisons remain necessary.

## Current strict inventory

- Canonical source symbols: 5,784.
- Proven non-runtime dispositions: 759 (13.12% of the inventory).
- Open dispositions: 5,025, comprising 646 partial and 4,379 not ported (86.88%).
- Solver families: 32 total; 3 numerically equivalent, 15 partial, 3 approximate, and 11 not implemented. Thus 9.38% are validated as equivalent; the 21 with some implementation are not all complete.

These are inventory disposition counts, not a percentage of all Cantera equations converted. There is no defensible whole-library conversion percentage until scientific-runtime symbols are mapped and validated individually.

## Reproduction

The `run_*_wave*.py` scripts in this folder call the repository's differential runners while directing their result files here. `compare_wave12.py` compares Waves 1–3; `compare_wave4_trace_heavy.py` records the Wave 4 probe comparison. `write_manifest.py` records source, corpus, oracle, result, and full-suite log hashes. The original case corpora were not changed by these reruns.
