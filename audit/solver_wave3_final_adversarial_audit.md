# Wave 3 adversarial audit
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

Pinned native reference: Cantera `726522be4e2a13454d8415b7ef799d621f665cf3`.

## Current differential evidence

The 9,608-case native/Python corpus covers all eight TPX fluids and all 14 public property-pair identifiers represented in the corpus. Both engines were freshly rerun on 2026-09-29. The first refreshed comparison found five failure-category differences among 114 paired failures; all successful outputs passed the configured tolerances. Investigation traced those differences to a Python `set_v` predicate that let NaN volumes pass. Python now uses the native source condition (`v > 0`), with regression coverage for both NaN sign bits. After rerunning Python on all three waves, all 9,608 Wave 3 outcomes match: 9,494 successful property results within tolerance, 114 paired failures with matching categories, and zero mismatches.

## Targeted Jacobian evidence

The current implementation performs direct determinant division, mapped to native `Sub.cpp:701-704`; it has no Python-only gradient fallback. A test-only trace sweep sampled 4,180 near-critical and property-pair attempts and observed zero fallback activations. That is expected because there is no fallback branch to activate; it is not validation of a fallback. The 88-case singular corpus was freshly compared on both engines: 88/88 statuses agree, 85 cases succeed, and all common saved observables pass their tolerances. Native records omit internal energy and quality, so those fields are unavailable for comparison.

## Independent Python property-pair campaign

The separate deterministic campaign contains 22,400 Python-only target/start attempts across eight fluids and 14 property pairs. Its latest saved run records 21,760 successes and 640 inversion failures (2.857%). It validates reachability and Python behavior only. It is not a native differential and does not prove that every failed input is outside the pinned model domain; those failures require source-paired replay before being labeled expected.

The earlier audit text claiming an unavailable live oracle, gradient fallback use, or zero mismatch across every Wave 3 failure path is superseded by these current findings.
