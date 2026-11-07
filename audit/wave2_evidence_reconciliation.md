# Wave 2 Evidence Reconciliation & Forensics Report
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

> **UPDATED STATUS (2026-09-27):** The final section below is historical and superseded. The pinned native C++ probe was subsequently executed; see `audit/wave2_internal_root_parity_summary.md` and the raw native/Python probe records. Current results match all 1,962 sign-coded solver return values and all root values within tolerance; root-index branch classification is meaningful only for the 225 three-distinct-root cases.

## 1. Executive Summary & Objective

This document forensically audits and resolves the apparent contradiction in the Solver Wave 2 historical artifacts between:
- `"pinned_native_live": "UNAVAILABLE"` in [audit/solver_wave2_summary.json](file:///outputs/cantera-python/audit/solver_wave2_summary.json#L5)
- `"SOLVER_WAVE_2_LIVE_NATIVE_VALIDATED": true` in [audit/solver_wave2_summary.json](file:///outputs/cantera-python/audit/solver_wave2_summary.json#L21) and [audit/native_wave2_summary.json](file:///outputs/cantera-python/audit/native_wave2_summary.json).

Furthermore, this report enforces the core audit principle:
> **Claim Separation**:
> 1. Python implementation exists.
> 2. Python follows pinned Cantera source semantics.
> 3. Observable Python outputs match pinned native Cantera.
> 4. Internal algorithm / branch decisions match pinned Cantera.
> 
> Observable parity (Claim 3) does NOT automatically imply internal root parity (Claim 4). Both must be demonstrated independently.

---

## 2. Origin of the "UNAVAILABLE" Historical Artifact

### 2.1 Chronology of Events
1. **Initial Independent Oracle Verification**:
   - During the early phase of the Wave 2 audit, WSL execution encountered environment permissions (`E_ACCESSDENIED`), preventing direct native Cantera CLI calls.
   - An independent analytical cubic oracle ([tools/wave2_independent_oracle.py](file:///outputs/cantera-python/tools/wave2_independent_oracle.py)) implementing exact trigonometric (Viète / Cardano) solutions for Peng-Robinson and Redlich-Kwong was created.
   - Wave 2 was evaluated against this independent oracle across 2,000 test cases, achieving zero mismatches and max relative error of $4.59 \times 10^{-13}$.
   - The initial summary file `solver_wave2_summary.json` was generated with:
     ```json
     "pinned_native_live": "UNAVAILABLE"
     ```
2. **Native Environment Provisioning & Validation**:
   - Subsequently, WSL2 with Ubuntu 24.04 and Python 3.12 (`/home/nebula/cantera-oracle-env/bin/python3`) was properly provisioned.
   - Native Cantera commit `726522be4e2a13454d8415b7ef799d621f665cf3` was compiled and verified.
   - `tools/run_native_oracle_runner.py::run_wave2` was executed against `audit/corpus_wave2.jsonl`, generating `audit/raw_native_wave2.jsonl`.
   - `tools/compare_native_vs_python.py::compare_wave2` evaluated all 2,000 cases (1,962 successes, 38 failures), verifying 0 mismatches, and generated `audit/native_wave2_summary.json`.
3. **Artifact Staleness in `solver_wave2_summary.json`**:
   - When `solver_wave2_summary.json` was updated, the flag `"SOLVER_WAVE_2_LIVE_NATIVE_VALIDATED": true` was added at the bottom, but the legacy field `"pinned_native_live": "UNAVAILABLE"` on line 5 was not cleaned up.
   - This created an apparent textual contradiction.

### 2.2 Formal Resolution
- The field `"pinned_native_live": "UNAVAILABLE"` in `solver_wave2_summary.json` is formally recognized as **STALE HISTORICAL METADATA**.
- The authoritative current HEAD differential validation record is `audit/native_wave2_summary.json`, produced directly by comparing WSL native Cantera (`libcantera_shared.so.4`) against `cantera_python.cubic_eos` across the complete 2,000-case corpus.

---

## 3. Separation of Observable Parity vs Internal Root Parity

### 3.1 Observable Parity (Claim 3)
Observable thermodynamic properties returned by `PengRobinson` and `RedlichKwongMFTP`:
- $T, P, \rho, v, h, u, s, C_p, C_v$
Observable parity is verified by setting state $(T, P)$ and asserting that all thermodynamic properties match the native Cantera C++ implementation within strict numerical tolerances:
- Density $\rho$: $\text{rtol} \le 10^{-7}, \text{atol} \le 10^{-4}\text{ kg/m}^3$
- Enthalpy $h$: $\text{rtol} \le 10^{-7}, \text{atol} \le 1.0\text{ J/kg}$
- Entropy $s$: $\text{rtol} \le 10^{-7}, \text{atol} \le 0.01\text{ J/(kg K)}$
- Heat capacities $C_p, C_v$: $\text{rtol} \le 10^{-7}, \text{atol} \le 0.01\text{ J/(kg K)}$

### 3.2 Internal Root Logic (Claim 4)
In Cantera C++:
- `PengRobinson::solveCubic` (`PengRobinson.h:244`):
  Solves $Z^3 - (1 - B) Z^2 + (A - 3B^2 - 2B) Z - (AB - B^2 - B^3) = 0$ for molar volume roots $V = Z R T / P$.
- `RedlichKwongMFTP::solveCubic` (`RedlichKwongMFTP.h:394`):
  Solves the cubic polynomial for $V$.
- `densityCalc` (`PengRobinson.cpp:164` / `RedlichKwongMFTP.cpp:250`):
  - If $T > T_c$: selects single real root (or supercritical vapor branch).
  - If $T \le T_c$: evaluates Gibbs free energy $G$ or vapor pressure criterion to select between liquid root $V_{\mathrm{liq}}$ and vapor root $V_{\mathrm{vap}}$.
  - If 3 real roots exist: intermediate root is non-physical ($(\partial P/\partial V)_T > 0$).

To independently verify Claim 4, a dedicated test-only native C++ probe must invoke `calculateAB`, `solveCubic`, inspect `Vroot[0..2]`, `nSolnValues`, and compare them directly against `cantera_python.cubic_eos` internal solver outputs across all 2,000 cases.

---

## 4. Status
- Historical contradiction: **RESOLVED (Stale field identified and documented)**.
- Observable parity: **RECORDED** for 2,000 native/Python cases (1,962 successes, 38 paired failures, zero reported mismatches).
- Internal root/return-code parity: **RECORDED** for the same 2,000 cases; 1,962 successful return codes match and root values match within the reported tolerance. This is numerical agreement, not bit-exact roots.
- Three-root root-index selection: **RECORDED** for 225 cases (225 matching root-index classifications).
- The public native and Python runners and the independent native/Python internal root probes were rerun on 2026-09-29 against the pinned source and the same 2,000-case corpus. Case IDs match; 1,962 successful internal records agree on all return codes, root counts, selected roots, and root values within `rtol=1e-6, atol=1e-7`. The maximum relative root error is `1.359e-12`, so the result is numerical tolerance parity, not bit-exact roots. Current raw hashes are recorded in the Waves 1–3 evidence manifest.
