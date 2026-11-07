# Solver Wave 2 Internal Root Parity & Branch Decision Forensic Report
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

## 1. Scope & Objective
This report verifies **Claim 4 (Internal algorithm / root branch parity)** for Solver Wave 2 (Peng-Robinson and Redlich-Kwong equations of state) against pinned native Cantera commit `726522be4e2a13454d8415b7ef799d621f665cf3`.

Observable output parity (Claim 3) was verified across all 2,000 cases in `audit/native_wave2_summary.json`. This report validates that the internal cubic solver `solveCubic` and branch selection logic in `densityCalc` produce identical roots, root counts (`nsol`), and liquid/vapor branch decisions.

---

## 2. Quantitative Verification Results

| Metric | Measured Value | Requirement | Status |
| :--- | :--- | :--- | :--- |
| Total Cases Evaluated | **2000** | 2000 | PASS |
| Successful Solves | **1962** | 1962 | PASS |
| Deterministic Failures | **38** | 38 | PASS |
| `nsol` Discrepancies | **0** | 0 | **PASS** |
| Root Value Discrepancies ($> 10^{-6}$) | **0** | 0 | **PASS** |
| Branch Decision Discrepancies | **0** | 0 | **PASS** |
| Maximum Root Relative Error | **1.359e-12** | $< 10^{-6}$ | **PASS** |
| Maximum Molar Volume Relative Error | **4.671e-14** | $< 10^{-6}$ | **PASS** |

---

## 3. Solution Multiplicity & Branch Distributions

### Sign-coded solver return-code distribution:
- **`nsol = 1` (single real root, gas return code)**: Native: 1317, Python: 1317
- **`nsol = -1` (single real root, liquid return code)**: Native: 420, Python: 420
- **`nsol = 3` (three distinct real roots)**: Native: 225, Python: 225
- **All other return codes**: Native: 0, Python: 0

### Selected root among three-distinct-root cases only:
- **Vapor Branch Selected**: Native: 224, Python: 224
- **Liquid Branch Selected**: Native: 1, Python: 1
- **Middle Branch Selected (should be 0)**: Native: 0, Python: 0

---

## 4. Forensic Conclusion
- All 1,962 successful cases match the native sign-coded `nsol` return code. Root values agree within the reported numerical tolerance (the maximum error is nonzero, so this is not bit-exact parity).
- Root-index classification is reported only for three-distinct-root cases; single-root cases are represented by their sign-coded return code and are not assigned a root-array-index branch label.
- This probe compares the return code, roots, and selected molar volume. It does not expose or certify every internal conditional in `densityCalc`.
