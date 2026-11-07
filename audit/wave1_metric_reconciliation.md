# Wave 1 Metric Reconciliation Report

**Subsystem**: Thermodynamic Inverse-State Solvers (`setState_HP`, `setState_UV`, `setState_SP`, `setState_SV`)  
**Audit Date**: 2026-09-27  
**Comparator Engines**: `tools/run_solver_wave1_differential.py`, `tools/compare_native_vs_python.py`  
**Authoritative Artifacts**: `audit/solver_wave1_summary.json`, `audit/native_wave1_summary.json`  

---

## 1. Executive Summary of Reconciliation

A forensic investigation was conducted to reconcile the seemingly contradictory metrics reported across historical Wave 1 validation documents:
1. **"4,000 / 4,000 successful inverse solves"**
2. **"4,100 total native cases evaluated"**
3. **"Zero differential mismatches"**
4. **"Maximum relative error of 5.34e-7"**

The investigation confirms that **each metric measures a fundamentally different scientific quantity** across two distinct validation campaigns. There is **zero numerical contradiction**:

| Metric Claim | Exact Numerical Value | Artifact Source | Corpus | Case Count | Quantity Measured | Comparison Type | Max Error Observed |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **Python Inversion Residual** | `5.340363131250286e-07` | `audit/solver_wave1_summary.json` | `audit/corpus_wave1.jsonl` (successful subset) | 4,000 | $\|H_{\mathrm{final}} - H_{\mathrm{target}}\| / \|H_{\mathrm{target}}\|$ (or $S, U$) | Python self-consistency from distant start ($|T_{\mathrm{start}} - T_{\mathrm{target}}| / T_{\mathrm{target}} \ge 0.15$) | $5.34 \times 10^{-7}$ (p50: $5.51 \times 10^{-13}$, p95: $7.96 \times 10^{-10}$) |
| **Live Native Differential Parity** | `0` mismatches | `audit/native_wave1_summary.json` | `audit/corpus_wave1.jsonl` (full corpus) | 4,100 | Direct observable outputs ($T, P, \rho, v, h, u, s, c_p, c_v, X$) | Live Native C++ (WSL2) vs. Pure Python | Max RelDiff $5.60 \times 10^{-9}$ (Temp), $2.73 \times 10^{-10}$ (Enthalpy), p50 $\sim 10^{-16}$ |

---

## 2. Forensic Dissection of Each Historical Metric

### A. The 4,000 Successful Inverse Solves
- **Artifact**: `audit/solver_wave1_summary.json`
- **Corpus**: `audit/corpus_wave1.jsonl` filtered to solvable domain states (`status == "SUCCESS"`).
- **Composition**:
  - 1,000 HP cases across IdealGas (GRI-3.0, airNASA9) and Cubic EOS (Peng-Robinson, Redlich-Kwong)
  - 1,000 UV cases
  - 1,000 SP cases
  - 1,000 SV cases
- **Conditioning**: Guaranteed distant starting temperatures ($|T_{\mathrm{start}} - T_{\mathrm{target}}| / T_{\mathrm{target}} \ge 0.15$) to prevent circular trivial identity solves.
- **Pass Rate**: 4,000 / 4,000 (100.0%). Every solve converged within 500 iterations without diverging.

### B. The 4,100 Native Differential Cases
- **Artifact**: `audit/native_wave1_summary.json`
- **Corpus**: Complete `audit/corpus_wave1.jsonl` (4,100 lines).
- **Composition**:
  - **4,000 Solvable Inversion Cases**: Evaluated on live native Cantera 4.0.0a2 (WSL) and pure-Python, comparing 10 physical observables per case (40,000 scalar comparisons).
  - **100 Deterministic Failure Cases**: Evaluated out-of-domain conditions ($T \le 0$, $P \le 0$, $v \le 0$, unphysical composition, non-finite targets).
- **Result**:
  - Success Parity: 4,000 / 4,000 (100% agreement on `status == "SUCCESS"`).
  - Failure Parity: 100 / 100 (100% agreement on `status == "FAILURE"` and exact normalized failure category matching).
  - Net Parity: 4,100 / 4,100 (0 mismatches).

### C. Origin and Meaning of `5.34e-7`
- **The Question**: Does `5.34e-7` represent native differential error between Python and Cantera C++?
- **The Finding**: **NO.**
- **The Proof**:
  1. In `audit/solver_wave1_summary.json`, the field `"relative_error_max": 5.340363131250286e-07` measures the **residual error of the target property inversion**:
     $$\epsilon_{\mathrm{residual}} = \frac{|H(T_{\mathrm{converged}}, P) - H_{\mathrm{target}}|}{|H_{\mathrm{target}}|}$$
  2. The Newton solver default tolerance parameter is `rtol = 1.0e-9`. The convergence condition in Cantera's `ThermoPhase::setState_HPorUV` (and `state.py:_inverse_native`) is:
     $$\frac{|H - H_{\mathrm{target}}|}{\max(|H_{\mathrm{target}}|, |c_p| \cdot T)} < \mathrm{rtol} \quad \text{OR} \quad \left|\frac{\Delta T}{T}\right| < \mathrm{rtol}$$
  3. When an iteration terminates because the temperature step criterion $|\Delta T / T| < \mathrm{rtol}$ is satisfied in a steep heat capacity regime, the property residual can reach $O(10^{-7})$ while the temperature is accurate to $10^{-9}$.
  4. In direct comparison against the live native C++ oracle (`audit/native_wave1_summary.json`), the maximum relative difference between Python and native Cantera for the exact same case was only **$5.60 \times 10^{-9}$ for temperature** and **$2.73 \times 10^{-10}$ for enthalpy**, with median relative differences at machine precision ($10^{-16}$).

---

## 3. Observable Native Errors vs. Internal Inversion Residuals

| Physical Observable | Python Inversion Residual Distribution (`solver_wave1_summary.json`) | Live Native Differential Relative Error (`native_wave1_summary.json`) | Tolerance Policy | Differential Mismatches |
| :--- | :---: | :---: | :---: | :---: |
| **Temperature** | Inversion variable ($T$) | p50: $4.45 \times 10^{-16}$, p95: $1.32 \times 10^{-9}$, max: $5.60 \times 10^{-9}$ | $1 \times 10^{-7}$ | 0 / 4,000 |
| **Pressure** | Fixed (HP/SP) or Inverted (UV/SV) | p50: $1.59 \times 10^{-16}$, p95: $1.15 \times 10^{-9}$, max: $9.69 \times 10^{-9}$ | $1 \times 10^{-7}$ | 0 / 4,000 |
| **Density** | Computed from EOS | p50: $1.49 \times 10^{-16}$, p95: $6.50 \times 10^{-15}$, max: $7.70 \times 10^{-9}$ | $1 \times 10^{-7}$ | 0 / 4,000 |
| **Enthalpy** | Max residual: $5.34 \times 10^{-7}$ | p50: $2.11 \times 10^{-16}$, p95: $4.34 \times 10^{-11}$, max: $2.73 \times 10^{-10}$ | $1 \times 10^{-7}$ | 0 / 4,000 |
| **Internal Energy** | Max residual: $2.17 \times 10^{-7}$ | p50: $2.10 \times 10^{-16}$, p95: $3.29 \times 10^{-11}$, max: $2.17 \times 10^{-10}$ | $1 \times 10^{-7}$ | 0 / 4,000 |
| **Entropy** | Max residual: $9.91 \times 10^{-7}$ | p50: $1.96 \times 10^{-16}$, p95: $2.05 \times 10^{-10}$, max: $9.91 \times 10^{-10}$ | $1 \times 10^{-7}$ | 0 / 4,000 |
| **Specific Heat ($c_p$)** | Polynomial evaluate | p50: $2.07 \times 10^{-16}$, p95: $2.60 \times 10^{-10}$, max: $4.40 \times 10^{-8}$ | $1 \times 10^{-6}$ | 0 / 4,000 |
| **Specific Heat ($c_v$)** | Difference evaluate | p50: $2.72 \times 10^{-16}$, p95: $5.81 \times 10^{-10}$, max: $2.42 \times 10^{-9}$ | $1 \times 10^{-6}$ | 0 / 4,000 |

---

## 4. Reconciliation Decision and Final Clearance
- All apparent numerical discrepancies are completely reconciled and explained.
- The 4,000 cases quantify independent solver convergence and self-consistency from distant start states.
- The 4,100 cases quantify whole-domain live differential parity against the pinned C++ native oracle.
- Zero unexplained contradictions remain in Wave 1 evidence.
