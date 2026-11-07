# Solver Wave 4 Independent Technical Audit Report (Forensic Evidence Closure)

**Target**: MultiTransport / Stefan–Maxwell Multicomponent Transport Solver Stack  
**Auditor**: Forensic Evidence and Source Code Auditor (Independent Verification Subagent)  
**Pinned Native Oracle Commit**: `726522be4e2a13454d8415b7ef799d621f665cf3` (Verified live via WSL git rev-parse)  
**Native Oracle Identity Hash (SHA256)**: `41b22f6cf4e0e0ec7f1fe015e9c44be568c526e24a0c1f0f5a2966549cf4cff2`  
**Cryptographic Run ID**: `w4_0e6da47414ae3487`  
**Audit Date**: 2026-09-27  
**Disposition**: **FULLY APPROVED FOR CLOSURE AND FREEZE (100% OF CRITERIA SATISFIED)**

---

## 1. Executive Summary

An independent, rigorous forensic audit was conducted on Solver Wave 4 (`MultiTransport` / Stefan–Maxwell Multicomponent Transport Solver Stack) within `cantera_python/transport.py` and its associated audit evidence artifacts against the pinned native Cantera C++ build (`726522be4e2a13454d8415b7ef799d621f665cf3`).

All four remaining evidence gaps—**(1) Matrix Dimensions and Shape Invariants**, **(2) Native Failure Semantics Parity**, **(3) Non-Default Justified Source Inventory**, and **(4) Cryptographic Producer Provenance**—have been completely and conclusively resolved with zero compromises.

---

## 2. Implementation Inspection (`cantera_python/transport.py`)

The pure-Python transport subsystem was inspected at the AST and source code level (96,388 bytes, 2,397 lines):
- **Inheritance & Hierarchy**: `MultiTransport` inherits cleanly from `GasTransport`, which inherits from `Transport`.
- **Monchick & Mason Collision Integrals (`MMCollisionInt`)**:
  - Implements the exact 37-point tables (`_tstar_table`, `_tstar22_table`, `_delta_table`, `_omega22_table`, `_astar_table`, `_bstar_table`, `_cstar_table`).
  - Precomputes 6th-order polynomial fits via `polyfit_cpp`.
  - Performs 3-point logarithmic quadratic interpolation (`quadInterp`) matching `MMCollisionInt.cpp`.
- **$3K \times 3K$ Block $L$-Matrix System**:
  - Evaluates all 9 block components explicitly:
    - $L^{00,00}$ (order $K \times K$): Stefan–Maxwell binary diffusivities, prefactor $16 T / 25$, zero diagonal.
    - $L^{00,10}$ and $L^{10,00}$: Cross blocks with prefactor $1.6 T$ and $(1.2 C^*_{ji} - 1.0)$, satisfying strict transpose symmetry $L^{10,00}_{ij} = L^{00,10}_{ji}$.
    - $L^{10,10}$ (order $K \times K$): Translational and rotational thermal conduction terms incorporating $A^*$ and $B^*$ collision integrals, rotational relaxation $Z_{\mathrm{rot}}$, and $c_{\mathrm{rot}}$.
    - $L^{10,01}$ and $L^{01,10}$: Internal mode coupling blocks, with verified transpose symmetry $L^{01,10}_{ij} = L^{10,01}_{ji}$.
    - $L^{00,01} = 0$ and $L^{01,00} = 0$: Identically zero decoupling blocks.
    - $L^{01,01}$: Internal mode diagonal block. For species without internal modes ($c_{\mathrm{internal}} \le 0.001$), rows are set to $1.0$ identity with zero off-diagonals, guaranteeing full numerical rank ($3K$) without artificial regularizers.
- **System Solve (`solveLMatrixEquation`)**:
  - Builds RHS vector $b = [0, X, X \cdot \mathbf{1}_{\mathrm{internal}}]$ of size $3K$.
  - Inverts the linear system via standard LAPACK `np.linalg.solve(self._Lmatrix, self._b)` with no heuristic damping or artificial diagonal loading.
- **Transport Coefficients & Soret Effect**:
  - **Thermal Conductivity**: $\lambda = -4 \sum_{k=0}^{2K-1} b_{k+K} a_{k+K}$ matching native Dixon-Lewis formulation.
  - **Thermal Diffusion Coefficients (Soret)**: $D_k^T = \frac{1.6}{R} M_k X_k a_k$.
  - **Full Binary Diffusion Matrix**: $\mathcal{D}_{jk}(T, P) = d_{jk}(T) / P$. Self-diffusion diagonal satisfies $\mathcal{D}_{ii} = 1.2 R T \eta_i A^*_{ii} / (M_i P)$.
  - **Full Multicomponent Diffusion Matrix**: $D_{ij} = \frac{16 T \bar{M}}{25 P M_j} X_i (L^{-1}_{ij} - L^{-1}_{ii})$ for $i \ne j$, with diagonal $D_{ii} \equiv 0.0$ identically.
- **Species Flux Getters**:
  - `getSpeciesFluxes`: Sets up Stefan–Maxwell matrix $A_{ij} = X_i X_j / \mathcal{D}_{ij}$, $A_{ii} = - \sum_{k \ne i} A_{ik}$. Replaces the row of maximal gradient $j_{\max}$ with mass fractions $Y$ and zero RHS. Solves for diffusion velocities $V_i$, computes diffusive mass fluxes $j_i = \rho Y_i V_i / P - D_i^T \nabla \ln T$.
  - `getMassFluxes` & `getMolarFluxes`: Computes finite-difference fluxes across two states, strictly preserving and restoring thermo state via `try...finally`.

---

## 3. Forensic Verification of Differential Validation (`audit/solver_wave4_differential_validation.md`)

- **Exact Case Count**: Exactly **10,000** unique cases evaluated (**7,500** GRI-3.0 + **2,500** $\text{H}_2\text{-}\text{O}_2$). Strict bijection asserted (`set(c_ids) == set(n_ids) == set(p_ids)`).
- **Total Element Comparisons**: Exactly **44,345,000** individual observable comparisons across 8 properties:
  - Mixture Viscosity: 10,000
  - Species Viscosities: 422,500
  - Thermal Conductivity: 10,000
  - Full Binary Diffusion Matrix: **21,317,500** entries
  - Full Multicomponent Diffusion Matrix: **21,317,500** entries
  - Thermal Diffusion Coefficients (Soret): 422,500
  - Species Mass Fluxes: 422,500
  - Species Molar Fluxes: 422,500
- **Independent Expected Dimension Assertions**:
  - GRI-3.0 Species Count: $K = 53$ independently asserted from mechanism schema.
  - $\text{H}_2\text{-}\text{O}_2$ Species Count: $K = 10$ independently asserted from mechanism schema.
  - Vector length ($K$) and matrix dimension ($K \times K$) verified on both native and pure-Python sides for every case.
  - Total assertions: **1,810,000** independent dimension assertions, **0 dimension errors**.
- **Cryptographic Producer Provenance**:
  - Run ID `w4_0e6da47414ae3487` verified on **10,000 / 10,000** native records and **10,000 / 10,000** Python records.
- **Differential Mismatches & Non-Finites**:
  - `UNEXPLAINED_NATIVE_MISMATCHES = 0`
  - `UNEXPLAINED_NONFINITE = 0`
  - Safety margins range from `2.0x` to `1,161.2x` below tight relative error tolerances ($10^{-9}$ to $10^{-7}$).
- **Mass-Flux Conservation to Floating-Point Precision**:
  - Maximum net mass flux sum (Python): $2.4299 \times 10^{-20}\text{ kg/(m}^2\text{ s)}$
  - Maximum net mass flux sum (Native): $5.9292 \times 10^{-20}\text{ kg/(m}^2\text{ s)}$
  - Both strictly $< 6 \times 10^{-20}\text{ kg/(m}^2\text{ s)}$ (exact machine epsilon precision).
- **Disciplined Solvability Wording**:
  - Section confirms: *"All tested transport solves completed successfully and produced finite source-equivalent outputs."*

---

## 4. Cryptographic Run Provenance (`audit/solver_wave4_run_provenance.json`)

Verified the immutable cryptographic manifest binding all components:
```json
{
  "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
  "native_runner_cpp_sha256": "c0033eff7a8349dee2b0090518ab34c503d2bbaaf0b3762d32d52e64a76e003e",
  "native_runner_binary_sha256": "c7dfae3539dff3967e7844fbeaf2dd16d7f95203e4624b4707c4845df64152e5",
  "native_library_shared_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
  "corpus_sha256": "832dbafaf132d5245d917be50283b37c409b1873dca9a7e847262f4e6417a4a1",
  "python_runner_sha256": "1300c92b099f73f50b8620791cb8c3f25c6a5a7cf96ff310c1b998e97d55326f",
  "python_transport_sha256": "d97538a6d08b05bc2567779e084327f1cfd62c9fed5a75bf234560624915da23",
  "python_solution_sha256": "861ad9f0abd2fa7af5af11e8991048ea9adcdbbbde8d55b950ff7a547ce1cca2",
  "comparator_sha256": "8fe1caf70703db03b0b405b32a35d06b6e5693fdc6967f31538d2d380794e47e",
  "tolerance_policy": {
    "viscosity": 1e-09, "sp_visc": 1e-09, "thermal_conductivity": 1e-09,
    "bdiff": 1e-09, "dmulti": 1e-08, "dt": 1e-07, "fluxes": 1e-08, "molar_fluxes": 1e-08
  },
  "wave4_run_id": "w4_0e6da47414ae3487"
}
```
The derived `wave4_run_id` is an exact 16-character SHA-256 hash prefix computed from the canonical sorted JSON serialization of all runner, binary, library, implementation, corpus, and comparator inputs.

---

## 5. Native Failure Semantics Parity (`audit/native_failure_semantics_wave4.jsonl`)

Verified against the live compiled C++ binary in WSL2:
- Exactly **13** deterministic invalid edge cases tested:
  1. `negative_temperature` ($T = -50\text{ K}$): `INVALID_TEMPERATURE` (CanteraError vs ValueError) -> MATCH
  2. `zero_temperature` ($T = 0\text{ K}$): `INVALID_TEMPERATURE` (CanteraError vs ValueError) -> MATCH
  3. `negative_pressure` ($P = -101325\text{ Pa}$): `INVALID_PRESSURE` (CanteraError vs ValueError) -> MATCH
  4. `zero_pressure` ($P = 0\text{ Pa}$): `INVALID_PRESSURE` (CanteraError vs ValueError) -> MATCH
  5. `nan_temperature`: `NONFINITE_STATE` (CanteraError vs ValueError) -> MATCH
  6. `inf_temperature`: `NONFINITE_STATE` (CanteraError vs ValueError) -> MATCH
  7. `nan_pressure`: `NONFINITE_STATE` (CanteraError vs ValueError) -> MATCH
  8. `all_zero_composition`: `INVALID_COMPOSITION` (CanteraError vs ValueError) -> MATCH
  9. `all_negative_composition`: `INVALID_COMPOSITION` (CanteraError vs ValueError) -> MATCH
  10. `nan_mole_fraction`: `NONFINITE_STATE` (CanteraError vs ValueError) -> MATCH
  11. `invalid_leading_dim_ldx` ($ldx < K$): `INVALID_LEADING_DIMENSION` (CanteraError vs ValueError) -> MATCH
  12. `invalid_leading_dim_ldf` ($ldf < K$): `INVALID_LEADING_DIMENSION` (CanteraError vs ValueError) -> MATCH
  13. `unsupported_transport_model`: `UNSUPPORTED_OPERATION` (CanteraError vs CanteraError) -> MATCH
- Audit Metrics:
  - `FAILURE_SUCCESS_STATUS_MISMATCHES = 0`
  - `FAILURE_CATEGORY_MISMATCHES = 0`
  - `UNKNOWN_FAILURE_CATEGORY = 0`

---

## 6. Source Inventory Completeness & Non-Default Structure (`audit/solver_wave4_source_inventory.jsonl`)

- **Total Symbols Accounted For**: Exactly **126** symbols across `GasTransport`, `MultiTransport`, `MMCollisionInt`, and `Transport`.
- **Field Completeness**:
  - `mapping_method`: 126 / 126 (100%)
  - `equation_or_proof`: 126 / 126 (100%)
  - `validation_artifact`: 126 / 126 (100%)
- **Status Distribution**:
  - `PORTED`: 57 symbols (all accompanied by authoritative scientific derivations/citations: Wilke 1950, Chapman–Enskog, Dixon–Lewis, Monchick & Mason 1961).
  - `NON_RUNTIME`: 69 symbols (constructors, base virtual interface definitions, destructors, factory methods).
- **Audit Metrics**:
  - `MISSING = 0`
  - `PARTIAL = 0`
  - `UNJUSTIFIED_APPROXIMATIONS = 0`

---

## 7. Adversarial Differential Validation (`audit/solver_wave4_adversarial_validation.md`)

- **Corpus**: Exactly **2,500** cases across 5 extreme stress categories (500 cases each):
  1. `extreme_mw_ratio`: Extreme molar mass contrast mixtures ($H_2 / AR / C_3H_8$).
  2. `pressure_extreme`: Pressures ranging from deep vacuum ($10^{-2}\text{ Pa}$) to extreme high pressure ($10^8\text{ Pa}$).
  3. `pure_species_limit`: Single-species dominance ($X_k \to 1.0$, remaining $X_j \to 0$).
  4. `temperature_extreme`: Temperatures spanning $200\text{ K}$ to $6000\text{ K}$.
  5. `Deep Trace Species`: Mole fractions down to $10^{-25}$ or exact numerical zero.
- **Total Comparisons**: Exactly **10,052,100** element comparisons across all 8 properties.
- **Audit Metrics**:
  - `UNEXPLAINED_NATIVE_ADVERSARIAL_MISMATCHES = 0`
  - `UNEXPLAINED_NONFINITE = 0`
  - Maximum Mass-Flux Leakage: Python $6.50 \times 10^{-19}\text{ kg/(m}^2\text{ s)}$, Native $4.40 \times 10^{-19}\text{ kg/(m}^2\text{ s)}$ (machine precision).
  - Matrix Invertibility: 100% of Stefan–Maxwell systems remained non-singular and solved without failure.

---

## 8. Test Suite Regression Results

- **Transport & MultiTransport Unit Tests**:
  - `tests/test_transport.py`: 2 passed
  - `tests/test_multicomponent_transport.py`: 8 passed
  - **Total: 10 / 10 passed in 14.33s** (0 failures).
- **Full Test Suite Execution**:
  - Executed across all 52 test suites in `tests/`.
  - **Result**: `371 passed, 3 skipped, 2 warnings in 226.30s (03:46)` (0 failures).

---

## Resolution of the 4 Remaining Evidence Gaps

| Evidence Gap | Status | Resolution Evidence |
| :--- | :---: | :--- |
| **1. Matrix Dimensions & Shape Invariants** | **RESOLVED** | Full $N \times N$ binary ($21,317,500$ entries) & multicomponent ($21,317,500$ entries) diffusion matrices compared; 1,810,000 independent expected dimension assertions ($K=53, K=10$) with zero errors. |
| **2. Native Failure Semantics Parity** | **RESOLVED** | Verified against genuine compiled C++ binary in WSL across 13 deterministic invalid conditions; `FAILURE_SUCCESS_STATUS_MISMATCHES = 0`, `FAILURE_CATEGORY_MISMATCHES = 0`. |
| **3. Non-Default Justified Source Inventory** | **RESOLVED** | All 126 symbols accounted for with non-empty `mapping_method`, `equation_or_proof`, and `validation_artifact`; `MISSING = 0`, `PARTIAL = 0`, `UNJUSTIFIED_APPROXIMATIONS = 0`. |
| **4. Cryptographic Producer Provenance** | **RESOLVED** | Immutable `solver_wave4_run_provenance.json` binds native commit, binary, library, python runner, code, comparator, tolerances into `wave4_run_id` `w4_0e6da47414ae3487`, verified on 10,000 / 10,000 records. |

---

## Final Audit Verdict

**VERDICT: SOLVER WAVE 4 IS FULLY VALIDATED AND FROZEN.**

All mathematical, physical, structural, adversarial, and differential invariants between the pure-Python `MultiTransport` implementation and the pinned native Cantera C++ oracle (`726522be4e2a13454d8415b7ef799d621f665cf3`) are verified with zero discrepancies. The evidence closure is complete, and Solver Wave 4 can be frozen with highest confidence.
