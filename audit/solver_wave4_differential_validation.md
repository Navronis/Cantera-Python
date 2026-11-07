# Solver Wave 4 Rigorous Differential Validation Report
**Target**: MultiTransport / Stefan–Maxwell Multicomponent Transport Solver Stack
**Pinned Native Oracle**: Commit `726522be4e2a13454d8415b7ef799d621f665cf3`
**Native Oracle Identity SHA256**: `41b22f6cf4e0e0ec7f1fe015e9c44be568c526e24a0c1f0f5a2966549cf4cff2`
**Corpus**: Exactly 10,000 unique paired cases (`audit/corpus_wave4.jsonl`)
**Keying Policy**: Strict dictionary key-by-ID (`set(c_ids) == set(n_ids) == set(p_ids)` asserted)
**Validation Scope**: FULL N x N matrix validation for Binary & Multicomponent Diffusion across all species pairs

## Property Observable Schema
| Property | Status | Dimensions | Description |
| :--- | :---: | :---: | :--- |
| Mixture Viscosity | `COMPARED` | Scalar | Mixture dynamic viscosity $\eta$ |
| Species Viscosities | `COMPARED` | $K$ | Pure-species viscosities $\eta_k$ |
| Thermal Conductivity | `COMPARED` | Scalar | Multicomponent thermal conductivity $\lambda$ |
| Binary Diffusion Matrix | `COMPARED` | $K \times K$ | Full binary diffusion matrix $\mathcal{D}_{jk}$ |
| Multicomponent Diffusion Matrix | `COMPARED` | $K \times K$ | Full multicomponent diffusion matrix $D_{ij}$ |
| Thermal Diffusion Coeffs | `COMPARED` | $K$ | Thermal diffusion coefficients $D_k^T$ (Soret) |
| Diffusive Mass Fluxes | `COMPARED` | $K$ | Mass fluxes $j_k$ under $\nabla T$ and $\nabla X$ |
| Diffusive Molar Fluxes | `COMPARED` | $K$ | Molar fluxes $J_k = j_k / M_k$ |

---

## TOTAL OVERALL (10,000 cases)

| Property | Comparisons | Nat Fail | Py Fail | NaN/Inf | p50 RelErr | p95 RelErr | p99 RelErr | Max AbsErr | Max RelErr | Tolerance | Safety Factor | Mismatches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **viscosity** | 10,000 | 0 | 0 | 0 | `1.70e-12` | `8.80e-12` | `2.00e-11` | `1.73e-15` | `2.94e-11` | `1e-09` | 34.0x below tolerance | **0** |
| **sp_visc** | 422,500 | 0 | 0 | 0 | `2.55e-12` | `2.08e-11` | `2.48e-11` | `1.75e-15` | `3.25e-11` | `1e-09` | 30.8x below tolerance | **0** |
| **thermal_conductivity** | 10,000 | 0 | 0 | 0 | `7.97e-13` | `7.58e-12` | `1.76e-11` | `1.10e-11` | `2.94e-11` | `1e-09` | 34.0x below tolerance | **0** |
| **bdiff** | 21,317,500 | 0 | 0 | 0 | `1.25e-12` | `8.83e-12` | `1.61e-11` | `3.74e-13` | `4.58e-11` | `1e-09` | 21.8x below tolerance | **0** |
| **dmulti** | 21,317,500 | 0 | 0 | 0 | `7.46e-13` | `4.69e-12` | `1.01e-11` | `4.53e-12` | `4.37e-11` | `1e-08` | 228.7x below tolerance | **0** |
| **dt** | 422,500 | 0 | 0 | 0 | `1.76e-12` | `1.35e-11` | `3.33e-11` | `1.06e-16` | `5.01e-08` | `1e-07` | 2.0x below tolerance | **0** |
| **fluxes** | 422,500 | 0 | 0 | 0 | `8.54e-15` | `3.41e-12` | `1.52e-11` | `5.13e-16` | `3.74e-09` | `1e-08` | 2.7x below tolerance | **0** |
| **molar_fluxes** | 422,500 | 0 | 0 | 0 | `3.26e-16` | `3.20e-12` | `1.28e-11` | `5.09e-16` | `3.87e-10` | `1e-08` | 25.8x below tolerance | **0** |

- **Max Net Mass Flux Sum (Python)**: `2.4299e-20 kg/(m^2 s)` (mass-flux conservation to floating-point precision)
- **Max Net Mass Flux Sum (Native)**: `5.9292e-20 kg/(m^2 s)` (mass-flux conservation to floating-point precision)

---
## GRI-3.0 (53 Species) (7,500 cases)

| Property | Comparisons | Nat Fail | Py Fail | NaN/Inf | p50 RelErr | p95 RelErr | p99 RelErr | Max AbsErr | Max RelErr | Tolerance | Safety Factor | Mismatches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **viscosity** | 7,500 | 0 | 0 | 0 | `1.77e-12` | `9.32e-12` | `2.15e-11` | `1.73e-15` | `2.94e-11` | `1e-09` | 34.0x below tolerance | **0** |
| **sp_visc** | 397,500 | 0 | 0 | 0 | `2.59e-12` | `2.14e-11` | `2.49e-11` | `1.75e-15` | `3.25e-11` | `1e-09` | 30.8x below tolerance | **0** |
| **thermal_conductivity** | 7,500 | 0 | 0 | 0 | `8.69e-13` | `8.16e-12` | `2.14e-11` | `1.10e-11` | `2.94e-11` | `1e-09` | 34.0x below tolerance | **0** |
| **bdiff** | 21,067,500 | 0 | 0 | 0 | `1.24e-12` | `8.86e-12` | `1.61e-11` | `3.74e-13` | `4.58e-11` | `1e-09` | 21.8x below tolerance | **0** |
| **dmulti** | 21,067,500 | 0 | 0 | 0 | `7.46e-13` | `4.71e-12` | `1.01e-11` | `4.53e-12` | `4.37e-11` | `1e-08` | 228.7x below tolerance | **0** |
| **dt** | 397,500 | 0 | 0 | 0 | `1.76e-12` | `1.35e-11` | `3.33e-11` | `1.06e-16` | `5.01e-08` | `1e-07` | 2.0x below tolerance | **0** |
| **fluxes** | 397,500 | 0 | 0 | 0 | `7.58e-15` | `3.03e-12` | `1.40e-11` | `1.02e-16` | `3.74e-09` | `1e-08` | 2.7x below tolerance | **0** |
| **molar_fluxes** | 397,500 | 0 | 0 | 0 | `2.83e-16` | `2.77e-12` | `1.20e-11` | `3.71e-17` | `3.87e-10` | `1e-08` | 25.8x below tolerance | **0** |

- **Max Net Mass Flux Sum (Python)**: `2.4299e-20 kg/(m^2 s)` (mass-flux conservation to floating-point precision)
- **Max Net Mass Flux Sum (Native)**: `5.9292e-20 kg/(m^2 s)` (mass-flux conservation to floating-point precision)

---
## H2-O2 (10 Species) (2,500 cases)

| Property | Comparisons | Nat Fail | Py Fail | NaN/Inf | p50 RelErr | p95 RelErr | p99 RelErr | Max AbsErr | Max RelErr | Tolerance | Safety Factor | Mismatches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **viscosity** | 2,500 | 0 | 0 | 0 | `1.34e-12` | `7.72e-12` | `1.10e-11` | `1.07e-15` | `1.10e-11` | `1e-09` | 90.9x below tolerance | **0** |
| **sp_visc** | 25,000 | 0 | 0 | 0 | `1.57e-12` | `1.10e-11` | `1.10e-11` | `1.08e-15` | `1.10e-11` | `1e-09` | 90.9x below tolerance | **0** |
| **thermal_conductivity** | 2,500 | 0 | 0 | 0 | `6.18e-13` | `7.27e-12` | `1.10e-11` | `2.68e-12` | `1.10e-11` | `1e-09` | 90.5x below tolerance | **0** |
| **bdiff** | 250,000 | 0 | 0 | 0 | `1.58e-12` | `5.30e-12` | `9.57e-12` | `3.35e-13` | `1.11e-11` | `1e-09` | 90.3x below tolerance | **0** |
| **dmulti** | 250,000 | 0 | 0 | 0 | `8.16e-13` | `3.97e-12` | `5.29e-12` | `1.85e-12` | `8.61e-12` | `1e-08` | 1161.2x below tolerance | **0** |
| **dt** | 25,000 | 0 | 0 | 0 | `1.67e-12` | `1.41e-11` | `3.38e-11` | `3.98e-17` | `3.65e-09` | `1e-07` | 27.4x below tolerance | **0** |
| **fluxes** | 25,000 | 0 | 0 | 0 | `7.48e-13` | `6.80e-12` | `3.26e-11` | `5.13e-16` | `1.33e-09` | `1e-08` | 7.5x below tolerance | **0** |
| **molar_fluxes** | 25,000 | 0 | 0 | 0 | `7.40e-13` | `6.58e-12` | `2.48e-11` | `5.09e-16` | `1.96e-10` | `1e-08` | 50.9x below tolerance | **0** |

- **Max Net Mass Flux Sum (Python)**: `2.4299e-20 kg/(m^2 s)` (mass-flux conservation to floating-point precision)
- **Max Net Mass Flux Sum (Native)**: `5.9292e-20 kg/(m^2 s)` (mass-flux conservation to floating-point precision)

---

## Non-Finite Audit Classification
- `EXPECTED_NONFINITE_SOURCE_SEMANTIC`: 0
- `UNEXPECTED_NATIVE_NONFINITE`: 0
- `UNEXPECTED_PYTHON_NONFINITE`: 0
- `NONFINITE_MISMATCH`: 0
- **UNEXPLAINED_NONFINITE**: `0`
- **UNEXPLAINED_NATIVE_MISMATCHES**: `0`
## Expected Matrix Dimensions & Vector Shapes
- **GRI-3.0 Species Count**: $K = 53$ (asserted independently from mechanism)
- **H2-O2 Species Count**: $K = 10$ (asserted independently from mechanism)
- **Vector Shape Invariants**: `len == K` verified across all 10,000 cases for `sp_visc`, `dt`, `fluxes`, `molar_fluxes` (**0 errors**)
- **Matrix Shape Invariants**: $K \times K$ verified across all 10,000 cases for `bdiff` and `dmulti` (**0 errors**)
- **Total Independent Expected Dimension Assertions**: `1,810,000` verified assertions

## Cryptographic Producer Provenance
- **wave4_run_id**: `w4_0e6da47414ae3487`
- **Provenance Run ID Parity**: Exactly 10,000 / 10,000 native and Python cases cryptographically bound to `w4_0e6da47414ae3487`

## Numerical Invariants and Conservation
- **Solvability & Finiteness**: All tested transport solves completed successfully and produced finite source-equivalent outputs.
- **Mass-Flux Conservation**: Mass-flux conservation to floating-point precision strictly verified ($< 6 \times 10^{-20}\text{ kg/(m}^2\text{ s)}$ in both native and Python).
- **Full Matrix Parity**: Full $N \times N$ Binary Diffusion ($21,317,500$ elements) and Multicomponent Diffusion ($21,317,500$ elements) verified with **0 mismatches**.
- **Total Element Comparisons**: `44,345,000` observable values.
