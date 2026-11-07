# Forensic Audit: Waves 1–3 Current HEAD Dependency Integrity

**Date**: 2026-09-27  
**Repository Working Directory**: `outputs/cantera-python`  
**Git Branch**: `wave/4-multitransport`  
**Pinned Native Source Oracle**: `outputs/cantera-original` at commit `726522be4e2a13454d8415b7ef799d621f665cf3`  
**Historical Frozen Manifest**: `audit/thermodynamic_solver_waves_1_3_frozen_manifest.json` (frozen at commit `147b4c7`)  

---

## 1. Executive Summary

A forensic SHA-256 hash comparison was conducted across all pure-Python implementation files relevant to Solver Waves 1–3 (Thermodynamic Inverse Solvers, Cubic EOS Root Solvers, and TPX Pure/Real Fluid Solver Stack) between the current HEAD on branch `wave/4-multitransport` and the historical frozen manifest `audit/thermodynamic_solver_waves_1_3_frozen_manifest.json`.

**Key Finding**:
- 9 of the 10 core implementation files recorded in the historical manifest are **100% bit-exact identical** (matching SHA-256 hashes).
- Exactly one file, `cantera_python/solution.py`, has a different SHA-256 hash (`861ad9f0abd2fa7af5af11e8991048ea9adcdbbbde8d55b950ff7a547ce1cca2` vs frozen `3b50978c5dfa6e47236b11080f2b5046da39a55f56a236bca5d0ad22d04011f8`).
- Detailed source AST and diff inspection demonstrates that the change in `solution.py` was introduced during Wave 4 multicomponent transport integration on branch `wave/4-multitransport`. The modification affects **only transport delegation properties** (`viscosity`, `species_viscosities`, and `thermal_conductivity`) to safely handle callable vs property getters.
- **Zero thermodynamic methods, inverse solvers, Cubic EOS routines, or TPX fluid equations in Waves 1–3 are affected by this change.**

---

## 2. Complete File Hash Inventory

| File Path | Historical Frozen SHA-256 | Current HEAD SHA-256 | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `cantera_python/state.py` | `6fe015a744b915edaf2f1a8e3e538241e9cf8e28bf3aae2bd481464cf03b9bdc` | `6fe015a744b915edaf2f1a8e3e538241e9cf8e28bf3aae2bd481464cf03b9bdc` | **MATCH** | Wave 1 Newton inverse solver engine |
| `cantera_python/thermo.py` | `1a034d833d2de4c6e9221b747a12319dd60db377f50c600c24781c0aaf4c1b68` | `1a034d833d2de4c6e9221b747a12319dd60db377f50c600c24781c0aaf4c1b68` | **MATCH** | ThermoPhase core interface |
| `cantera_python/ideal_gas.py` | `98445f219100be1d3417451e40a759c6c17b8823592f7b994dfae2087e0ca49d` | `98445f219100be1d3417451e40a759c6c17b8823592f7b994dfae2087e0ca49d` | **MATCH** | IdealGasPhase implementation |
| `cantera_python/phases.py` | `03f3cab9aa9a220b6f7c818460d1e6d33f59018866bb904f9ecf04d436f3f365` | `03f3cab9aa9a220b6f7c818460d1e6d33f59018866bb904f9ecf04d436f3f365` | **MATCH** | Phase hierarchy & TPX phase bindings |
| `cantera_python/cubic_eos.py` | `435e62df706c6700b54b3e6a8139421673749b2a5722a18c5d8124d8a84a3054` | `435e62df706c6700b54b3e6a8139421673749b2a5722a18c5d8124d8a84a3054` | **MATCH** | Wave 2 Peng-Robinson & Redlich-Kwong |
| `cantera_python/tpx.py` | `20e9b056719f4c664c4fd0c915c3c1a81acaed687acba898bd8efbbf39726e06` | `20e9b056719f4c664c4fd0c915c3c1a81acaed687acba898bd8efbbf39726e06` | **MATCH** | Wave 3 8 pure fluid TPX stack |
| `cantera_python/tpx_exact.py` | `d61011adc5ff601decfb1196e8a022118da964d63ee9d78c918ef47f227d4c87` | `d61011adc5ff601decfb1196e8a022118da964d63ee9d78c918ef47f227d4c87` | **MATCH** | Wave 3 exact thermodynamic relations |
| `cantera_python/numerics.py` | `69d58120fa2f9377b0105cf18d858ec31be4d20b6a7161a429e19098d5ecb284` | `69d58120fa2f9377b0105cf18d858ec31be4d20b6a7161a429e19098d5ecb284` | **MATCH** | Linear algebra & polynomial utilities |
| `cantera_python/state_revision.py` | `b00b2df4d7392be6d7e4a5bad2edb36f36430bd4690ed11f5869548957bfa77b` | `b00b2df4d7392be6d7e4a5bad2edb36f36430bd4690ed11f5869548957bfa77b` | **MATCH** | Thermodynamic state cache manager |
| `cantera_python/constants.py` | *(identical to 147b4c7)* | `43b17585371daa7a54a93feed5eff32d3c490b8f970a22e6a671d48fa1bc6cb4` | **UNMODIFIED** | Physical constants ($R, N_A, \text{etc.}$) |
| `cantera_python/units.py` | *(identical to 147b4c7)* | `671675843c76fa50706bebe84f550dea2a30c6266b732b176c89d3cc4bce10f4` | **UNMODIFIED** | Unit conversion utilities |
| `cantera_python/elements.py` | *(identical to 147b4c7)* | `93f3433ed80e0041e630e60280850ae6c505fdfce25beea16df76b2059af1b97` | **UNMODIFIED** | Chemical element atomic weights |
| `cantera_python/solution.py` | `3b50978c5dfa6e47236b11080f2b5046da39a55f56a236bca5d0ad22d04011f8` | `861ad9f0abd2fa7af5af11e8991048ea9adcdbbbde8d55b950ff7a547ce1cca2` | **DIFFERENT** | Transport delegation properties updated |

---

## 3. Forensic Analysis of `cantera_python/solution.py` Diff

### A. Git History and Commit Provenance
Commit `147b4c7` is the historical commit that established the frozen snapshot for Waves 1–3 (`chore: freeze solver waves 1-3 thermodynamic foundation`).
Subsequent changes on the active branch `wave/4-multitransport` modified `cantera_python/solution.py` and `cantera_python/transport.py`.

### B. Exact Line-by-Line Code Diff
The complete diff between `147b4c7` and current HEAD in `cantera_python/solution.py` is restricted to lines 725–746:

```diff
diff --git a/cantera_python/solution.py b/cantera_python/solution.py
index 4e137a1..f5b02bd 100644
--- a/cantera_python/solution.py
+++ b/cantera_python/solution.py
@@ -725,14 +725,25 @@ class Solution:
 
     @property
     def viscosity(self) -> float:
-        return self._transport.viscosity if self._transport else 0.0
+        if not self._transport:
+            return 0.0
+        v = self._transport.viscosity
+        return float(v() if callable(v) else v)
 
     def species_viscosities(self):
-        return self._transport.species_viscosities() if self._transport else ()
+        if not self._transport:
+            return ()
+        sv = getattr(self._transport, 'species_viscosities', None)
+        return sv() if callable(sv) else (sv if sv is not None else ())
 
     @property
     def thermal_conductivity(self) -> float:
-        return self._transport.thermal_conductivity if self._transport else 0.0
+        if not self._transport:
+            return 0.0
+        tc = getattr(self._transport, 'thermal_conductivity', None)
+        if tc is None:
+            tc = getattr(self._transport, 'thermalConductivity', None)
+        return float(tc() if callable(tc) else tc)
 
     @property
     def mix_diff_coeffs(self):
```

### C. Execution Path and Scientific Scope Analysis
1. **Module Independence**:
   - `cantera_python/state.py` does not import `Solution`.
   - `cantera_python/thermo.py` does not import `Solution`.
   - `cantera_python/ideal_gas.py` does not import `Solution`.
   - `cantera_python/phases.py` does not import `Solution`.
   - `cantera_python/cubic_eos.py` does not import `Solution`.
   - `cantera_python/tpx.py` does not import `Solution`.
   - `cantera_python/numerics.py` does not import `Solution`.

2. **Wave 1 Execution Paths**:
   - Wave 1 inverse state setters (`setState_HP`, `setState_UV`, `setState_SP`, `setState_SV`) are methods of `IdealGasMixture` / `IdealGasPhase` in `ideal_gas.py` and `state.py`.
   - When called via a composite `Solution` instance (`sol.HP = h, p`), `Solution.__getattr__` delegates directly to `self._thermo.setState_HP(...)`.
   - No lines in `solution.py` alter enthalpy, internal energy, entropy, temperature, pressure, density, heat capacity, or the Newton-bisection solver loop.

3. **Wave 2 Execution Paths**:
   - Wave 2 Cubic EOS solvers (`solveCubic`, `calculateAB`, `densityCalc`, `setState_HP/SP/UV/SV`) reside entirely in `cantera_python/cubic_eos.py`.
   - Cubic EOS root selection and continuation branches do not pass through `solution.py` transport getters.

4. **Wave 3 Execution Paths**:
   - Wave 3 TPX real fluids (`Water`, `CarbonDioxide`, `Nitrogen`, `Oxygen`, `Hydrogen`, `Methane`, `Heptane`, `HFC134a`) reside entirely in `cantera_python/tpx.py`.
   - All 14 property-pair inversions, saturation dome evaluations, quality calculations, and `_set_xy_source` solvers execute inside `tpx.py` and `tpx_exact.py`.

5. **Impact Verdict**:
   - The modification in `cantera_python/solution.py` is **100% confined to Wave 4 transport property resolution**.
   - It introduces zero scientific divergence, numerical perturbation, or interface change for Waves 1–3.

---

## 4. Historical Freeze vs. Current-HEAD Reconciliation

- **Historical Freeze**: Documented in `audit/thermodynamic_solver_waves_1_3_frozen_manifest.json` at git commit `147b4c7`. That manifest accurately recorded the implementation hashes as they existed at freeze time.
- **Current HEAD**: Because `solution.py` was updated for Wave 4, claiming the historical manifest describes current HEAD would violate cryptographic provenance.
- **Resolution**:
  - The historical manifest `audit/thermodynamic_solver_waves_1_3_frozen_manifest.json` is preserved as a permanent historical audit artifact.
  - A new current-HEAD recertification manifest, `audit/thermodynamic_solver_waves_1_3_current_head_manifest.json`, will be emitted upon successful completion of the current-HEAD recertification campaign.
