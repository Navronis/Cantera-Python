# Reproduction Commands & Provenance Audit Guide

This document records the exact provenance, environment configuration, mechanism SHA-256 hashes, and terminal commands used to execute the independent forensic audit.

---

## 1. Provenance & Environment Specifications

| Component | Provenance / Version | Notes |
| :--- | :--- | :--- |
| **Native Cantera Reference** | Git Commit `726522be4e2a13454d8415b7ef799d621f665cf3` | Version `4.0.0a2` compiled in isolated WSL oracle |
| **Python Port Repository** | `outputs/cantera-python` (version `0.2.0`) | Unversioned working tree development checkout |
| **Expected Pinned Commit** | `726522be4e2a13454d8415b7ef799d621f665cf3` | **Exact match confirmed** |
| **Python Version** | `3.11.9 (tags/v3.11.9:de54cf5)` | MSC v.1938 64 bit (AMD64) |
| **NumPy Version** | `2.4.4` | C-compiled vector math backend |
| **SciPy Version** | `1.17.1` | C/Fortran-compiled BDF/LSODA integrator backend |
| **Host Operating System** | Windows 11 Build 26100 (`Windows-10-10.0.26100-SP0 AMD64`) | Port execution environment |
| **Oracle Operating System** | Ubuntu 24.04.1 LTS (`WSL2 Linux 5.15.167.4-microsoft-standard-WSL2`) | Isolated native Cantera oracle environment |
| **Oracle Compilers** | GCC 13.3.0 / G++ 13.3.0 | Ubuntu 13.3.0-6ubuntu2~24.04.1 |

---

## 2. SHA-256 Hashes of Tested Mechanism Files

All mechanism files located under `outputs/cantera-original/data/`:

* `gri30.yaml`:
  `e7eb89d227c635943794fd96820a23c3b09da33b83f97e6aeed8293737c1a13f`
* `airNASA9.yaml`:
  `8ba7fafd17eeab2cc75f9b558a43b04ba0bb2e40a41e54975aa99bb555871724`
* `h2o2.yaml`:
  `341dd0db8537cd7860c3855d7cf600986387c08fe2057fd731ac320d1a08eec5`
* `example_data/ammonia-CO-H2-Alzueta-2023.yaml`:
  `5f8dced73f22e2202d0727cd9687aa99f76fefdbd374f01eda70dde7b9a0c9af`
* `ptcombust.yaml`:
  `4edd16561d233751deff2b0566dfbf4d8bdb9a046767c3589034668089690eb9`
* `diamond.yaml`:
  `b27040a660818f63a33405a41f2c2dfef674c4a38f0f7928ce8ae978f29f99b7`

---

## 3. Exact Commands Used During Audit

All commands executed from `outputs/cantera-python/` (or referencing scripts directly):

### A. Full Existing Pytest Suite Execution
```powershell
pytest -o pythonpath=. -q
```
*Result: 188 passed, 684 subtests passed in 70.86s.*

### B. Runtime Purity and Independence Audit
```powershell
python C:/Users/Dell/.gemini/antigravity/brain/993c9574-1ba3-41f4-9ef9-ecf417b147c2/scratch/verify_runtime_purity.py
```
*Generates: `audit/runtime_purity_report.txt`*

### C. Test Suite Quality & Weak Assertion Audit
```powershell
python C:/Users/Dell/.gemini/antigravity/brain/993c9574-1ba3-41f4-9ef9-ecf417b147c2/scratch/generate_test_report.py
```
*Generates: `audit/test_coverage_report.txt`*

### D. Feature Matrix Generation
```powershell
python C:/Users/Dell/.gemini/antigravity/brain/993c9574-1ba3-41f4-9ef9-ecf417b147c2/scratch/build_feature_matrix.py
```
*Generates: `audit/feature_matrix.csv`*

### E. WSL Native Oracle Benchmark Generation (10,050 States, Kinetics, Reactors)
```bash
wsl /home/nebula/cantera-oracle-env/bin/python /mnt/c/Users/Dell/.gemini/antigravity/brain/993c9574-1ba3-41f4-9ef9-ecf417b147c2/scratch/generate_native_oracle_differential.py
```
*Generates: `audit/wsl_oracle_data.json` (70.4 MB)*

### F. Differential Numerical Verification (Python vs Native Oracle)
```powershell
python -u C:/Users/Dell/.gemini/antigravity/brain/993c9574-1ba3-41f4-9ef9-ecf417b147c2/scratch/evaluate_differential_against_oracle.py
```
*Generates: `audit/oracle_results.json` and `audit/oracle_failures.json`*

### G. Mechanism Parser Corpus & Isolated Reaction Kinetics Audit
```powershell
python C:/Users/Dell/.gemini/antigravity/brain/993c9574-1ba3-41f4-9ef9-ecf417b147c2/scratch/verify_kinetics_and_mechanisms.py
```
*Generates: `audit/mechanism_parser_audit.json`*
