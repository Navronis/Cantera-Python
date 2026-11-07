# Reproduction commands

Run from `outputs/cantera-python` with the repository validation environment:

```powershell
& '..\..\work\reference-env\Scripts\python.exe' -m unittest discover -s tests -v
& '..\..\work\reference-env\Scripts\python.exe' tools\oracle\compare_pinned.py
& '..\..\work\reference-env\Scripts\python.exe' tools\reconcile_oracle_fixture.py
& '..\..\work\reference-env\Scripts\python.exe' tools\build_mechanism_compatibility_csv.py
& '..\..\work\reference-env\Scripts\python.exe' tools\validate_equilibrium_released.py
& '..\..\work\reference-env\Scripts\python.exe' tools\normalize_feature_matrix.py
& '..\..\work\reference-env\Scripts\python.exe' tools\build_symbol_closure_dashboard.py
& '..\..\work\reference-env\Scripts\python.exe' tools\build_symbol_work_maps.py
& '..\..\work\reference-env\Scripts\python.exe' tools\build_canonical_symbol_inventory.py
& '..\..\work\reference-env\Scripts\python.exe' tools\revalidate_equivalence_claims.py
& '..\..\work\reference-env\Scripts\python.exe' -m coverage run --branch -m pytest -o pythonpath=. -q -k 'not reactor'
& '..\..\work\reference-env\Scripts\python.exe' -m coverage json -o provenance\coverage_current.json
& '..\..\work\reference-env\Scripts\python.exe' -m coverage report -m
```

`audit/wsl_oracle_data.json` is a serialized output of the native pinned build. This run verified its SHA-256 and replayed it. Regeneration was attempted but the host denied WSL launch with `E_ACCESSDENIED`.

The coverage figures currently recorded in `PORT_STATUS.json` are an earlier 180-test snapshot and are not the current full-suite coverage. The authoritative uninstrumented regression run recorded there exercised all 235 tests.

`tools/generate_port_status.py` is a legacy generator tied to the earlier 192-test manifest and is intentionally excluded until its evidence inputs are migrated. Running it would overwrite newer verified status fields with stale data.
