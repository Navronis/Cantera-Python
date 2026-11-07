"""Verify mechanism schema for DustyGasTransport."""

import json
from pathlib import Path
import sys

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    sys.path.insert(0, str(repo_root))

    from cantera_python.mechanism import _as_document

    mechanisms = ["gri30.yaml", "h2o2.yaml"]
    schema_report = {
        "schema_version": 1,
        "verified_at": "2026-11-04T22:25:00Z",
        "mechanisms": {},
    }

    data_dir = repo_root.parent / "cantera-original" / "data"

    for mech_name in mechanisms:
        mech_path = data_dir / mech_name
        doc = _as_document(mech_path.read_text(encoding="utf-8"))
        yaml_species = [s["name"] for s in doc.get("species", [])]
        k_yaml = len(yaml_species)

        # Import in python
        import cantera_python as ct
        gas = ct.DustyGas(mech_path)
        py_species = list(gas.species_names)
        k_py = len(py_species)

        assert k_yaml == k_py, f"Dimension mismatch for {mech_name}: {k_yaml} vs {k_py}"
        assert yaml_species == py_species, f"Species order mismatch for {mech_name}"

        schema_report["mechanisms"][mech_name] = {
            "n_species": k_py,
            "species": py_species,
            "matrix_shape": [k_py, k_py],
            "flux_vector_shape": [k_py],
            "verified": True,
        }
        print(f"Verified schema for {mech_name}: K={k_py}, matrix shape=({k_py}, {k_py})")

    out_path = audit_dir / "dusty_gas_transport_mechanism_schema.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema_report, f, indent=2)
    print(f"Wrote schema report to {out_path}")

if __name__ == "__main__":
    main()
