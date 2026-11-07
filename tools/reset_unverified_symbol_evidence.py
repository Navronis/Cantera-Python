"""Remove heuristic symbol promotions from the strict closure ledger.

The previous evidence generator accepted name matches and source-file presence
as semantic validation. Those records are not scientific proof and can map a
native symbol to an unrelated Python class. This reset retains only the
symbol-linked DustyGas equation evidence produced by its dedicated tests;
future evidence must be added by similarly scoped, reviewable validators.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
SOURCE = PROVENANCE / "original_cantera_symbols.jsonl"
DESTINATION = PROVENANCE / "symbol_implementation_evidence.jsonl"
COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"

MAPPING = {
    "transportModel": "cantera_python.transport.DustyGasTransport.transport_model",
    "getMultiDiffCoeffs": "cantera_python.transport.DustyGasTransport.getMultiDiffCoeffs",
    "setPorosity": "cantera_python.transport.DustyGasTransport.setPorosity",
    "setTortuosity": "cantera_python.transport.DustyGasTransport.setTortuosity",
    "setMeanPoreRadius": "cantera_python.transport.DustyGasTransport.setMeanPoreRadius",
    "setMeanParticleDiameter": "cantera_python.transport.DustyGasTransport.setMeanParticleDiameter",
    "setPermeability": "cantera_python.transport.DustyGasTransport.setPermeability",
    "gasTransport": "cantera_python.transport.DustyGasTransport.gasTransport",
    "initialize": "cantera_python.transport.DustyGasTransport.__init__",
    "updateTransport_T": "cantera_python.transport.DustyGasTransport.multi_diff_coeffs",
    "updateTransport_C": "cantera_python.transport.DustyGasTransport._resistance_matrix",
    "updateBinaryDiffCoeffs": "cantera_python.transport.DustyGasTransport._effective_binary_diff_coeffs",
    "updateMultiDiffCoeffs": "cantera_python.transport.DustyGasTransport.multi_diff_coeffs",
    "updateKnudsenDiffCoeffs": "cantera_python.transport.DustyGasTransport.knudsen_diff_coeffs",
    "eval_H_matrix": "cantera_python.transport.DustyGasTransport._resistance_matrix",
    "cbar": "cantera_python.transport.DustyGasTransport.getMolarFluxes:cbar",
    "gradc": "cantera_python.transport.DustyGasTransport.getMolarFluxes:gradc",
}


def main():
    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    rows = []
    for item in source:
        if "DustyGasTransport" not in item["source_file"]:
            continue
        symbol = item["symbol"]
        if symbol not in MAPPING:
            continue
        rows.append({
            "current_status": "PARTIAL",
            "python_mapping": MAPPING[symbol],
            "python_mapping_confidence": "PINNED_SOURCE_BODY_AND_EQUATION_TESTS_REVIEWED",
            "remaining_evidence": (
                "Exact pinned-native differential validation remains pending; "
                "the subordinate multicomponent gas transport layer requires "
                "additional pinned-source comparison."),
            "source_commit": COMMIT,
            "source_hash": item["source_sha256"],
            "stable_symbol_id": item["id"],
            "validation_evidence": [
                "cantera_python/transport.py",
                "cantera_python/solution.py",
                "tests/test_dusty_gas_transport.py",
            ],
        })
    rows.sort(key=lambda item: item["stable_symbol_id"])
    DESTINATION.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )
    print(json.dumps({"retained_symbol_evidence": len(rows), "output": str(DESTINATION)}))


if __name__ == "__main__":
    main()
