from pathlib import Path

def main():
    path = Path(__file__).resolve().parent.parent / "cantera_python" / "transport.py"
    content = path.read_text(encoding="utf-8")

    target1 = (
        "        self.initializeCriticalProperties()\n"
        "        self._P_corr_ij = np.ones((nsp, nsp), dtype=float)"
    )
    replacement1 = (
        "        self._mw = np.asarray(self._thermo.molecular_weights, dtype=float)\n"
        "        self._dipole = np.zeros(nsp, dtype=float)\n"
        "        if hasattr(self, '_transport_data') and self._transport_data:\n"
        "            for k in range(nsp):\n"
        "                name = self._thermo.species_name(k)\n"
        "                td = self._transport_data.get(name)\n"
        "                if td and hasattr(td, 'dipole'):\n"
        "                    self._dipole[k] = float(td.dipole)\n"
        "        self.initializeCriticalProperties()\n"
        "        self._P_corr_ij = np.ones((nsp, nsp), dtype=float)"
    )

    if target1 in content:
        content = content.replace(target1, replacement1, 1)
    else:
        # Check CRLF
        target1_crlf = target1.replace("\n", "\r\n")
        replacement1_crlf = replacement1.replace("\n", "\r\n")
        assert target1_crlf in content, "target1 not found in transport.py"
        content = content.replace(target1_crlf, replacement1_crlf, 1)

    target2 = (
        "class ChungHighPressureGasTransport(HighPressureGasTransportBase):\n"
        '    """High-pressure gas transport using the Chung et al. dense gas correlation."""\n\n'
        "    def transportModel(self) -> str:"
    )
    replacement2 = (
        "class ChungHighPressureGasTransport(HighPressureGasTransportBase):\n"
        '    """High-pressure gas transport using the Chung et al. dense gas correlation."""\n\n'
        "    def __init__(self, thermo, transport_data: Mapping[str, GasTransportData] | None = None):\n"
        "        super().__init__(thermo, transport_data)\n"
        "        self.initializePureFluidProperties()\n\n"
        "    def transportModel(self) -> str:"
    )

    if target2 in content:
        content = content.replace(target2, replacement2, 1)
    else:
        target2_crlf = target2.replace("\n", "\r\n")
        replacement2_crlf = replacement2.replace("\n", "\r\n")
        assert target2_crlf in content, "target2 not found in transport.py"
        content = content.replace(target2_crlf, replacement2_crlf, 1)

    path.write_text(content, encoding="utf-8")
    print("Successfully patched transport.py")

if __name__ == "__main__":
    main()
