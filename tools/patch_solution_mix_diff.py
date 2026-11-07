from pathlib import Path

def main():
    path = Path("outputs/cantera-python/cantera_python/solution.py")
    content = path.read_text(encoding="utf-8")

    old = """    @property
    def mix_diff_coeffs_mole(self):
        return self._transport.mix_diff_coeffs_mole if self._transport else ()

    @property
    def mix_diff_coeffs_mass(self):
        return self._transport.mix_diff_coeffs_mass if self._transport else ()"""

    new = """    @property
    def mix_diff_coeffs_mole(self):
        if not self._transport:
            return ()
        v = self._transport.mix_diff_coeffs_mole
        return v() if callable(v) else v

    @property
    def mix_diff_coeffs_mass(self):
        if not self._transport:
            return ()
        v = self._transport.mix_diff_coeffs_mass
        return v() if callable(v) else v"""

    if old in content:
        content = content.replace(old, new, 1)
    else:
        old_crlf = old.replace("\n", "\r\n")
        new_crlf = new.replace("\n", "\r\n")
        assert old_crlf in content, "old not found"
        content = content.replace(old_crlf, new_crlf, 1)

    path.write_text(content, encoding="utf-8")
    print("Patched solution.py mix_diff_coeffs_mole and mass")

if __name__ == "__main__":
    main()
