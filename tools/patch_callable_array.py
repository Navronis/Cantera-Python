from pathlib import Path

def main():
    path = Path("outputs/cantera-python/cantera_python/transport.py")
    content = path.read_text(encoding="utf-8")

    # Replace duplicate CallableFloat with CallableArray
    target1 = """class CallableFloat(float):
    \"\"\"A float subclass that is callable as a zero-argument function.\"\"\"
    def __call__(self):
        return float(self)


class CallableFloat(float):
    \"\"\"A float subclass that is callable as a zero-argument function.\"\"\"
    def __call__(self):
        return float(self)"""

    replacement1 = """class CallableFloat(float):
    \"\"\"A float subclass that is callable as a zero-argument function.\"\"\"
    def __call__(self):
        return float(self)


class CallableArray(np.ndarray):
    \"\"\"A numpy ndarray subclass that is callable as a zero-argument function.\"\"\"
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __call__(self):
        return self"""

    if target1 in content:
        content = content.replace(target1, replacement1, 1)
    else:
        target1_crlf = target1.replace("\n", "\r\n")
        replacement1_crlf = replacement1.replace("\n", "\r\n")
        assert target1_crlf in content, "target1 not found"
        content = content.replace(target1_crlf, replacement1_crlf, 1)

    target2 = """    def mix_diff_coeffs_mole(self) -> np.ndarray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return np.array([bdiff[0, 0]], dtype=float)
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum2 = sum(molefracs[j] / bdiff[j, i] for j in range(nsp) if j != i)
            if sum2 <= 0.0:
                d[i] = bdiff[i, i]
            else:
                d[i] = (1.0 - molefracs[i]) / sum2
        return d"""

    replacement2 = """    @property
    def mix_diff_coeffs_mole(self) -> CallableArray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return CallableArray(np.array([bdiff[0, 0]], dtype=float))
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum2 = sum(molefracs[j] / bdiff[j, i] for j in range(nsp) if j != i)
            if sum2 <= 0.0:
                d[i] = bdiff[i, i]
            else:
                d[i] = (1.0 - molefracs[i]) / sum2
        return CallableArray(d)"""

    if target2 in content:
        content = content.replace(target2, replacement2, 1)
    else:
        target2_crlf = target2.replace("\n", "\r\n")
        replacement2_crlf = replacement2.replace("\n", "\r\n")
        assert target2_crlf in content, "target2 not found"
        content = content.replace(target2_crlf, replacement2_crlf, 1)

    target3 = """    def mix_diff_coeffs_mass(self) -> np.ndarray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return np.array([bdiff[0, 0]], dtype=float)
        mmw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(nsp):
                if j == i:
                    continue
                sum1 += molefracs[j] / bdiff[i, j]
                sum2 += molefracs[j] * mw[j] / bdiff[i, j]
            denom = mmw - mw[i] * molefracs[i]
            sum2 *= molefracs[i] / denom if abs(denom) > Tiny else 0.0
            d[i] = 1.0 / (sum1 + sum2)
        return d"""

    replacement3 = """    @property
    def mix_diff_coeffs_mass(self) -> CallableArray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return CallableArray(np.array([bdiff[0, 0]], dtype=float))
        mmw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(nsp):
                if j == i:
                    continue
                sum1 += molefracs[j] / bdiff[i, j]
                sum2 += molefracs[j] * mw[j] / bdiff[i, j]
            denom = mmw - mw[i] * molefracs[i]
            sum2 *= molefracs[i] / denom if abs(denom) > Tiny else 0.0
            d[i] = 1.0 / (sum1 + sum2)
        return CallableArray(d)"""

    if target3 in content:
        content = content.replace(target3, replacement3, 1)
    else:
        target3_crlf = target3.replace("\n", "\r\n")
        replacement3_crlf = replacement3.replace("\n", "\r\n")
        assert target3_crlf in content, "target3 not found"
        content = content.replace(target3_crlf, replacement3_crlf, 1)

    path.write_text(content, encoding="utf-8")
    print("Patched CallableArray and properties in transport.py")

if __name__ == "__main__":
    main()
