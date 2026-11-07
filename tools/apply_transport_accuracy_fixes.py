"""Apply accuracy fixes to WaterTransport and IonGasTransport."""
import re

with open("cantera_python/transport.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update WaterTransport kappa and alpha extraction
target_wt = """        if 0.9970 <= tbar <= 1.0082 and 0.755 <= rhobar <= 1.290:
            kappa_fn = getattr(self._thermo, "isothermalCompressibility", None)
            kappa = float(kappa_fn()) if callable(kappa_fn) else getattr(self._thermo, "isothermal_compressibility", 0.0)
            drhodp = kappa * dens * (_PRESSTAR / _RHOSTAR)
            xsi = rhobar * drhodp
            if xsi >= 21.93:
                mu2bar = 0.922 * (xsi ** 0.0263)"""

replace_wt = """        if 0.9970 <= tbar <= 1.0082 and 0.755 <= rhobar <= 1.290:
            kappa = 0.0
            for attr in ("isothermalCompressibility", "isothermal_compressibility", "_isothermal_compressibility"):
                val = getattr(self._thermo, attr, None)
                if val is not None:
                    kappa = float(val() if callable(val) else val)
                    break
            drhodp = kappa * dens * (_PRESSTAR / _RHOSTAR)
            xsi = rhobar * drhodp
            if xsi >= 21.93:
                mu2bar = 0.922 * (xsi ** 0.0263)"""

if "\r\n" in text:
    target_wt = target_wt.replace("\n", "\r\n")
    replace_wt = replace_wt.replace("\n", "\r\n")

if target_wt in text:
    text = text.replace(target_wt, replace_wt, 1)
    print("Updated WaterTransport viscosity kappa handling")
else:
    print("Warning: target_wt not found")

target_lam = """        t2r2 = tbar2 / (rhobar * rhobar)
        kappa_fn = getattr(self._thermo, "isothermalCompressibility", None)
        kappa = float(kappa_fn()) if callable(kappa_fn) else getattr(self._thermo, "isothermal_compressibility", 0.0)
        xsi = rhobar * rhobar * kappa * _PRESSTAR
        xsipow = (xsi ** 0.4678) if xsi > 0.0 else 0.0
        temp2 = (tbar - 1.0) * (tbar - 1.0)
        alpha_fn = getattr(self._thermo, "thermalExpansionCoeff", None)
        alpha = float(alpha_fn()) if callable(alpha_fn) else getattr(self._thermo, "thermal_expansion_coeff", 0.0)"""

replace_lam = """        t2r2 = tbar2 / (rhobar * rhobar)
        kappa = 0.0
        for attr in ("isothermalCompressibility", "isothermal_compressibility", "_isothermal_compressibility"):
            val = getattr(self._thermo, attr, None)
            if val is not None:
                kappa = float(val() if callable(val) else val)
                break
        xsi = rhobar * rhobar * kappa * _PRESSTAR
        xsipow = (xsi ** 0.4678) if xsi > 0.0 else 0.0
        temp2 = (tbar - 1.0) * (tbar - 1.0)
        alpha = 0.0
        for attr in ("thermalExpansionCoeff", "thermal_expansion_coeff", "_thermal_expansion_coeff"):
            val = getattr(self._thermo, attr, None)
            if val is not None:
                alpha = float(val() if callable(val) else val)
                break"""

if "\r\n" in text:
    target_lam = target_lam.replace("\n", "\r\n")
    replace_lam = replace_lam.replace("\n", "\r\n")

if target_lam in text:
    text = text.replace(target_lam, replace_lam, 1)
    print("Updated WaterTransport thermalConductivity kappa/alpha handling")
else:
    print("Warning: target_lam not found")

# 2. Update IonGasTransport O2 / O2- Prager fit
target_prager1 = """        temp_prager = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0]
        om11_O2_vals = [120.0, 107.0, 98.1, 92.1, 83.0, 77.0, 72.6, 67.9, 62.7, 59.3, 56.7, 53.8]
        self._om11_O2 = np.polyfit(temp_prager, om11_O2_vals, 5)[::-1]"""

replace_prager1 = """        # Pinned native Cantera polynomial coefficients for O2/O2- from Prager (2005)
        self._om11_O2 = np.array([0.0, 0.50570048102629783, -0.00080573149983813373,
                                  5.0274392926237489e-07, -1.3558670838184069e-10,
                                  1.3142329548561772e-14], dtype=float)"""

if "\r\n" in text:
    target_prager1 = target_prager1.replace("\n", "\r\n")
    replace_prager1 = replace_prager1.replace("\n", "\r\n")

while target_prager1 in text:
    text = text.replace(target_prager1, replace_prager1, 1)
    print("Updated IonGasTransport Prager polynomial")

with open("cantera_python/transport.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Finished writing cantera_python/transport.py")
