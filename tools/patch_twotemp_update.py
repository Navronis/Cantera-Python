from pathlib import Path

fpath = Path("cantera_python/rates.py")
text = fpath.read_text(encoding="utf-8")

old_block = """    def update(self, T_or_phase=None, kin_or_Te=None) -> bool:
        if hasattr(T_or_phase, "temperature"):
            phase = T_or_phase
            T = phase.temperature
            Te = getattr(phase, "electron_temperature", getattr(phase, "electronTemperature", T))
            changed = False
            if T != self.temperature:
                super().update(T=T)
                changed = True
            if Te != self.electronTemp:
                self.updateTe(Te)
                changed = True
            return changed
        elif kin_or_Te is not None:
            super().update(T=float(T_or_phase))
            self.updateTe(float(kin_or_Te))
            return True
        else:
            raise CanteraError("TwoTempPlasmaData requires electron temperature.")"""

new_block = """    def update(self, T_or_phase=None, kin_or_Te=None, T=None, Te=None) -> bool:
        if T is not None and T_or_phase is None:
            T_or_phase = T
        if Te is not None and kin_or_Te is None:
            kin_or_Te = Te
        if hasattr(T_or_phase, "temperature"):
            phase = T_or_phase
            T_val = phase.temperature
            Te_val = getattr(phase, "electron_temperature", getattr(phase, "electronTemperature", T_val))
            changed = False
            if T_val != self.temperature:
                super().update(T=T_val)
                changed = True
            if Te_val != self.electronTemp:
                self.updateTe(Te_val)
                changed = True
            return changed
        elif kin_or_Te is not None and T_or_phase is not None:
            super().update(T=float(T_or_phase))
            self.updateTe(float(kin_or_Te))
            return True
        else:
            raise CanteraError("TwoTempPlasmaData requires electron temperature.")"""

text_norm = text.replace("\r\n", "\n")
assert old_block in text_norm, "old_block not found in rates.py"
text_norm = text_norm.replace(old_block, new_block, 1)
fpath.write_text(text_norm, encoding="utf-8")
print("SUCCESS: updated TwoTempPlasmaData.update")
