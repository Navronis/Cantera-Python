"""Apply targeted fixes to transport and fixed_stoichiometry."""
import re

# 1. Update Transport class in cantera_python/transport.py
with open("cantera_python/transport.py", "r", encoding="utf-8") as f:
    tr_text = f.read()

target1 = """    def transportModel(self) -> str:
        return "None"
"""
replace1 = """    def transportModel(self) -> str:
        return "None"

    @property
    def transport_model(self) -> str:
        return self.transportModel()
"""

# Normalize CRLF / LF for robust matching
if "\r\n" in tr_text:
    target1 = target1.replace("\n", "\r\n")
    replace1 = replace1.replace("\n", "\r\n")

if target1 in tr_text:
    tr_text = tr_text.replace(target1, replace1, 1)
    with open("cantera_python/transport.py", "w", encoding="utf-8") as f:
        f.write(tr_text)
    print("Updated transport.py successfully")
else:
    print("Warning: target1 not found in transport.py")

# 2. Update fixed_stoichiometry.py
with open("cantera_python/fixed_stoichiometry.py", "r", encoding="utf-8") as f:
    fs_text = f.read()

target2 = """        state = phase.state or {}
        if 'T' in state:
            obj.T = float(state['T'])
        if 'P' in state:
            obj.P = float(state['P'])"""

replace2 = """        state = phase.state or {}
        if 'T' in state:
            raw_T = state['T']
            if isinstance(raw_T, (int, float)):
                obj.T = float(raw_T)
            elif isinstance(raw_T, str):
                parts = raw_T.split()
                obj.T = convert(float(parts[0]), parts[1], 'K') if len(parts) == 2 else float(parts[0])
        if 'P' in state:
            raw_P = state['P']
            if isinstance(raw_P, (int, float)):
                obj.P = float(raw_P)
            elif isinstance(raw_P, str):
                parts = raw_P.split()
                obj.P = convert(float(parts[0]), parts[1], 'Pa') if len(parts) == 2 else float(parts[0])"""

if "\r\n" in fs_text:
    target2 = target2.replace("\n", "\r\n")
    replace2 = replace2.replace("\n", "\r\n")

if target2 in fs_text:
    fs_text = fs_text.replace(target2, replace2, 1)
    with open("cantera_python/fixed_stoichiometry.py", "w", encoding="utf-8") as f:
        f.write(fs_text)
    print("Updated fixed_stoichiometry.py successfully")
else:
    print("Warning: target2 not found in fixed_stoichiometry.py")
