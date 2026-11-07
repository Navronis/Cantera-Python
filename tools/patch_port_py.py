from pathlib import Path

fpath = Path("outputs/cantera-python/tools/run_python_kinetics_family5_port.py")
text = fpath.read_text(encoding="utf-8")

text = text.replace('ct.Solution(mech, transport_model="none")', "ct.Solution(mech)")
text = text.replace('cfr.type()', 'cfr.type() if callable(getattr(cfr, "type", None)) else cfr.type')

fpath.write_text(text, encoding="utf-8")
print("SUCCESS: updated run_python_kinetics_family5_port.py")
