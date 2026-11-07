from pathlib import Path

fpath = Path("outputs/cantera-python/tools/run_native_kinetics_family5_oracle.cpp")
text = fpath.read_text(encoding="utf-8")
text = text.replace("newSolution(mech)", 'newSolution(mech, "", "none")')
fpath.write_text(text, encoding="utf-8")
print("Updated newSolution calls in cpp file.")
