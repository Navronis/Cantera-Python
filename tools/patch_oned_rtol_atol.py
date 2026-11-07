from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
oned_file = repo_root / "cantera_python" / "oned.py"

with open(oned_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = 0
for i, l in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    if l.startswith("    def rtol("):
        new_lines.append("    def rtol(self, k: int = 0) -> float:\n")
        new_lines.append("        k = int(k)\n")
        new_lines.append("        tols = getattr(self, 'm_rtol_ts', []) if getattr(self, 'm_rdt', 0.0) != 0.0 else getattr(self, 'm_rtol_ss', [])\n")
        new_lines.append("        if k < len(tols):\n")
        new_lines.append("            return float(tols[k])\n")
        new_lines.append("        return 1e-4\n")
        skip = 1
    elif l.startswith("    def atol("):
        new_lines.append("    def atol(self, k: int = 0) -> float:\n")
        new_lines.append("        k = int(k)\n")
        new_lines.append("        tols = getattr(self, 'm_atol_ts', []) if getattr(self, 'm_rdt', 0.0) != 0.0 else getattr(self, 'm_atol_ss', [])\n")
        new_lines.append("        if k < len(tols):\n")
        new_lines.append("            return float(tols[k])\n")
        new_lines.append("        return 1e-9\n")
        skip = 4  # skip the old 4 lines
    else:
        new_lines.append(l)

with open(oned_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Successfully patched rtol and atol in Domain1D")
