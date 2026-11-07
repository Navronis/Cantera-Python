from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
oned_file = repo_root / "cantera_python" / "oned.py"

with open(oned_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, l in enumerate(lines):
    if l.startswith("    def setupGrid(self, z: Sequence[float] | np.ndarray):"):
        new_lines.append(l)
        new_lines.append("        arr = np.asarray(z, dtype=float)\n")
        new_lines.append("        self._grid = arr.copy()\n")
        new_lines.append("        self.m_points = len(arr)\n")
        # skip next line 'self.grid = z'
        lines[i + 1] = ""
    elif "return added if added > 0 else (-discarded if discarded > 0 else 0)" in l:
        new_lines.append("        return 1 if (added != 0 or discarded != 0) else 0\n")
    elif "self.resize()" in l and i > 2700 and "restoreSteadySolution" in lines[i - 5]:
        # skip self.resize() in restoreSteadySolution
        continue
    else:
        if l:
            new_lines.append(l)

with open(oned_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Successfully updated setupGrid and refine return")
