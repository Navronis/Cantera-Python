"""Patch Domain1D.get_refine_criteria to return dict matching Python Cantera."""
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
oned_path = repo_root / "cantera_python" / "oned.py"

content = oned_path.read_text(encoding="utf-8")
old_block = (
    "    def get_refine_criteria(self) -> dict[str, float]:\n"
    "        return self.m_refiner.get_criteria()"
)
new_block = (
    "    def get_refine_criteria(self) -> dict[str, float]:\n"
    "        crit = self.m_refiner.get_criteria()\n"
    "        return {\n"
    '            "ratio": crit[0],\n'
    '            "slope": crit[1],\n'
    '            "curve": crit[2],\n'
    '            "prune": crit[3],\n'
    "        }"
)

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    oned_path.write_text(content, encoding="utf-8")
    print("Successfully patched Domain1D.get_refine_criteria")
else:
    print("old_block not found")
