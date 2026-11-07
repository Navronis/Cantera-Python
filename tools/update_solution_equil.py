"""Update Solution.equilibrate in cantera_python/solution.py to support ChemEquil solvers."""

from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    sol_path = repo_root / "cantera_python" / "solution.py"
    lines = sol_path.read_text(encoding="utf-8").splitlines(keepends=True)

    found = False
    for i, line in enumerate(lines):
        if 'if solver == "vcs":' in line:
            insert_idx = i + 6
            extra = [
                '        if solver in ("element_potential", "element_potentials", "chem_equil", "ChemEquil"):\n',
                '            from .equilibrium import ChemEquil\n',
                '            eq = ChemEquil(self)\n',
                '            eq.equilibrate(self, XY=mode, max_steps=max_steps, options={"relTolerance": rtol})\n',
                '            return None\n',
            ]
            lines = lines[:insert_idx] + extra + lines[insert_idx:]
            found = True
            break

    if not found:
        raise ValueError('Could not find if solver == "vcs": in solution.py')

    sol_path.write_text("".join(lines), encoding="utf-8")
    print("Updated solution.py successfully")

if __name__ == "__main__":
    main()
