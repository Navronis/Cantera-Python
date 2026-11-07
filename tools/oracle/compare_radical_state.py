"""Compare Python kinetics against the oracle at the radical-seeded state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import cantera_python as ct  # noqa: E402

GRI30 = ROOT.parent / 'cantera-original' / 'data' / 'gri30.yaml'
ORACLE = Path(r'C:\tmp\oracle_radical.json') if False else None


def rel_err(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if a == b else abs(a - b) / scale


def main(oracle_path):
    base = json.loads(Path(oracle_path).read_text())
    gas = ct.Solution(GRI30)
    gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4, OH:1e-6, H:1e-8'
    worst = {}
    for qty, py in [
        ('forward_rate_constants', gas.forward_rate_constants),
        ('reverse_rate_constants', gas.reverse_rate_constants),
        ('net_production_rates', gas.net_production_rates),
        ('creation_rates', gas.creation_rates),
        ('destruction_rates', gas.destruction_rates),
        ('partial_molar_enthalpies', gas.partial_molar_enthalpies),
    ]:
        nat = base[qty]
        e = max(rel_err(a, b) for a, b in zip(py, nat))
        worst[qty] = e
    # standard gibbs nondimensional
    mu0 = [g / (8314.46261815324 * gas.temperature) for g in gas.standard_chemical_potentials]
    worst['standard_gibbs_RT'] = max(rel_err(a, b) for a, b in zip(mu0, base['standard_gibbs_RT']))
    print(json.dumps(worst, indent=1))


if __name__ == '__main__':
    import sys as s
    main(s.argv[1])
