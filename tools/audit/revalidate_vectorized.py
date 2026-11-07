"""Revalidate _VectorizedKinetics: scalar Python vs vectorized Python vs
exact pinned native oracle, over randomized valid states.

Covers elementary, reversible, irreversible, third-body, and falloff
reactions; zero and trace concentrations; temperature and pressure sweeps.

Appends results to provenance/pinned_native_differential_results.jsonl.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import cantera_python as ct  # noqa: E402
from cantera_python.kinetics import GasKinetics  # noqa: E402

PROV = ROOT / 'provenance'
RESULTS = PROV / 'pinned_native_differential_results.jsonl'
GRI30 = ROOT.parent / 'cantera-original' / 'data' / 'gri30.yaml'
ORACLE_BASELINE = PROV / 'oracle_baseline_pinned.json'

QUANTITIES = [
    ('forward_rate_constants', 'kf'),
    ('reverse_rate_constants', 'kr'),
    ('equilibrium_constants', 'Kc'),
    ('forward_rates_of_progress', 'qf'),
    ('reverse_rates_of_progress', 'qr'),
    ('creation_rates', 'creation'),
    ('destruction_rates', 'destruction'),
    ('net_production_rates', 'wdot'),
]


def rel_err(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if a == b else (abs(a - b) / scale if scale else abs(a - b))


def make_state(rng):
    """Random valid state with zero/trace species mixed in."""
    T = rng.uniform(300.0, 3000.0)
    P = 10.0 ** rng.uniform(3.7, 6.7)  # ~5e3 Pa .. 5e6 Pa
    n = 6
    species = rng.sample(['CH4', 'O2', 'N2', 'H2', 'CO', 'CO2', 'H2O', 'OH',
                          'HO2', 'H2O2', 'O', 'H'], n)
    fracs = [rng.random() ** 3 for _ in species]  # skewed: some near-trace
    if rng.random() < 0.25:
        fracs[0] = 0.0  # force a zero-concentration species
    total = sum(fracs)
    if total == 0.0:
        fracs[0] = 1.0
        total = 1.0
    X = ', '.join(f'{s}:{f / total:.8g}' for s, f in zip(species, fracs))
    return T, P, X


def main():
    rng = random.Random(72652284)
    base = json.loads(ORACLE_BASELINE.read_text())
    k = GasKinetics.from_mechanism(GRI30)
    gas = k._thermo
    rows = []

    # Part 1: scalar vs vectorized across randomized states (both must match
    # the pinned oracle at the two baseline states).
    worst_sv = {q: 0.0 for _, q in QUANTITIES}
    n_states = 40
    for i in range(n_states):
        T, P, X = make_state(rng)
        gas.TPX = T, P, X
        for attr, q in QUANTITIES:
            sv = list(getattr(k, attr)())
            fv = getattr(k, f'{attr}_fast')() if hasattr(k, f'{attr}_fast') else None
            if fv is None:
                # only kf/ROP/production have fast paths; compute equivalence
                # where a fast implementation exists
                continue
            fv = list(fv)
            for a, b in zip(sv, fv):
                e = rel_err(a, b)
                worst_sv[q] = max(worst_sv[q], e)
                e = float(e)
                if e > 1e-9:
                    rows.append({
                        "suite": "vectorized_equivalence", "case": f"state_{i}",
                        "quantity": q, "input": {"T": T, "P": P, "X": X},
                        "python_value": a, "vectorized_value": b,
                "rel_error": e, "oracle_role": "SCALAR_REFERENCE",
                        "pass": bool(e <= 1e-9),
                    })

    # Part 2: pinned oracle comparison at the two oracle baseline states,
    # using both scalar and vectorized paths.
    oracle_rows = []
    state_defs = {
        'state_1200_ch4': (1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'),
        'state_2000_dilute': (2000.0, 101325.0, 'O2:0.3, N2:0.7'),
    }
    worst_oracle = {}
    for label, (T, P, X) in state_defs.items():
        gas.TPX = T, P, X
        snap = base['states'][label]
        for attr, q in QUANTITIES:
            nat = snap[attr]
            sv = list(getattr(k, attr)())
            e_scalar = max(rel_err(a, b) for a, b in zip(sv, nat))
            fv = getattr(k, f'{attr}_fast')() if hasattr(k, f'{attr}_fast') else None
            e_vec = None
            if fv is not None:
                fv = list(fv)
                e_vec = max(rel_err(a, b) for a, b in zip(fv, nat))
            worst_oracle[(label, q)] = (e_scalar, e_vec)
            e_scalar = float(e_scalar)
            if e_vec is not None:
                e_vec = float(e_vec)
            oracle_rows.append({
                "suite": "vectorized_vs_pinned_oracle", "case": label,
                "quantity": q, "input": {"T": T, "P": P, "X": X},
                "oracle_value": "vector", "scalar_rel_error": e_scalar,
                "vectorized_rel_error": e_vec,
                "oracle_role": "AUTHORITATIVE_ORACLE",
                "oracle_commit": base['oracle_commit'],
                "pass": bool((e_scalar <= 1e-8) and (e_vec is None or e_vec <= 1e-8)),
            })

    with RESULTS.open('a', encoding='utf-8') as fh:
        for r in rows + oracle_rows:
            fh.write(json.dumps(r) + '\n')

    all_scalar = [e for e, _ in worst_oracle.values()]
    all_vec = [e for _, e in worst_oracle.values() if e is not None]
    summary = {
        'randomized_states': n_states,
        'scalar_vs_vectorized_worst': worst_sv,
        'oracle_scalar_worst': max(all_scalar),
        'oracle_vectorized_worst': max(all_vec) if all_vec else None,
        'oracle_failures': sum(1 for r in oracle_rows if not r['pass']),
        'equivalence_failures': sum(1 for r in rows if not r.get('pass', True)),
    }
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
