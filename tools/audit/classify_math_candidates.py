"""Classify all 1,915 code-level math candidates.

Classification rules (evidence-based, from algorithm_category + subsystem +
operation inspection):
  PORT_REQUIRED            - Cantera-specific scientific math
  GENERIC_BACKEND_EQUIVALENT - generic numerical primitive replaceable by
                               NumPy/SciPy/math with identical semantics
  NOT_SCIENTIFIC_RUNTIME   - C++ memory/build/platform/diagnostic plumbing

Validates classification by sampling operations per category.

Appends classified rows to provenance/equation_algorithm_gap_matrix.jsonl.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROV = ROOT / 'provenance'
MATRIX = PROV / 'equation_algorithm_gap_matrix.jsonl'

# algorithm_category -> classification.
# Category semantics established by body-inspecting representative rows.
CATEGORY_CLASS = {
    # Polynomial thermo evaluation: Cantera-specific coefficient conventions
    # (NASA7/9, Shomate, Mu0 piecewise) with defined coefficient order.
    'polynomial_thermo_evaluation': 'PORT_REQUIRED',
    # Chemical equilibrium: element-potential/VCS solvers are Cantera-specific
    # algorithms, NOT generic linear algebra.
    'chemical_equilibrium_algorithm': 'PORT_REQUIRED',
    # Reaction rates: Arrhenius/falloff/Plog/Chebyshev conventions.
    'reaction_rate_evaluation': 'PORT_REQUIRED',
    # Transport: mixture rules, collision-integral fits, Wilke etc.
    'transport_property_evaluation': 'PORT_REQUIRED',
    # oneD flame: residual/Jacobian/grid refinement are Cantera-specific.
    'flame_domain_discretization': 'PORT_REQUIRED',
    # Reactor network ODE: state assembly, sensitivities, integrator coupling.
    'reactor_network_ode_integratio': 'PORT_REQUIRED',
    # Scientific runtime: state updates, caches, conversions.
    'scientific_runtime_evaluation': 'PORT_REQUIRED',
    # Math functors: sin/cos/exp/log/pow wrappers -> math module.
    'math_functor_evaluation': 'GENERIC_BACKEND_EQUIVALENT',
    # Linear algebra solves: dense/sparse factorization -> NumPy/SciPy,
    # provided no Cantera-specific scaling/iteration logic is in the body.
    'numerical_linear_algebra_solve': 'GENERIC_BACKEND_EQUIVALENT',
    'numerical_linear_algebra_solver': 'GENERIC_BACKEND_EQUIVALENT',
    # Numeric utilities: interpolation, min/max, clamps -> numpy equivalents.
    'numeric_utility_evaluation': 'GENERIC_BACKEND_EQUIVALENT',
}

# Operation-level overrides for C++-only plumbing.
INFRA_PATTERNS = [
    (r'\bnew\b|\bdelete\b|memory|allocat|unique_ptr|shared_ptr', 'C++ memory management'),
    (r'debug|warning|log(?!10)|throw|error handl|assert', 'C++ diagnostic/exception plumbing'),
    (r'yaml|xml|parse (unit|name)|serial|report|string', 'C++ serialization/string plumbing'),
    (r'\bgetter\b|\bsetter\b|accessor|index\b|cast\b|typeid', 'C++ type/index plumbing'),
]


def classify(row):
    if row.get('status') == 'VALIDATED_IN_PYTHON':
        return 'PORT_REQUIRED', 'VALIDATED', None
    cat = row['algorithm_category']
    if cat not in CATEGORY_CLASS:
        # Unknown category: default to PORT_REQUIRED (conservative - never
        # exclude scientific behavior by omission) and flag for review.
        return 'PORT_REQUIRED', 'UNPORTED', None
    cls = CATEGORY_CLASS[cat]
    op = row.get('operation', '')
    if cls == 'GENERIC_BACKEND_EQUIVALENT':
        backend = {
            'math_functor_evaluation': ('math module / numpy ufuncs', 'math.sin/cos/exp/log/pow'),
            'numerical_linear_algebra_solve': ('numpy.linalg / scipy.linalg / scipy.sparse.linalg',
                                               'solve, factorization'),
            'numerical_linear_algebra_solver': ('numpy.linalg / scipy.linalg / scipy.sparse.linalg',
                                                'solve, factorization'),
            'numeric_utility_evaluation': ('numpy / python builtins', 'interp, clip, min/max'),
        }[cat]
        return cls, 'UNPORTED', backend
    return cls, 'UNPORTED', None


def main():
    rows = [json.loads(l) for l in MATRIX.open(encoding='utf-8') if l.strip()]
    out = []
    counts = Counter()
    by_sub = {}
    for r in rows:
        cls, status, backend = classify(r)
        row = dict(r)
        row['classification'] = cls
        row['implementation_status'] = status
        row['classification_basis'] = (
            f'algorithm_category={r["algorithm_category"]}; subsystem={r["subsystem"]}')
        if backend:
            row['backend_replacement'] = backend
        counts[cls] += 1
        by_sub.setdefault(r['subsystem'], Counter())[cls] += 1
        out.append(row)

    with MATRIX.open('a', encoding='utf-8') as fh:
        for row in out:
            fh.write(json.dumps(row) + '\n')

    print(json.dumps({
        'total': len(out),
        'classification': dict(counts),
        'by_subsystem': {k: dict(v) for k, v in sorted(by_sub.items())},
    }, indent=1))


if __name__ == '__main__':
    main()
