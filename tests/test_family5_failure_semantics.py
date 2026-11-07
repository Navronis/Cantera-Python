"""Failure semantics tests for Kinetics Family 5 (Plasma, Vibrational, MultiRate).

Verifies source-exact exceptions and domain constraint enforcement for:
- Negative pre-exponential factor rejection unless explicitly enabled
- Irreversible requirement enforcement for TwoTempPlasmaRate
- MultiRate empty container error modes
- ElectronCollisionPlasmaRate grid dimension and monotonicity invariants
- CustomFunc1Rate unconfigured validation and invalid function type rejection
- Non-positive temperature rejection across specialized rate evaluators
"""

import math
import pytest
import cantera_python as ct
from cantera_python.constants import CanteraError
from cantera_python.kinetics import Reaction
from cantera_python.rates import (
    TwoTempPlasmaRate, TwoTempPlasmaData,
    ElectronCollisionPlasmaRate, ElectronCollisionPlasmaData,
    ConstantVibrationalRelaxationRate, StarikovskiyVibrationalRelaxationRate,
    CastelaVibrationalRelaxationRate, MultiStateResolvedVibrationalRelaxationRate,
    CustomFunc1Rate, MultiRate, newReactionRate
)


def test_negative_A_rejection_without_flag():
    with pytest.raises(ValueError, match="Negative A"):
        TwoTempPlasmaRate(A=-1.0e10, allow_negative_A=False)

    with pytest.raises(ValueError, match="Negative A"):
        ConstantVibrationalRelaxationRate(A=-100.0, allow_negative_A=False)

    with pytest.raises(ValueError, match="Negative A"):
        StarikovskiyVibrationalRelaxationRate(A=-50.0, allow_negative_A=False)


def test_negative_A_allowed_when_flagged():
    r2t = TwoTempPlasmaRate(A=-1.0e10, allow_negative_A=True)
    val = r2t(300.0, 1000.0)
    assert val < 0.0

    r_cv = ConstantVibrationalRelaxationRate(A=-100.0, allow_negative_A=True)
    assert r_cv(300.0) == -100.0


def test_two_temp_plasma_reversible_rejection():
    rxn = Reaction()
    rxn.reversible = True
    rate = TwoTempPlasmaRate()
    with pytest.raises(CanteraError, match="does not support reversible"):
        rate.setContext(rxn=rxn)


def test_multirate_empty_type_error():
    mr = MultiRate()
    with pytest.raises(CanteraError, match="empty rate handler"):
        mr.type()


def test_electron_collision_grid_validation():
    # Length mismatch
    with pytest.raises(ValueError, match="equal length"):
        ElectronCollisionPlasmaRate([0.0, 1.0, 2.0], [1.0e-20, 2.0e-20])

    # Non-monotonic energy grid
    with pytest.raises(ValueError, match="strictly increasing"):
        ElectronCollisionPlasmaRate([0.0, 2.0, 1.0], [1.0e-20, 2.0e-20, 3.0e-20])

    # Minimum length requirement
    with pytest.raises(ValueError, match="at least two"):
        ElectronCollisionPlasmaRate([1.0], [1.0e-20])

    # Eval grid vs distribution mismatch
    rate = ElectronCollisionPlasmaRate([0.0, 1.0, 2.0], [1.0e-20, 2.0e-20, 3.0e-20])
    with pytest.raises(ValueError, match="equal length"):
        rate([0.0, 1.0, 2.0], [0.5, 0.5])


def test_custom_func1_unconfigured_error():
    cfr = CustomFunc1Rate()
    with pytest.raises(ValueError, match="not configured"):
        cfr.validate("reaction_test")

    # Unconfigured evaluation returns NaN (matching native)
    assert math.isnan(cfr(500.0))

    # Invalid function type raises TypeError
    with pytest.raises(TypeError, match="must be callable"):
        cfr.set_rate_function(12345)


def test_non_positive_temperature_rejection():
    r2t = TwoTempPlasmaRate(A=1.0e10)
    with pytest.raises(ValueError, match="positive"):
        r2t(0.0, 1000.0)

    with pytest.raises(ValueError, match="positive"):
        r2t(1000.0, 0.0)

    with pytest.raises(ValueError, match="positive"):
        r2t(-500.0, 1000.0)

    r_st = StarikovskiyVibrationalRelaxationRate(A=100.0)
    with pytest.raises(ValueError, match="positive"):
        r_st(0.0)

    with pytest.raises(ValueError, match="positive"):
        r_st(-300.0)
