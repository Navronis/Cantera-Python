"""IEEE edge behavior used by the pinned TPX set_xy Newton update."""

import math
import warnings

import numpy as np
import pytest

from cantera_python.tpx import TPXError, Water, _ieee_divide


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (2.0, 0.0, math.inf),
        (-2.0, 0.0, -math.inf),
        (2.0, -0.0, -math.inf),
        (-2.0, -0.0, math.inf),
        (0.0, 0.0, math.nan),
        (math.nan, 0.0, math.nan),
        (6.0, 3.0, 2.0),
    ],
)
def test_ieee_division_matches_native_floating_point(numerator, denominator, expected):
    # Pinned src/tpx/Sub.cpp:701-704 divides the Newton numerators directly
    # by the determinant. Python's float division needs this adapter for ±0.
    result = _ieee_divide(numerator, denominator)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


def test_ieee_division_numpy_scalars_do_not_emit_numpy_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        positive = _ieee_divide(np.float64(2.0), np.float64(0.0))
        indeterminate = _ieee_divide(np.float64(0.0), np.float64(0.0))

    assert positive == math.inf
    assert math.isnan(indeterminate)
    assert caught == []


@pytest.mark.parametrize("value", [math.nan, math.copysign(math.nan, -1.0)])
def test_set_v_rejects_nan_like_native_positive_predicate(value):
    # Native Sub.cpp::set_v commits only when v0 > 0; NaN fails that test.
    with pytest.raises(TPXError, match="Negative specific volume"):
        Water().set_v(value)
