"""Pinned TPX TX/PX bounds and failure-state semantics."""

import pytest

from cantera_python.tpx import PropertyPair, TPXError, Water


def _water_at_reference_state():
    water = Water()
    water.Set(PropertyPair.TP, 400.0, 2.0e5)
    return water


def test_tx_rejects_quality_before_mutating_phase_state():
    water = _water_at_reference_state()
    before = (water.T, water.Rho)

    with pytest.raises(TPXError, match="Invalid vapor fraction"):
        water.Set(PropertyPair.TX, 350.0, -0.01)

    assert (water.T, water.Rho) == before


@pytest.mark.parametrize("temperature", [272.0, 648.2])
def test_tx_rejects_out_of_saturation_temperature_range(temperature):
    water = _water_at_reference_state()
    before = (water.T, water.Rho)

    with pytest.raises(TPXError, match="Illegal temperature value"):
        water.Set(PropertyPair.TX, temperature, 0.5)

    assert (water.T, water.Rho) == before


def test_px_invalid_quality_leaves_phase_at_tmin_like_pinned_source():
    water = _water_at_reference_state()

    with pytest.raises(TPXError, match="Invalid vapor fraction"):
        water.Set(PropertyPair.PX, 1.0e5, 1.01)

    assert water.T == water.Tmin()


@pytest.mark.parametrize("pressure", [0.0, 25.0e6])
def test_px_rejects_pressure_outside_saturation_domain(pressure):
    water = _water_at_reference_state()

    with pytest.raises(TPXError, match="Illegal pressure value"):
        water.Set(PropertyPair.PX, pressure, 0.5)

    # Pinned Substance::Set(PX) sets T=Tmin before validating pressure.
    assert water.T == water.Tmin()


_PAIR_VALUES = {
    "TV": lambda phase: (phase.T, phase.v()),
    "HP": lambda phase: (phase.h(), phase.P()),
    "SP": lambda phase: (phase.s(), phase.P()),
    "PV": lambda phase: (phase.P(), phase.v()),
    "TP": lambda phase: (phase.T, phase.P()),
    "UV": lambda phase: (phase.u(), phase.v()),
    "ST": lambda phase: (phase.s(), phase.T),
    "SV": lambda phase: (phase.s(), phase.v()),
    "UP": lambda phase: (phase.u(), phase.P()),
    "VH": lambda phase: (phase.v(), phase.h()),
    "TH": lambda phase: (phase.T, phase.h()),
    "SH": lambda phase: (phase.s(), phase.h()),
    "PX": lambda phase: (phase.Ps(), 0.4),
    "TX": lambda phase: (373.15, 0.4),
}

_REVERSE_CODES = {
    "TV": PropertyPair.VT, "HP": PropertyPair.PH, "SP": PropertyPair.PS,
    "PV": PropertyPair.VP, "TP": PropertyPair.PT, "UV": PropertyPair.VU,
    "ST": PropertyPair.TS, "SV": PropertyPair.VS, "UP": PropertyPair.PU,
    "VH": PropertyPair.HV, "TH": PropertyPair.HT, "SH": PropertyPair.HS,
    "PX": PropertyPair.XP, "TX": PropertyPair.XT,
}


@pytest.mark.parametrize("pair_name", tuple(_PAIR_VALUES))
@pytest.mark.parametrize("use_reverse_code", [False, True])
def test_all_tpx_property_pairs_dispatch_in_both_orders(pair_name, use_reverse_code):
    target = _water_at_reference_state()
    if pair_name in ("PX", "TX"):
        target.Set(PropertyPair.TX, 373.15, 0.4)
    values = _PAIR_VALUES[pair_name](target)
    code = _REVERSE_CODES[pair_name] if use_reverse_code else getattr(PropertyPair, pair_name)
    if use_reverse_code:
        values = values[::-1]

    phase = _water_at_reference_state()
    phase.Set(code, *values)

    if pair_name in ("PX", "TX"):
        assert phase.x() == pytest.approx(0.4, rel=1e-7, abs=1e-9)
    else:
        assert phase.T == pytest.approx(target.T, rel=2e-6, abs=2e-5)
        assert phase.P() == pytest.approx(target.P(), rel=2e-6, abs=1e-3)
