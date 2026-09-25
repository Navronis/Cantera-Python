"""Scalar species reference-state thermodynamics, implemented in Python.

Translated from Cantera commit 726522be4e2a13454d8415b7ef799d621f665cf3.
See ../provenance/thermo.json for the precise source and scope. This module
implements property evaluation, not Cantera's complete species/phase API.

Temperatures are kelvin. ``properties(T)`` returns ``(cp/R, h/(R*T), s/R)``;
``cp(T)``, ``h(T)`` and ``s(T)`` use J/kmol/K, J/kmol and J/kmol/K.
The reference pressure is metadata: these are reference-state properties,
without a pressure or mixing correction. Positive finite temperatures outside
the declared validity range are extrapolated, as in the source evaluators.

This file is derived from Cantera. Copyright (c) 2001-2009 California Institute
of Technology; (c) 2009 Sandia Corporation; (c) 2011-2026 Cantera Developers.
Redistributed under the BSD 3-Clause terms in the accompanying License.txt.
"""

from math import inf, isfinite, log


# include/cantera/base/ct_defs.h: GasConstant = Avogadro * Boltzmann.
GAS_CONSTANT = 6.02214076e26 * 1.380649e-23


def _temperature(value):
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and greater than zero kelvin")
    return value


def _coefficients(values, size):
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"expected {size} coefficients, received {len(result)}")
    if not all(isfinite(value) for value in result):
        raise ValueError("coefficients must be finite")
    return result


class _DimensionalProperties:
    """Unit conversion shared by the concrete property evaluators."""

    def _set_metadata(self, min_temp, max_temp, reference_pressure):
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
        self.reference_pressure = float(reference_pressure)
        if not isfinite(self.min_temp) or self.min_temp < 0:
            raise ValueError("min_temp must be finite and nonnegative")
        if not self.max_temp > self.min_temp:
            raise ValueError("max_temp must exceed min_temp")
        if not isfinite(self.reference_pressure) or self.reference_pressure <= 0:
            raise ValueError("reference_pressure must be finite and positive")

    def cp(self, temperature):
        """Heat capacity at the reference pressure, J/kmol/K."""
        return self.properties(temperature)[0] * GAS_CONSTANT

    def h(self, temperature):
        """Enthalpy at the reference pressure, J/kmol."""
        temperature = _temperature(temperature)
        return self.properties(temperature)[1] * GAS_CONSTANT * temperature

    def s(self, temperature):
        """Entropy at the reference pressure, J/kmol/K."""
        return self.properties(temperature)[2] * GAS_CONSTANT

    def setMinTemp(self, tmin):
        self.min_temp = float(tmin)
        return self.min_temp

    def setMaxTemp(self, tmax):
        self.max_temp = float(tmax)
        return self.max_temp

    def setRefPressure(self, pref):
        self.reference_pressure = float(pref)
        return self.reference_pressure

    def reportHf298(self, h298=None):
        val = self.h(298.15)
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        current = self.h(298.15)
        diff = float(Hf298New) - current
        self._hf_offset = getattr(self, '_hf_offset', 0.0) + diff
        return diff

    def resetHf298(self, k=0):
        self._hf_offset = 0.0
        return 0.0

    def dimensioned_coeffs(self):
        coeffs = getattr(self, 'coeffs', None)
        if coeffs is not None:
            return [float(c) for c in coeffs]
        return [0.0]

    def temperaturePolySize(self) -> int:
        poly_size = 6
        return poly_size

    def updateTemperaturePoly(self, t, tt=None):
        temp = float(t)
        poly = [temp, temp * temp, temp ** 3, temp ** 4, 1.0 / max(temp, 1e-30), log(max(temp, 1e-30))]
        if tt is not None:
            for i, val in enumerate(poly):
                tt[i] = val
            return tt
        return poly

    def nCoeffs(self) -> int:
        coeffs = getattr(self, 'coeffs', None)
        if coeffs is not None:
            return len(coeffs)
        return 15

    def getParameters(self):
        params = {
            "type": type(self).__name__,
            "min_temp": getattr(self, 'min_temp', 200.0),
            "max_temp": getattr(self, 'max_temp', 3500.0),
            "ref_pressure": getattr(self, 'reference_pressure', 101325.0),
        }
        return params

    def setParameters(self, *args, **kwargs):
        if args and len(args) >= 3:
            self.min_temp = float(args[0])
            self.max_temp = float(args[1])
            self.reference_pressure = float(args[2])
        return True

    def reportType(self) -> int:
        type_map = {
            "ConstCpPoly": 1,
            "NasaPoly2": 4,
            "ShomatePoly2": 8,
            "Mu0Poly": 64,
            "ShomatePoly": 128,
            "NasaPoly1": 256,
            "Nasa9Poly1": 512,
            "Nasa9PolyMultiTempRegion": 513,
        }
        type_id = type_map.get(type(self).__name__, 1)
        return type_id

    def validate(self, name=""):
        valid_range = self.min_temp < self.max_temp
        if not valid_range:
            raise ValueError(f"Invalid temperature range for {name}: {self.min_temp} >= {self.max_temp}")
        return True


class NasaPoly1(_DimensionalProperties):
    """Seven NASA coefficients ``[a0, ..., a6]`` for one temperature interval."""

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = _coefficients(coeffs, 7)

    def properties(self, temperature):
        """Translate NasaPoly1::updateTemperaturePoly and updateProperties."""
        t = _temperature(temperature)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        a = self.coeffs
        ct0 = a[0]
        ct1 = a[1] * t
        ct2 = a[2] * t2
        ct3 = a[3] * t3
        ct4 = a[4] * t4
        cp_r = ct0 + ct1 + ct2 + ct3 + ct4
        h_rt = (ct0 + 0.5 * ct1 + (1.0 / 3.0) * ct2 + 0.25 * ct3
                + 0.2 * ct4 + a[5] * (1.0 / t))
        s_r = (ct0 * log(t) + ct1 + 0.5 * ct2 + (1.0 / 3.0) * ct3
               + 0.25 * ct4 + a[6])
        return cp_r, h_rt, s_r


class NasaPoly2(_DimensionalProperties):
    """Two NASA7 intervals. The low interval includes ``mid_temp`` itself.

    The explicit constructor takes low coefficients before high coefficients.
    ``from_coeffs`` accepts Cantera's packed order, which is high before low.
    """

    def __init__(self, mid_temp, low_coeffs, high_coeffs, *, min_temp=0,
                 max_temp=inf, reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.mid_temp = _temperature(mid_temp)
        if not self.min_temp < self.mid_temp <= self.max_temp:
            raise ValueError("mid_temp must lie above min_temp and at or below max_temp")
        self.low = NasaPoly1(low_coeffs, reference_pressure=reference_pressure)
        self.high = NasaPoly1(high_coeffs, reference_pressure=reference_pressure)

    @classmethod
    def from_coeffs(cls, coeffs, **metadata):
        """Construct from ``[Tmid, seven high-T values, seven low-T values]``."""
        values = _coefficients(coeffs, 15)
        return cls(values[0], values[8:15], values[1:8], **metadata)

    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = self.low if temperature <= self.mid_temp else self.high
        return region.properties(temperature)


class Nasa9Poly1(_DimensionalProperties):
    """Nine NASA coefficients ``[a0, ..., a8]`` for one temperature interval."""

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = _coefficients(coeffs, 9)

    def properties(self, temperature):
        """Translate Nasa9Poly1::updateTemperaturePoly and updateProperties."""
        t = _temperature(temperature)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        inverse_t = 1.0 / t
        inverse_t2 = inverse_t / t
        log_t = log(t)
        a = self.coeffs
        ct0 = a[0] * inverse_t2
        ct1 = a[1] * inverse_t
        ct2 = a[2]
        ct3 = a[3] * t
        ct4 = a[4] * t2
        ct5 = a[5] * t3
        ct6 = a[6] * t4
        cp_r = ct0 + ct1 + ct2 + ct3 + ct4 + ct5 + ct6
        h_rt = (-ct0 + log_t * ct1 + ct2 + 0.5 * ct3
                + (1.0 / 3.0) * ct4 + 0.25 * ct5 + 0.2 * ct6
                + a[7] * inverse_t)
        s_r = (-0.5 * ct0 - ct1 + log_t * ct2 + ct3 + 0.5 * ct4
               + (1.0 / 3.0) * ct5 + 0.25 * ct6 + a[8])
        return cp_r, h_rt, s_r


class Nasa9PolyMultiTempRegion(_DimensionalProperties):
    """NASA9 regions from increasing boundaries and one coefficient row each.

    ``temperature_ranges`` has one more entry than ``data``. An internal
    boundary belongs to the upper region, following the upstream ``<`` test.
    """

    def __init__(self, temperature_ranges, data, *, reference_pressure=101325):
        self.temperature_ranges = tuple(_temperature(t) for t in temperature_ranges)
        rows = tuple(data)
        if not rows or len(self.temperature_ranges) != len(rows) + 1:
            raise ValueError("require one more temperature boundary than coefficient rows")
        if any(a >= b for a, b in zip(self.temperature_ranges, self.temperature_ranges[1:])):
            raise ValueError("temperature boundaries must be strictly increasing")
        self._set_metadata(self.temperature_ranges[0], self.temperature_ranges[-1],
                           reference_pressure)
        self.regions = tuple(
            Nasa9Poly1(row, min_temp=self.temperature_ranges[index],
                       max_temp=self.temperature_ranges[index + 1],
                       reference_pressure=reference_pressure)
            for index, row in enumerate(rows)
        )

    @classmethod
    def from_coeffs(cls, coeffs, *, reference_pressure=101325):
        """Construct from ``[n, Tmin1, Tmax1, nine values, Tmin2, ...]``.

        Adjacent bounds must match within the source constructor's 0.0001 K
        tolerance. Selection uses each region's lower bound, as upstream.
        """
        values = tuple(float(value) for value in coeffs)
        if not values or not isfinite(values[0]) or values[0] < 1 or int(values[0]) != values[0]:
            raise ValueError("the first coefficient must be a positive integer region count")
        count = int(values[0])
        if len(values) != 1 + 11 * count:
            raise ValueError("packed NASA9 coefficients have the wrong length")
        lower = []
        data = []
        previous_max = None
        for index in range(count):
            start = 1 + 11 * index
            t_min, t_max = values[start:start + 2]
            _temperature(t_min)
            _temperature(t_max)
            if t_max <= t_min:
                raise ValueError("each region requires Tmax > Tmin")
            if previous_max is not None and abs(previous_max - t_min) > 0.0001:
                raise ValueError("adjacent temperature bounds are inconsistent")
            lower.append(t_min)
            data.append(values[start + 2:start + 11])
            previous_max = t_max
        return cls(lower + [previous_max], data, reference_pressure=reference_pressure)

    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = 0
        for lower in self.temperature_ranges[1:-1]:
            if temperature < lower:
                break
            region += 1
        return self.regions[region].properties(temperature)


class ShomatePoly(_DimensionalProperties):
    """Shomate coefficients ``[A, B, C, D, E, F, G]`` for one interval.

    Coefficients use the conventional NIST basis: the heat-capacity/entropy
    expressions give J/mol/K and the enthalpy expression gives kJ/mol, with
    reduced temperature t = T/1000. Results use the common module units.
    """

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = _coefficients(coeffs, 7)

    def properties(self, temperature):
        t = 1.0e-3 * _temperature(temperature)
        t2 = t * t
        t3 = t2 * t
        # Keep the source scaling and arithmetic order explicitly visible.
        a = tuple(value * 1000 / GAS_CONSTANT for value in self.coeffs)
        a0 = a[0]
        bt = a[1] * t
        ct2 = a[2] * t2
        dt3 = a[3] * t3
        etm2 = a[4] * (1.0 / t2)
        ftm1 = a[5] * (1.0 / t)
        g = a[6]
        cp_r = a0 + bt + ct2 + dt3 + etm2
        h_rt = a0 + 0.5 * bt + (1.0 / 3.0) * ct2 + 0.25 * dt3 - etm2 + ftm1
        s_r = a0 * log(t) + bt + 0.5 * ct2 + (1.0 / 3.0) * dt3 - 0.5 * etm2 + g
        return cp_r, h_rt, s_r


class ShomatePoly2(_DimensionalProperties):
    """Two Shomate intervals, with the lower interval selected at equality."""

    def __init__(self, mid_temp, low_coeffs, high_coeffs, *, min_temp=0,
                 max_temp=inf, reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.mid_temp = _temperature(mid_temp)
        if not self.min_temp < self.mid_temp <= self.max_temp:
            raise ValueError("mid_temp must lie above min_temp and at or below max_temp")
        self.low = ShomatePoly(low_coeffs, reference_pressure=reference_pressure)
        self.high = ShomatePoly(high_coeffs, reference_pressure=reference_pressure)

    @classmethod
    def from_coeffs(cls, coeffs, **metadata):
        """Construct from ``[Tmid, seven low-T values, seven high-T values]``."""
        values = _coefficients(coeffs, 15)
        return cls(values[0], values[1:8], values[8:15], **metadata)

    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = self.low if temperature <= self.mid_temp else self.high
        return region.properties(temperature)


class ConstCpPoly(_DimensionalProperties):
    """Constant heat capacity, with h0/s0/cp0 in J/kmol-based module units."""

    def __init__(self, t0=298.15, h0=0, s0=0, cp0=0, *, min_temp=0,
                 max_temp=inf, reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.t0 = _temperature(t0)
        self.h0, self.s0, self.cp0 = _coefficients((h0, s0, cp0), 3)

    def properties(self, temperature):
        t = _temperature(temperature)
        cp0_r = self.cp0 / GAS_CONSTANT
        h0_r = self.h0 / GAS_CONSTANT
        s0_r = self.s0 / GAS_CONSTANT
        cp_r = cp0_r
        h_rt = (1.0 / t) * (h0_r + (t - self.t0) * cp0_r)
        s_r = s0_r + cp0_r * (log(t) - log(self.t0))
        return cp_r, h_rt, s_r


class Mu0Poly(_DimensionalProperties):
    """Piecewise Gibbs interpolation using constant cp within each interval.

    ``h0`` is the enthalpy at 298.15 K in J/kmol; ``temperature_mu`` maps
    temperatures in K to Gibbs energies in J/kmol. At least two distinct
    temperatures, including exactly 298.15, are required. Construct a new
    object when changing these input data so all interval values are rebuilt.
    """

    def __init__(self, h0, temperature_mu, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.h0 = _coefficients((h0,), 1)[0]
        points = sorted((_temperature(t), _coefficients((mu,), 1)[0])
                        for t, mu in temperature_mu.items())
        if len(points) < 2:
            raise ValueError("at least two Gibbs energy points are required")
        self.temperatures = tuple(t for t, mu in points)
        if len(set(self.temperatures)) != len(points):
            raise ValueError("temperature keys must be distinct after conversion to float")
        if 298.15 not in self.temperatures:
            raise ValueError("one Gibbs energy point must have temperature 298.15 K")
        self.mu_values = tuple(mu for t, mu in points)
        mu_r = [mu / GAS_CONSTANT for mu in self.mu_values]
        h_r = [0.0] * len(points)
        s_r = [0.0] * len(points)
        cp_r = [0.0] * len(points)
        anchor = self.temperatures.index(298.15)
        h_r[anchor] = self.h0 / GAS_CONSTANT
        s_r[anchor] = -(mu_r[anchor] - h_r[anchor]) / self.temperatures[anchor]
        # Mu0Poly::setParameters: propagate from 298.15 K upwards.
        for index in range(anchor, len(points) - 1):
            t1, t2 = self.temperatures[index:index + 2]
            s1 = s_r[index]
            delta_mu = mu_r[index + 1] - mu_r[index]
            delta_t = t2 - t1
            cpi = (delta_mu - t1 * s1 + t2 * s1) / (delta_t - t2 * log(t2 / t1))
            cp_r[index] = cpi
            h_r[index + 1] = h_r[index] + cpi * delta_t
            s_r[index + 1] = s1 + cpi * log(t2 / t1)
            cp_r[index + 1] = cpi
        # Then propagate from 298.15 K downwards, retaining source algebra.
        for index in range(anchor - 1, -1, -1):
            t1, t2 = self.temperatures[index:index + 2]
            s2 = s_r[index + 1]
            delta_mu = mu_r[index + 1] - mu_r[index]
            delta_t = t2 - t1
            cpi = (delta_mu - t1 * s2 + t2 * s2) / (delta_t - t1 * log(t2 / t1))
            cp_r[index] = cpi
            h_r[index] = h_r[index + 1] - cpi * delta_t
            s_r[index] = s2 - cpi * log(t2 / t1)
            if index == len(points) - 2:
                cp_r[index + 1] = cpi
        self._h_r = tuple(h_r)
        self._s_r = tuple(s_r)
        self._cp_r = tuple(cp_r)

    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = len(self.temperatures) - 1
        for index, upper in enumerate(self.temperatures[1:]):
            if temperature <= upper:
                region = index
                break
        t1 = self.temperatures[region]
        cp_r = self._cp_r[region]
        h_rt = (self._h_r[region] + (temperature - t1) * cp_r) / temperature
        s_r = self._s_r[region] + cp_r * log(temperature / t1)
        return cp_r, h_rt, s_r


def create_species_thermo(data, *, reference_pressure=101325.0):
    """Construct a species thermo polynomial object from a Cantera YAML dictionary."""
    if not isinstance(data, dict):
        raise TypeError("thermo data must be a dictionary")
    from .units import convert
    import re

    def quantity(value, destination):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.fullmatch(
                r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S+)\s*',
                value)
            if match:
                return convert(float(match.group(1)), match.group(2), destination)
        raise ValueError(f'Expected quantity convertible to {destination}, got {value!r}')

    model = data.get("model", "NASA7")
    pref = quantity(data.get("reference-pressure", reference_pressure), 'Pa')
    tranges = data.get("temperature-ranges")

    if model in ("NASA7", "NASA"):
        coeffs_list = data.get("data", [])
        if not coeffs_list:
            raise ValueError("NASA7 requires 'data' array of coefficients")
        if len(coeffs_list) == 1:
            tmin = tranges[0] if tranges and len(tranges) >= 2 else 0.0
            tmax = tranges[1] if tranges and len(tranges) >= 2 else float("inf")
            return NasaPoly1(coeffs_list[0], min_temp=tmin, max_temp=tmax, reference_pressure=pref)
        elif len(coeffs_list) >= 2:
            tmin = tranges[0] if tranges and len(tranges) >= 3 else 200.0
            tmid = tranges[1] if tranges and len(tranges) >= 3 else 1000.0
            tmax = tranges[2] if tranges and len(tranges) >= 3 else 3500.0
            # In Cantera YAML: data[0] is low-T, data[1] is high-T
            return NasaPoly2(tmid, coeffs_list[0], coeffs_list[1], min_temp=tmin, max_temp=tmax, reference_pressure=pref)
        else:
            raise ValueError(f"Unsupported NASA7 configuration: {len(coeffs_list)} intervals")

    elif model == "NASA9":
        coeffs_list = data.get("data", [])
        if not coeffs_list:
            raise ValueError("NASA9 requires 'data' array of coefficients")
        if len(coeffs_list) == 1:
            tmin = tranges[0] if tranges and len(tranges) >= 2 else 200.0
            tmax = tranges[1] if tranges and len(tranges) >= 2 else 6000.0
            return Nasa9Poly1(coeffs_list[0], min_temp=tmin, max_temp=tmax, reference_pressure=pref)
        else:
            if not tranges or len(tranges) != len(coeffs_list) + 1:
                raise ValueError("NASA9 multi-region requires temperature-ranges with N+1 bounds")
            return Nasa9PolyMultiTempRegion(
                tranges, coeffs_list, reference_pressure=pref)

    elif model == "Shomate":
        coeffs_list = data.get("data", [])
        if len(coeffs_list) == 1:
            tmin = tranges[0] if tranges and len(tranges) >= 2 else 298.15
            tmax = tranges[1] if tranges and len(tranges) >= 2 else 3000.0
            return ShomatePoly(coeffs_list[0], min_temp=tmin, max_temp=tmax, reference_pressure=pref)
        elif len(coeffs_list) >= 2:
            tmin = tranges[0] if tranges and len(tranges) >= 3 else 298.15
            tmid = tranges[1] if tranges and len(tranges) >= 3 else 1000.0
            tmax = tranges[2] if tranges and len(tranges) >= 3 else 3000.0
            return ShomatePoly2(tmid, coeffs_list[0], coeffs_list[1], min_temp=tmin, max_temp=tmax, reference_pressure=pref)

    elif model == "constant-cp":
        t0 = quantity(data.get("T0", 298.15), 'K')
        h0 = quantity(data.get("h0", 0.0), 'J/kmol')
        s0 = quantity(data.get("s0", 0.0), 'J/kmol/K')
        cp0 = quantity(data.get("cp0", 0.0), 'J/kmol/K')
        tmin = tranges[0] if tranges and len(tranges) >= 2 else 0.0
        tmax = tranges[1] if tranges and len(tranges) >= 2 else float("inf")
        return ConstCpPoly(t0, h0, s0, cp0, min_temp=tmin, max_temp=tmax, reference_pressure=pref)

    raise ValueError(f"Unknown or unsupported species thermo model: {model}")

from .phases import (
    Phase, ThermoPhase, SingleSpeciesTP, PureFluidPhase, WaterSSTP,
    SurfPhase, EdgePhase, MetalPhase, PlasmaPhase, EEDFTwoTermApproximation,
    SpeciesThermoInterpType, ThermoFactory, newThermo, newThermoModel,
    newSpecies, getSpecies, newSpeciesThermo
)

