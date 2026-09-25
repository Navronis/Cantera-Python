"""Selected thermodynamic state inversion equations from Cantera.

Only the fixed-composition ideal-gas ``HP`` path is implemented. It returns a
new immutable state instead of mutating a ``ThermoPhase`` object.
"""
from math import isfinite

from .constants import GasConstant as R
from .ideal_gas import IdealGasMixture


def _clip(value, lower, upper):
    return max(lower, min(value, upper))


def set_state_tp(phase, temperature, pressure):
    """Return a fixed-composition ideal-gas state at temperature and pressure.

    This is the direct setter for temperature and pressure.
    """
    if not isinstance(phase, IdealGasMixture) and not all(
            hasattr(phase, name) for name in ('temperature', 'mean_molecular_weight', 'at')):
        raise TypeError('phase must provide ideal-gas state properties')
    temperature = float(temperature)
    pressure = float(pressure)
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive, K')
    if not isfinite(pressure) or pressure <= 0:
        raise ValueError('Pressure must be finite and positive, Pa')
    return phase.at(temperature=temperature, pressure=pressure)


def set_state_dp(phase, density, pressure):
    """Return a fixed-composition ideal-gas state at density and pressure.

    Solves the ideal gas relation T = P * meanMW / (R * rho). Validation
    follows pinned ``IdealGasPhase::setState_DP`` order: pressure, density,
    then the derived temperature. Since this scalar API is immutable, only
    the mutable ``IdealGasPhase`` wrapper exposes the source's partial state
    mutation when the final temperature validation fails.
    """
    if not isinstance(phase, IdealGasMixture):
        raise TypeError('phase must be an IdealGasMixture')
    density = float(density)
    pressure = float(pressure)
    if pressure <= 0.0:
        raise ValueError('pressure must be positive')
    if not density > 0.0:
        raise ValueError(f'density must be positive. density = {density}')
    denominator = R * density
    numerator = pressure * phase.mean_molecular_weight
    temperature = float('inf') if denominator == 0.0 and numerator > 0.0 \
        else numerator / denominator
    if not temperature > 0.0:
        raise ValueError(f'temperature must be positive. T = {temperature}')
    return phase._at_density_unchecked(temperature, density)


def set_state_td(phase, temperature, density):
    """Return a fixed-composition ideal-gas state at temperature and density.

    This direct setter is the immutable form of the ``IdealGasPhase`` density
    relation ``rho = P * meanMW / (R*T)`` solved for pressure.
    """
    if not isinstance(phase, IdealGasMixture):
        raise TypeError('phase must be an IdealGasMixture')
    temperature = float(temperature)
    density = float(density)
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive, K')
    if not isfinite(density) or density <= 0:
        raise ValueError('Density must be finite and positive, kg/m3')
    pressure = density * R * temperature / phase.mean_molecular_weight
    return phase.at(temperature=temperature, pressure=pressure)


def _inverse_native(phase, target, second, *, quantity, rtol, max_iterations, trace=None):
    """Translate ThermoPhase::setState_HPorUV / setState_SPorSV.

    ``quantity`` is ``H``, ``U``, ``S_P`` or ``S_V``.  The structure mirrors
    the pinned source: bounded 100 K Newton steps, monotonicity brackets,
    stability retries, and the two source convergence tests.  ``phase.at``
    supplies the state-restoration boundary of the public immutable API.
    """
    do_volume = quantity in ('U', 'S_V')
    if not isinstance(phase, IdealGasMixture):
        raise TypeError('phase must be an IdealGasMixture')
    if not isfinite(target):
        raise ValueError(f'Target {quantity} must be finite')
    if not isfinite(second) or second < 1.0e-300:
        label = 'specific volume' if do_volume else ('pressure')
        raise ValueError(f'Input {label} is too small or negative. {label[0]} = {second}')
    if not isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive')
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    enforced = bool(getattr(phase, 'temperatureLimitsEnforced', lambda: False)())
    max_temp = float(getattr(phase, 'max_temp', 1e4))
    min_temp = float(getattr(phase, 'min_temp', 1.0))
    tmax = max_temp + 0.1 if enforced else max(max_temp + 1000.0, 10.0 * max_temp)
    tmin = min_temp - 0.1 if enforced else max(min_temp, 1e-300, min(100.0, min_temp))
    if tmax <= tmin:
        tmax = tmin + 20.0

    def at_temperature(t):
        if do_volume:
            p = R * t / (phase.mean_molecular_weight * second)
        else:
            p = second
        return phase.at(temperature=t, pressure=p)

    tnew = float(phase.temperature)
    tinit = tnew
    if tnew > tmax:
        tnew = tmax - 1.0
    elif tnew < tmin:
        tnew = tmin + 1.0
    current = at_temperature(tnew)
    value_name = 'int_energy_mass' if quantity == 'U' else 'enthalpy_mass' if quantity == 'H' else 'entropy_mass'
    deriv_name = 'cv_mass' if do_volume else 'cp_mass'
    value = float(getattr(current, value_name))
    deriv = float(getattr(current, deriv_name))
    top_value = bot_value = value
    top_t = bot_t = tnew
    ignore_bounds = False
    unstable = False
    unstable_last = -1.0
    for _ in range(max_iterations):
        iteration = _
        told, old_value, old_deriv = tnew, value, deriv
        if old_deriv < 0.0:
            unstable = True
            unstable_last = told
        if quantity in ('S_P', 'S_V'):
            dt = _clip((target - old_value) * told / old_deriv, -100.0, 100.0)
        else:
            dt = _clip((target - old_value) / old_deriv, -100.0, 100.0)
        raw_dt = (target - old_value) * told / old_deriv if quantity in ('S_P', 'S_V') else (target - old_value) / old_deriv
        branch = 'newton'
        safeguard = False
        tnew = told + dt
        if ((dt > 0 and unstable) or (dt <= 0 and not unstable)):
            if bot_value < target and tnew < (0.75 * bot_t + 0.25 * told):
                dt = 0.75 * (bot_t - told)
                tnew = told + dt
                branch, safeguard = 'lower_bracket_safeguard', True
        elif top_value > target and tnew > (0.75 * top_t + 0.25 * told):
            dt = 0.75 * (top_t - told)
            tnew = told + dt
            branch, safeguard = 'upper_bracket_safeguard', True
        if tnew > tmax and not ignore_bounds:
            edge = at_temperature(tmax)
            edge_value = float(getattr(edge, value_name))
            if edge_value >= target:
                if top_value < target:
                    top_t, top_value = tmax, edge_value
                tnew, dt = tmax, tmax - told
            elif enforced:
                raise ValueError(f'Target {quantity} = {target} cannot be reached within temperature bounds of {tmin} K to {tmax} K')
            else:
                tnew, ignore_bounds = tmax + 1.0, True
        if tnew < tmin and not ignore_bounds:
            edge = at_temperature(tmin)
            edge_value = float(getattr(edge, value_name))
            if edge_value <= target:
                if bot_value > target:
                    bot_t, bot_value = tmin, edge_value
                tnew, dt = tmin, tmin - told
            elif enforced:
                raise ValueError(f'Target {quantity} = {target} cannot be reached within temperature bounds of {tmin} K to {tmax} K')
            else:
                tnew, ignore_bounds = tmin - 1.0, True
        for _ in range(10):
            tnew = told + dt
            if tnew < told / 3.0:
                tnew, dt = told / 3.0, -2.0 * told / 3.0
            candidate = at_temperature(tnew)
            value = float(getattr(candidate, value_name))
            deriv = float(getattr(candidate, deriv_name))
            if deriv >= 0:
                break
            unstable_last = tnew
            if not unstable:
                dt *= 0.25
                branch, safeguard = 'negative_heat_capacity_retry', True
        if trace is not None:
            trace.append({'iteration': iteration, 'temperature': told,
                          'calculated_property': old_value, 'target_property': target,
                          'residual': target - old_value, 'derivative': old_deriv,
                          'raw_newton_step': raw_dt, 'clipped_step': dt,
                          'lower_bracket': [bot_t, bot_value],
                          'upper_bracket': [top_t, top_value], 'branch': branch,
                          'safeguard': safeguard, 'accepted_temperature': tnew,
                          'converged': False})
        if value == target:
            if trace:
                trace[-1]['converged'] = True
            return candidate
        if value > target and (top_value < target or value < top_value):
            top_t, top_value = tnew, value
        elif value < target and (bot_value > target or value > bot_value):
            bot_t, bot_value = tnew, value
        error = target - value
        denom = max(abs(target), max(abs(old_deriv), 1.0e-5) * tnew)
        conv = abs(error * tnew / denom) if quantity in ('S_P', 'S_V') else abs(error / denom)
        if conv < rtol or abs(dt / tnew) < rtol:
            if trace:
                trace[-1]['converged'] = True
            return candidate
    kind = 'UV' if quantity == 'U' else 'SV' if quantity == 'S_V' else quantity.replace('_P', '')
    raise RuntimeError(f'No convergence in {max_iterations} iterations ({kind}); target={target}, T={tnew}, value={value}, dT={dt}')


def set_state_hp(phase, target_enthalpy_mass, pressure, *, rtol=1.0e-9,
                 max_iterations=500, trace=None):
    """Return an ideal-gas state with target mass enthalpy and pressure.

    This is the bounded ideal-gas subset of ``ThermoPhase::setState_HP`` and
    ``setState_HPorUV``. The Newton update is limited to 100 K, as upstream,
    and protects the positive-temperature state by not stepping below one
    third of the current temperature. Composition is held fixed.
    """
    return _inverse_native(phase, float(target_enthalpy_mass), float(pressure), quantity='H', rtol=float(rtol), max_iterations=max_iterations, trace=trace)


def set_state_sp(phase, target_entropy_mass, pressure, *, rtol=1.0e-9,
                 max_iterations=500, trace=None):
    """Return an ideal-gas state with target mass entropy and pressure.

    This is the bounded ideal-gas subset of ``ThermoPhase::setState_SP`` and
    ``setState_SPorSV``. Its Newton increment is ``(Starget-S)*T/Cp`` and is
    capped to plus or minus 100 K, exactly as in the source path.
    """
    if float(pressure) < 1.0e-300:
        raise ValueError('Input pressure is too small or negative')
    return _inverse_native(phase, float(target_entropy_mass), float(pressure), quantity='S_P', rtol=float(rtol), max_iterations=max_iterations, trace=trace)


def set_state_uv(phase, target_int_energy_mass, specific_volume, *,
                 rtol=1.0e-9, max_iterations=500, trace=None):
    """Return an ideal-gas state with target mass internal energy and volume.

    This is the ideal-gas ``UV`` subset of ``ThermoPhase::setState_UV`` and
    ``setState_HPorUV``. Specific volume is held constant, so every temperature
    trial derives its pressure from ``P = R*T / (mean_molecular_weight*v)``.
    """
    return _inverse_native(phase, float(target_int_energy_mass), float(specific_volume), quantity='U', rtol=float(rtol), max_iterations=max_iterations, trace=trace)


def set_state_sv(phase, target_entropy_mass, specific_volume, *,
                 rtol=1.0e-9, max_iterations=500, trace=None):
    """Return an ideal-gas state with target mass entropy and volume.

    This is the ideal-gas ``SV`` subset of ``ThermoPhase::setState_SV`` and
    ``setState_SPorSV``. Specific volume is held constant, and temperature
    iteration uses ``cv_mass`` to compute increments: ``dt = (Starget - S) * T / cv``.
    """
    return _inverse_native(phase, float(target_entropy_mass), float(specific_volume), quantity='S_V', rtol=float(rtol), max_iterations=max_iterations, trace=trace)
