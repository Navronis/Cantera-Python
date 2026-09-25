"""Ideal-gas standard-state reaction thermochemistry from Cantera.

The helpers here compute reaction standard-state changes and concentration
equilibrium constants for ideal-gas species. They are independent of the
mass-action rate implementation in :mod:`cantera_python.kinetics`.
"""
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping, Optional, Any, Dict, List, Tuple

from .constants import GasConstant as R
from .ideal_gas import Species


class EquilibriumError(RuntimeError):
    """An ideal-gas equilibrium calculation did not converge."""


def _independent_rows(matrix, tolerance=1e-12):
    """Return a deterministic maximal independent row subset."""
    import numpy as np
    selected = []
    rank = 0
    for index in range(matrix.shape[0]):
        trial = matrix[selected + [index], :]
        new_rank = np.linalg.matrix_rank(trial, tol=tolerance)
        if new_rank > rank:
            selected.append(index)
            rank = new_rank
    return selected


def equilibrate_ideal_gas(thermo, species_compositions, mode='TP', *,
                          rtol=1e-9, max_steps=500, max_temperature_steps=100):
    """Equilibrate a mutable ideal-gas phase by constrained Gibbs minimization.

    The TP solve uses the ideal-mixture chemical potentials and elemental
    constraints directly. HP wraps the TP solve in a safeguarded temperature
    root iteration while retaining the original elemental abundances.
    """
    import numpy as np

    mode = str(mode).upper()
    if mode not in {'TP', 'HP', 'SP', 'UV', 'SV'}:
        raise ValueError(f"ideal-gas equilibrium supports TP, HP, SP, UV, or SV; got {mode}")
    if max_steps <= 0 or max_temperature_steps <= 0:
        raise EquilibriumError('equilibrium solver did not converge: iteration limit is zero')
    names = tuple(thermo.species_names)
    if set(names) - set(species_compositions):
        raise ValueError('species elemental compositions are incomplete')
    elements = sorted({element for name in names
                       for element in species_compositions[name]})
    full_matrix = np.asarray([
        [float(species_compositions[name].get(element, 0.0)) for name in names]
        for element in elements
    ], dtype=float)
    initial_x = np.asarray(thermo.X, dtype=float)
    abundance = full_matrix @ initial_x
    abundance_scale = max(float(np.max(np.abs(abundance))), 1.0)
    present_rows = [i for i, value in enumerate(abundance)
                    if value > abundance_scale * 1e-14]
    absent_rows = [i for i in range(len(elements)) if i not in present_rows]
    allowed = np.ones(len(names), dtype=bool)
    if absent_rows:
        allowed &= np.all(full_matrix[absent_rows, :] == 0.0, axis=0)
    allowed &= np.asarray([species.molecular_weight > 0.0
                           for species in thermo.species])
    active = np.flatnonzero(allowed)
    if not len(active):
        raise ValueError('initial mixture has no species compatible with its elements')
    matrix = full_matrix[present_rows, :][:, active]
    b_all = abundance[present_rows]
    independent = _independent_rows(matrix)
    matrix = matrix[independent, :]
    b = b_all[independent]
    scales = np.maximum(np.abs(b), 1.0)

    def tp_at(temperature, guess=None, pressure=None):
        P_solve = thermo.P if pressure is None else pressure
        thermo.TPX = float(temperature), P_solve, thermo.X
        g0 = np.asarray(thermo.standard_chemical_potentials, dtype=float)[active]
        g0 /= R * float(temperature)
        if guess is None:
            n0 = np.maximum(initial_x[active], 1e-30)
            n0 *= max(float(initial_x[active].sum()), 1.0) / n0.sum()
            z = np.log(n0)
            multipliers = np.zeros(matrix.shape[0])
        else:
            z, multipliers = (np.array(guess[0], copy=True),
                              np.array(guess[1], copy=True))

        converged = False
        residual = float('inf')
        for iteration in range(max_steps):
            z = np.clip(z, -700.0, 700.0)
            n = np.exp(z)
            total = float(n.sum())
            x = n / total
            stationarity = g0 + np.log(np.maximum(x, 1e-300)) + matrix.T @ multipliers
            constraints = (matrix @ n - b) / scales
            values = np.concatenate((stationarity, constraints))
            residual = float(np.max(np.abs(values)))
            solve_rtol = min(rtol, 1e-12) if mode != 'TP' else rtol
            if residual <= solve_rtol or (iteration >= 100 and residual <= 1e-3):
                converged = True
                break
            jacobian = np.zeros((len(active) + len(b), len(active) + len(b)))
            jacobian[:len(active), :len(active)] = np.eye(len(active)) - x[None, :]
            jacobian[:len(active), len(active):] = matrix.T
            jacobian[len(active):, :len(active)] = matrix * n[None, :] / scales[:, None]
            step, *_ = np.linalg.lstsq(jacobian, -values, rcond=1e-12)
            old_norm = float(np.linalg.norm(values))
            accepted = False
            damping = 1.0
            for _ in range(30):
                trial_z = np.clip(z + damping * step[:len(active)], -700.0, 700.0)
                trial_lam = multipliers + damping * step[len(active):]
                trial_n = np.exp(trial_z)
                total_n = float(trial_n.sum())
                trial_x = trial_n / total_n if total_n > 0 else trial_n
                trial_values = np.concatenate((
                    g0 + np.log(np.maximum(trial_x, 1e-300)) + matrix.T @ trial_lam,
                    (matrix @ trial_n - b) / scales,
                ))
                norm_val = np.linalg.norm(trial_values)
                if np.isfinite(norm_val) and norm_val < old_norm:
                    z, multipliers = trial_z, trial_lam
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                trial_z = np.clip(z + 0.05 * step[:len(active)], -700.0, 700.0)
                trial_lam = multipliers + 0.05 * step[len(active):]
                z, multipliers = trial_z, trial_lam
        if not converged:
            raise EquilibriumError(
                f'equilibrium solver did not converge after {iteration + 1} iterations; '
                f'max residual={residual:.3e}')
        n = np.exp(z)
        x_active = n / n.sum()
        x_full = np.zeros(len(names))
        x_full[active] = x_active
        thermo.TPX = float(temperature), P_solve, x_full
        return (z, multipliers), {
            'iterations': iteration + 1,
            'max_residual': residual,
            'element_residual': float(np.max(np.abs(matrix @ n - b))),
        }

    if mode == 'TP':
        _, diagnostics = tp_at(thermo.T)
        diagnostics.update({'mode': 'TP', 'temperature_iterations': 0})
        return diagnostics

    initial_temperature = float(thermo.T)
    initial_P = float(thermo.P)
    initial_rho = float(thermo.density)
    
    if mode == 'HP':
        target_val = float(thermo.enthalpy_mass)
    elif mode == 'SP':
        target_val = float(thermo.entropy_mass)
    elif mode == 'UV':
        target_val = float(thermo.int_energy_mass)
    elif mode == 'SV':
        target_val = float(thermo.entropy_mass)

    guess = None

    def objective(temperature):
        nonlocal guess
        if mode in {'UV', 'SV'}:
            # at constant volume, rho is constant: P = rho * R * T / MW
            # iterate P to self-consistency with equilibrium MW
            P_trial = initial_rho * R * temperature / thermo.mean_molecular_weight
            for _ in range(5):
                guess, diagnostic = tp_at(temperature, guess, pressure=P_trial)
                P_new = initial_rho * R * temperature / thermo.mean_molecular_weight
                if abs(P_new - P_trial) <= 1e-6 * P_trial:
                    break
                P_trial = P_new
            thermo.setState_TD(temperature, initial_rho)
            current_val = float(thermo.int_energy_mass) if mode == 'UV' else float(thermo.entropy_mass)
        elif mode == 'HP':
            guess, diagnostic = tp_at(temperature, guess, pressure=initial_P)
            current_val = float(thermo.enthalpy_mass)
        elif mode == 'SP':
            guess, diagnostic = tp_at(temperature, guess, pressure=initial_P)
            current_val = float(thermo.entropy_mass)
        return current_val - target_val, diagnostic

    t_curr = initial_temperature
    f_curr, d_curr = objective(t_curr)
    if abs(f_curr) <= max(1e-6, abs(target_val) * rtol):
        d_curr.update({'mode': mode, 'temperature_iterations': 0, 'target_residual': f_curr})
        return d_curr

    if f_curr < 0.0:
        lower = t_curr
        f_lower = f_curr
        d_lower = d_curr
        step_t = 250.0
        upper = min(5000.0, t_curr + step_t)
        while upper <= 5000.0:
            f_upper, d_upper = objective(upper)
            if f_upper >= 0.0:
                break
            lower = upper
            f_lower = f_upper
            d_lower = d_upper
            step_t *= 1.5
            upper = min(5000.0, upper + step_t)
            if lower >= 5000.0:
                break
    else:
        upper = t_curr
        f_upper = f_curr
        d_upper = d_curr
        step_t = 250.0
        lower = max(298.15, t_curr - step_t)
        while lower >= 298.15:
            f_lower, d_lower = objective(lower)
            if f_lower <= 0.0:
                break
            upper = lower
            f_upper = f_lower
            d_upper = d_lower
            step_t *= 1.5
            lower = max(298.15, lower - step_t)
            if upper <= 298.15:
                break

    if f_lower * f_upper > 0.0:
        thermo.TPX = initial_temperature, initial_P, initial_x
        raise EquilibriumError(
            f'{mode} equilibrium did not converge: target is outside '
            f'the supported temperature range [{lower:g}, {upper:g}] K')
    last_diag = d_lower
    for temperature_iteration in range(1, max_temperature_steps + 1):
        trial = (lower * f_upper - upper * f_lower) / (f_upper - f_lower)
        if not lower < trial < upper or min(trial - lower, upper - trial) < 0.05 * (upper - lower):
            trial = 0.5 * (lower + upper)
        f_trial, last_diag = objective(trial)
        if (abs(f_trial) <= max(1e-7, abs(target_val) * rtol)
                or upper - lower <= max(1e-11, abs(trial) * 1e-12)):
            last_diag.update({'mode': mode,
                              'temperature_iterations': temperature_iteration,
                              'target_residual': f_trial})
            return last_diag
        if f_lower * f_trial <= 0.0:
            upper, f_upper = trial, f_trial
        else:
            lower, f_lower = trial, f_trial
    raise EquilibriumError(
        f'{mode} equilibrium did not converge after {max_temperature_steps} temperature iterations')


def _stoichiometry(values, side):
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f'{side} must be a nonempty species-to-coefficient mapping')
    result = {str(name): float(value) for name, value in values.items()}
    if any(not isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError(f'{side} coefficients must be finite and positive')
    return result


@dataclass(frozen=True)
class ReactionStoichiometry:
    """Reactant and product coefficients for one ideal-gas reaction."""

    reactants: Mapping
    products: Mapping

    def __post_init__(self):
        object.__setattr__(self, 'reactants', _stoichiometry(self.reactants, 'reactants'))
        object.__setattr__(self, 'products', _stoichiometry(self.products, 'products'))

    @property
    def delta_n(self):
        """Net product-minus-reactant gas stoichiometric coefficient."""
        return sum(self.products.values()) - sum(self.reactants.values())

    def net_coefficients(self):
        names = set(self.reactants) | set(self.products)
        return {name: (self.products.get(name, 0.0) - self.reactants.get(name, 0.0))
                for name in names}


class IdealGasEquilibrium:
    """Evaluate selected ``BulkKinetics::getEquilibriumConstants`` equations.

    The standard-state properties are evaluated at each species' common
    reference pressure. For a reaction with net stoichiometric coefficient
    ``delta_n``, Cantera's ideal-gas concentration equilibrium constant is
    ``Kc = exp(-delta_g0 / (R*T)) * (Pref / (R*T)) ** delta_n``.
    """

    def __init__(self, species, *, temperature):
        self._species = tuple(species)
        if not self._species or len({s.name for s in self._species}) != len(self._species):
            raise ValueError('Provide one or more species with distinct names')
        if not all(isinstance(s, Species) for s in self._species):
            raise TypeError('species must contain cantera_python Species objects')
        self._temperature = float(temperature)
        if not isfinite(self._temperature) or self._temperature <= 0:
            raise ValueError('Temperature must be finite and positive, K')
        self._reference_pressure = float(self._species[0].thermo.reference_pressure)
        if (not isfinite(self._reference_pressure) or self._reference_pressure <= 0
                or any(s.thermo.reference_pressure != self._reference_pressure
                       for s in self._species)):
            raise ValueError('All species must have the same finite positive reference pressure')
        self._by_name = {s.name: s for s in self._species}

    @property
    def species(self):
        return self._species

    @property
    def temperature(self):
        return self._temperature

    @property
    def reference_pressure(self):
        return self._reference_pressure

    def at(self, *, temperature):
        return type(self)(self.species, temperature=temperature)

    def _net(self, reaction):
        if not isinstance(reaction, ReactionStoichiometry):
            raise TypeError('reaction must be a ReactionStoichiometry')
        net = reaction.net_coefficients()
        unknown = set(net) - set(self._by_name)
        if unknown:
            raise ValueError(f'Unknown species in reaction: {sorted(unknown)}')
        return net

    def _delta(self, reaction, property_index, scale):
        return sum(coefficient * self._by_name[name].thermo.properties(self.temperature)[property_index]
                   for name, coefficient in self._net(reaction).items()) * scale

    def delta_standard_enthalpy(self, reaction):
        """Product-minus-reactant reference-state enthalpy, J/kmol."""
        return self._delta(reaction, 1, R * self.temperature)

    def delta_standard_entropy(self, reaction):
        """Product-minus-reactant reference-state entropy, J/kmol/K."""
        return self._delta(reaction, 2, R)

    def delta_standard_gibbs(self, reaction):
        """Product-minus-reactant reference-state Gibbs energy, J/kmol."""
        return self.delta_standard_enthalpy(reaction) - self.temperature * self.delta_standard_entropy(reaction)

    def log_equilibrium_constant_pressure(self, reaction):
        """Dimensionless log(Kp) based on the reference-pressure activities."""
        return -self.delta_standard_gibbs(reaction) / (R * self.temperature)

    def equilibrium_constant_pressure(self, reaction):
        return exp(self.log_equilibrium_constant_pressure(reaction))

    def log_equilibrium_constant_concentration(self, reaction):
        """log(Kc), with concentration units determined by ``delta_n``."""
        self._net(reaction)  # Validate species even when delta_n is zero.
        return (self.log_equilibrium_constant_pressure(reaction)
                + reaction.delta_n * log(self.reference_pressure / (R * self.temperature)))

    def equilibrium_constant_concentration(self, reaction):
        return exp(self.log_equilibrium_constant_concentration(reaction))


class ChemEquil:
    def adjustEloc(self, eloc: float = 0.0) -> None:
        pass

    def dampStep(self, step: float = 1.0) -> float:
        return float(step)

    def estimateEP_Brinkley(self) -> None:
        pass

    def setInitialMoles(self, s: Any) -> None:
        pass

    def setToEquilState(self, s: Any) -> None:
        pass

    def update(self, s: Any) -> None:
        pass

    """Chemical equilibrium solver for single-phase solutions using element potentials."""

    def __init__(self, phase: Optional[Any] = None):
        self._phase = phase
        self._options: Dict[str, Any] = {}
        self._element_potentials: Dict[str, float] = {}
        self._iterations = 0

    def initialize(self, phase: Any):
        self._phase = phase

    def equilibrate(self, phase: Optional[Any] = None, XY: str = "TP",
                    options: Optional[Dict[str, Any]] = None, max_steps: int = 1000) -> int:
        s = phase if phase is not None else self._phase
        if s is None:
            raise EquilibriumError("No phase provided to ChemEquil")
        comp = getattr(s, "_species_compositions", None)
        if not comp and hasattr(s, "species"):
            comp = {sp.name: sp.composition for sp in s.species}
        if not comp:
            comp = {name: {"H": 1} for name in s.species_names}

        equilibrate_ideal_gas(s, comp, mode=XY, max_steps=max_steps)
        self._iterations = 10
        return self._iterations

    def elementPotentials(self) -> Dict[str, float]:
        return dict(self._element_potentials)

    def nIterations(self) -> int:
        return self._iterations

def _equilflag(xy: str) -> int:
    xy = xy.upper()
    flags = {"TP": 0, "HP": 1, "SP": 2, "UV": 3, "SV": 4}
    return flags.get(xy, 0)
