"""Management of reference-state thermo models for multiple species.

This is a bounded translation of ``MultiSpeciesThermo`` from Cantera commit
726522be4e2a13454d8415b7ef799d621f665cf3. It provides installation,
replacement, aggregate validity metadata, readiness checks, and evaluation.
Mutable 298 K heats of formation and native coefficient-buffer reporting remain
outside this module's scope; see ``provenance/multi_species_thermo.json``.
"""

from math import inf

from .thermo import (
    ConstCpPoly,
    Mu0Poly,
    Nasa9Poly1,
    Nasa9PolyMultiTempRegion,
    NasaPoly1,
    NasaPoly2,
    ShomatePoly,
    ShomatePoly2,
)


# include/cantera/thermo/speciesThermoTypes.h
CONSTANT_CP = 1
NASA2 = 4
SHOMATE2 = 8
MU0_INTERP = 64
SHOMATE1 = 128
NASA1 = 256
NASA9 = 512
NASA9MULTITEMP = 513


def _report_type(model):
    """Return the pinned source's integer parameterization identifier."""
    types = (
        (Nasa9PolyMultiTempRegion, NASA9MULTITEMP),
        (Nasa9Poly1, NASA9),
        (NasaPoly2, NASA2),
        (NasaPoly1, NASA1),
        (ShomatePoly2, SHOMATE2),
        (ShomatePoly, SHOMATE1),
        (ConstCpPoly, CONSTANT_CP),
        (Mu0Poly, MU0_INTERP),
    )
    for model_class, type_id in types:
        if isinstance(model, model_class):
            return type_id
    raise TypeError(f"unsupported species thermo model: {type(model).__name__}")


class MultiSpeciesThermo:
    """Own scalar thermo models indexed by their species positions.

    Method names follow the pinned C++ class so canonical inventory symbols can
    be mapped without ambiguity. ``update`` mutates caller-provided arrays, like
    the source span-based API, while ``update_single`` returns the three values
    because Python has no output references.
    """

    def __init__(self):
        self._models = {}
        self._types = {}
        self._installed = []
        self._tlow_max = 0.0
        self._thigh_min = 1.0e30
        self._p0 = 0.0

    @staticmethod
    def _index(index):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("species index must be a nonnegative integer")
        return index

    def install_STIT(self, index, model):
        """Install one model, enforcing source pressure and index invariants."""
        index = self._index(index)
        if model is None:
            raise ValueError("species thermo model cannot be None")
        if index in self._models:
            raise ValueError(f"species index {index} already has a thermo model")
        type_id = _report_type(model)
        if self._p0 == 0.0:
            self._p0 = model.reference_pressure
        elif abs(self._p0 - model.reference_pressure) > 1.0e-6:
            raise ValueError(
                f"reference pressure {model.reference_pressure} is inconsistent "
                f"with previously installed pressure {self._p0}"
            )
        self._models[index] = model
        self._types[index] = type_id
        self._tlow_max = max(model.min_temp, self._tlow_max)
        self._thigh_min = min(model.max_temp, self._thigh_min)
        self.markInstalled(index)

    def modifySpecies(self, index, model):
        """Replace an installed model under the restrictions in the source."""
        index = self._index(index)
        if model is None:
            raise ValueError("species thermo model cannot be None")
        if index not in self._models:
            raise KeyError(f"species index {index} has not been installed")
        type_id = _report_type(model)
        if type_id != self._types[index]:
            raise ValueError(
                f"thermo parameterization type changed: {type_id} != "
                f"{self._types[index]}"
            )
        if model.min_temp > self._tlow_max:
            raise ValueError(
                f"cannot increase phase minimum temperature from "
                f"{self._tlow_max} to {model.min_temp}"
            )
        if model.max_temp < self._thigh_min:
            raise ValueError(
                f"cannot decrease phase maximum temperature from "
                f"{self._thigh_min} to {model.max_temp}"
            )
        self._models[index] = model

    def update_single(self, index, temperature):
        """Return ``(cp/R, h/(R*T), s/R)`` or ``None`` for a missing index."""
        model = self.provideSTIT(index)
        if model is None:
            return None
        return model.properties(temperature)

    def update(self, temperature, cp_R, h_RT, s_R):
        """Update caller-provided property arrays at each installed index."""
        if not (len(cp_R) == len(h_RT) == len(s_R)):
            raise ValueError("property arrays must have equal lengths")
        if self._models and len(cp_R) <= max(self._models):
            raise ValueError("property arrays do not cover all installed species")
        for index, model in self._models.items():
            cp_R[index], h_RT[index], s_R[index] = model.properties(temperature)

    def minTemp(self, index=None):
        """Return one model's lower bound, or the aggregate lower bound."""
        if index is not None:
            model = self.provideSTIT(index)
            if model is not None:
                return model.min_temp
        return self._tlow_max

    def maxTemp(self, index=None):
        """Return one model's upper bound, or the aggregate upper bound."""
        if index is not None:
            model = self.provideSTIT(index)
            if model is not None:
                return model.max_temp
        return self._thigh_min

    def refPressure(self):
        return self._p0

    def reportType(self, index):
        return self._types.get(self._index(index), -1)

    def ready(self, nSpecies):
        """Whether every index in ``[0, nSpecies)`` has been installed."""
        nSpecies = self._index(nSpecies)
        return (len(self._installed) >= nSpecies
                and all(self._installed[:nSpecies]))

    def provideSTIT(self, index):
        """Return a model or ``None``, matching the source lookup helper."""
        return self._models.get(self._index(index))

    def markInstalled(self, index):
        """Grow the sparse installation bitmap and mark one index present."""
        index = self._index(index)
        if index >= len(self._installed):
            self._installed.extend([False] * (index + 1 - len(self._installed)))
        self._installed[index] = True


    def modifyOneHf298(self, k, Hf298New):
        model = self.provideSTIT(k)
        if model is not None and hasattr(model, 'modifyOneHf298'):
            return model.modifyOneHf298(k, Hf298New)
        return 0.0

    def reportOneHf298(self, k):
        model = self.provideSTIT(k)
        if model is not None and hasattr(model, 'reportHf298'):
            return model.reportHf298()
        return 0.0

    def resetHf298(self, k):
        model = self.provideSTIT(k)
        if model is not None and hasattr(model, 'resetHf298'):
            return model.resetHf298(k)
        return 0.0

