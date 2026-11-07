"""Test failure semantics for remaining Transport models:
- UnityLewisTransport
- IonGasTransport
- WaterTransport
- TransportFactory
"""
import pytest
import numpy as np

import cantera_python as ct
from cantera_python.transport import (
    UnityLewisTransport, IonGasTransport, WaterTransport,
    TransportFactory, newTransport, Transport
)
from cantera_python.constants import ArraySizeError, CanteraError

def test_unity_lewis_failure_semantics():
    gas = ct.Solution("gri30.yaml", "gri30")
    gas.transport_model = "unity-Lewis-number"
    tran = gas.transport

    # 1. getMixDiffCoeffsMole is not implemented in unity-Lewis
    with pytest.raises(NotImplementedError):
        tran.getMixDiffCoeffsMole()

    # 2. Array size check for getMixDiffCoeffs
    small_arr = np.zeros(gas.n_species - 2)
    with pytest.raises(ArraySizeError):
        tran.getMixDiffCoeffs(small_arr)

    # 3. Check species index out of range
    with pytest.raises(IndexError):
        tran.checkSpeciesIndex(-1)
    with pytest.raises(IndexError):
        tran.checkSpeciesIndex(gas.n_species + 5)

def test_ion_gas_failure_semantics():
    sol = ct.Solution("ch4_ion.yaml", "gas")
    sol.transport_model = "ionized-gas"
    tran = sol.transport

    # 1. CK_Mode rejection / false
    assert tran.CKMode() is False

    # 2. Array size check for getMobilities
    small_mobi = np.zeros(sol.n_species - 1)
    with pytest.raises(ArraySizeError):
        tran.getMobilities(small_mobi)

    # 3. Array size check for getMixDiffCoeffs
    small_diff = np.zeros(sol.n_species - 1)
    with pytest.raises(ArraySizeError):
        tran.getMixDiffCoeffs(small_diff)

    # 4. Out-of-bounds species index
    with pytest.raises(IndexError):
        tran.checkSpeciesIndex(-1)
    with pytest.raises(IndexError):
        tran.checkSpeciesIndex(sol.n_species + 10)

def test_water_transport_semantics():
    sol = ct.Solution("water.yaml")
    tran = WaterTransport(sol.thermo)
    assert tran.transport_model == "Water"
    assert tran.bulk_viscosity == 0.0

def test_transport_factory_failure_semantics():
    gas = ct.Solution("gri30.yaml", "gri30")
    fac = TransportFactory.factory()

    # Unknown transport model
    with pytest.raises((ValueError, KeyError, CanteraError)):
        fac.newTransport("nonexistent_transport_model_xyz", gas.thermo)
