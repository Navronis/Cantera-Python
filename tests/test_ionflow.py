import numpy as np
import pytest
import cantera_python as ct
from cantera_python.oned import IonFlow

def test_ionflow_instantiation_and_electric_field():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1500.0, ct.OneAtm, "CH4:1.0, O2:2.0"

    ion_flow = IonFlow(gas, id="flame_with_ions", points=5)
    assert ion_flow.domainType() in ("IonFlow", "unstrained-ion-flow")
    assert ion_flow.doElectricField() is False

    ion_flow.solveElectricField()
    assert ion_flow.doElectricField() is True

    ion_flow.fixElectricField()
    assert ion_flow.doElectricField() is False

    # Electron transport profile
    tfix = [300.0, 1000.0, 2000.0]
    d_e = [0.4, 0.4, 0.4]
    m_e = [0.1, 0.1, 0.1]
    ion_flow.setElectronTransport(tfix, d_e, m_e)
