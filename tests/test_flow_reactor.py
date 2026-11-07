import numpy as np
import pytest
import cantera_python as ct
from cantera_python.reactor import (
    Reactor, IdealGasReactor, ConstPressureReactor, IdealGasConstPressureReactor,
    FlowReactor, ReactorSurface
)


def test_reactor_types_instantiation():
    gas = ct.Solution("gri30.yaml")
    
    r1 = Reactor(gas)
    assert r1.type() == "Reactor"
    
    r2 = IdealGasReactor(gas)
    assert r2.type() == "IdealGasReactor"
    
    r3 = ConstPressureReactor(gas)
    assert r3.type() == "ConstPressureReactor"
    
    r4 = IdealGasConstPressureReactor(gas)
    assert r4.type() == "IdealGasConstPressureReactor"


def test_flow_reactor_dae_eval():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0"

    fl = FlowReactor(gas)
    assert fl.type() == "FlowReactor"
    assert fl.timeIsIndependent() is False
    assert fl.componentIndex("density") == 0
    assert fl.componentIndex("speed") == 1
    assert fl.componentIndex("pressure") == 2
    assert fl.componentIndex("temperature") == 3
    assert fl.componentName(0) == "density"
    assert fl.componentName(4) == gas.species_names[0]

    fl.setArea(0.01)
    assert fl.area() == 0.01
    fl.setMassFlowRate(0.5)

    # Evaluate DAE residual
    neq = 4 + gas.n_species
    y = np.zeros(neq)
    y[0] = gas.density
    y[1] = 10.0  # speed
    y[2] = ct.OneAtm
    y[3] = 1200.0
    y[4:] = gas.Y

    ydot = np.zeros(neq)
    res = np.zeros(neq)
    fl.evalDae(0.0, y, ydot, res)

    # In Cantera FlowReactor.cpp: residual[0] = m_rho - m_thermo->density()
    assert np.isclose(res[0], 0.0, atol=1e-10)


def test_reactor_surface():
    surf = ReactorSurface()
    assert surf.type() == "ReactorSurface"
    assert surf.area() == 1.0
