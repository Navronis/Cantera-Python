"""Unit tests for IonFlow and DomainFactory source-faithful implementations."""

import threading
import numpy as np
import pytest
from pytest import approx

import cantera_python as ct
from cantera_python.oned import (
    DomainFactory,
    IonFlow,
    Flow1D,
    Inlet1D,
    Outlet1D,
    newDomain,
    newBoundary1D,
    newFlow1D,
    c_offset_U,
    c_offset_V,
    c_offset_T,
    c_offset_L,
    c_offset_E,
    c_offset_Y,
)
from cantera_python.constants import CanteraError, Avogadro, ElectronCharge, epsilon_0


class TestDomainFactory:
    def test_factory_singleton_and_deletion(self):
        f1 = DomainFactory.factory()
        f2 = DomainFactory.factory()
        assert f1 is f2
        DomainFactory.deleteFactory()
        f3 = DomainFactory.factory()
        assert f3 is not f1

    def test_factory_dispatch_boundary_domains(self):
        gas = ct.Solution("h2o2.yaml")
        assert isinstance(newDomain("inlet", gas, "inlet_test"), Inlet1D)
        assert isinstance(newDomain("outlet", gas), Outlet1D)
        assert isinstance(newBoundary1D("inlet", gas), Inlet1D)

    def test_factory_dispatch_flow_domains(self):
        gas = ct.Solution("h2o2.yaml")
        flow = newDomain("free-flow", gas, "flame")
        assert isinstance(flow, Flow1D)
        assert flow.domain_type == "free-flow"

        # Ionized gas should dispatch to IonFlow
        gas_ion = ct.Solution("ch4_ion.yaml")
        ion_flow = newDomain("free-flow", gas_ion, "ion_flame")
        assert isinstance(ion_flow, IonFlow)
        assert ion_flow.domain_type == "free-ion-flow"

        ion_axisymm = newDomain("axisymmetric-flow", gas_ion)
        assert isinstance(ion_axisymm, IonFlow)
        assert ion_axisymm.domain_type == "axisymmetric-ion-flow"

        ion_unstrained = newDomain("unstrained-flow", gas_ion)
        assert isinstance(ion_unstrained, IonFlow)
        assert ion_unstrained.domain_type == "unstrained-ion-flow"

    def test_factory_invalid_types_and_casts(self):
        gas = ct.Solution("h2o2.yaml")
        with pytest.raises(CanteraError, match="Unknown domain type"):
            newDomain("invalid-nonexistent-domain", gas)

        with pytest.raises(CanteraError, match="Invalid cast"):
            newBoundary1D("free-flow", gas)

        with pytest.raises(CanteraError, match="Invalid cast"):
            newFlow1D("inlet", gas)


class TestIonFlow:
    @pytest.fixture
    def ion_setup(self):
        gas = ct.Solution("ch4_ion.yaml")
        gas.TPX = 1500.0, ct.OneAtm, "CH4:0.1, O2:0.2, N2:0.7"
        grid = np.linspace(0.0, 0.02, 6)
        flow = IonFlow(gas, id="ion_test", points=len(grid))
        flow.setupGrid(grid)
        return gas, flow, grid

    def test_charged_species_indexing(self, ion_setup):
        gas, flow, grid = ion_setup
        assert len(flow.m_kCharge) == 4
        assert len(flow.m_kNeutral) == 20
        assert flow.m_kElectron == gas.species_index("E")

        charged_names = {gas.species_name(k) for k in flow.m_kCharge}
        assert charged_names == {"HCO+", "H3O+", "E", "O2-"}

        assert flow.domainType() == "unstrained-ion-flow"
        flow.setFreeFlow()
        assert flow.domainType() == "free-ion-flow"
        flow.setAxisymmetricFlow()
        assert flow.domainType() == "axisymmetric-ion-flow"

    def test_electric_field_controls_and_refiner(self, ion_setup):
        gas, flow, grid = ion_setup
        assert flow.doElectricField() is False
        assert flow.analyticJacobianSupported() is False

        flow.solveElectricField()
        assert flow.doElectricField() is True
        assert flow.componentActive(c_offset_E) is True

        flow.fixElectricField()
        assert flow.doElectricField() is False

    def test_charge_density_and_gauss_law(self, ion_setup):
        gas, flow, grid = ion_setup
        sim = ct.Sim1D((ct.Inlet1D(gas), flow, ct.Outlet1D(gas)))
        x = np.zeros(sim.size(), dtype=float)

        nc = flow.nComponents()
        nsp = gas.n_species
        flow_loc = flow.loc()

        for j in range(len(grid)):
            base = flow_loc + j * nc
            x[base + c_offset_U] = 1.0
            x[base + c_offset_T] = 1500.0
            x[base + c_offset_E] = 50.0 * j
            # Small concentrations of ions
            x[base + c_offset_Y + gas.species_index("H3O+")] = 1.0e-5
            x[base + c_offset_Y + gas.species_index("E")] = 1.0e-6

        flow.updateThermo(x, 0, len(grid) - 1)
        flow.solveElectricField()

        # Check charge density rho_e
        rho_e_0 = flow.rho_e(x, 1)
        assert rho_e_0 != 0.0

        # Evaluate electric field residual
        rsd = np.zeros(sim.size(), dtype=float)
        diag = np.zeros(sim.size(), dtype=int)
        flow.evalElectricField(x, rsd, diag, 0.0, 0, len(grid) - 1)

        # At left boundary, rsd[E] = E(x, 0)
        assert rsd[flow_loc + c_offset_E] == approx(0.0)

        # In interior, rsd[E] = dEdz - rho_e / eps0
        j = 2
        base_j = flow_loc + j * nc
        dEdz_val = flow.dEdz(x, j)
        expected_rsd = dEdz_val - flow.rho_e(x, j) / epsilon_0
        assert rsd[base_j + c_offset_E] == approx(expected_rsd, rel=1e-7)

    def test_electron_transport_fitting(self, ion_setup):
        gas, flow, grid = ion_setup
        t_samples = np.linspace(300.0, 2500.0, 10)
        diff_samples = 1e-4 * (t_samples / 300.0) ** 1.5
        mobi_samples = 0.4 * np.ones_like(t_samples)

        flow.setElectronTransport(t_samples, diff_samples, mobi_samples)
        assert flow.m_import_electron_transport is True
        assert len(flow.m_diff_e_fix) == 6
        assert len(flow.m_mobi_e_fix) == 6
