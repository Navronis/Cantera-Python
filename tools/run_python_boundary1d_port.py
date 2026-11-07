"""Run deterministic differential validation cases through pure Python Boundary1D subsystem.

Reads:
  - audit/corpus_boundary1d.jsonl
Writes:
  - audit/python_boundary1d_port_results.json
"""

import json
import math
import sys
from pathlib import Path

# Add outputs/cantera-python to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import cantera_python as ct
from cantera_python.oned import (
    Boundary1D,
    Inlet1D,
    Outlet1D,
    OutletRes1D,
    Empty1D,
    Symm1D,
    Surf1D,
    ReactingSurf1D,
    Flow1D,
    FreeFlow,
    AxisymmetricFlow,
    UnstrainedFlow,
    OneDim,
    c_offset_U,
    c_offset_V,
    c_offset_T,
    c_offset_L,
    c_offset_E,
    c_offset_Uo,
    c_offset_Y,
)
from cantera_python.constants import CanteraError


class TestBoundary1D(Boundary1D):
    def __init__(self):
        super().__init__()
        self.m_index = -1


class TestReactingSurf1D(ReactingSurf1D):
    def testInit(self):
        self.m_nv = self.m_nsp
        self.resize(self.m_nsp, 1)
        self.m_fixed_cov = np.zeros(self.m_nsp, dtype=float)
        self.m_fixed_cov[0] = 1.0
        n_tot = getattr(self.m_kin, "n_total_species", self.m_nsp)
        self.m_work = np.zeros(n_tot, dtype=float)
        for n in range(self.m_nsp):
            self.setBounds(n, -1.0e-5, 2.0)


def run_port():
    corpus_path = repo_root / "audit" / "corpus_boundary1d.jsonl"
    output_path = repo_root / "audit" / "python_boundary1d_port_results.json"

    gas_mech = str(repo_root.parent / "cantera-original" / "data" / "h2o2.yaml")
    surf_mech = str(repo_root.parent / "cantera-original" / "data" / "ptcombust.yaml")

    results = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case_info = json.loads(line)
            case_id = case_info["case_id"]
            print(f"Running Python case: {case_id}")

            data = {}

            if case_id == "boundary1d_01_base_defaults":
                bdry = TestBoundary1D()
                data["domain_type"] = bdry.domainType()
                data["is_connector"] = bdry.isConnector()
                data["n_points"] = bdry.nPoints()
                data["n_components"] = bdry.nComponents()
                data["init_temp"] = bdry.temperature()
                bdry.setTemperature(500.0)
                data["set_temp"] = bdry.temperature()
                data["init_mdot"] = bdry.mdot()
                bdry.setMdot(1.25)
                data["set_mdot"] = bdry.mdot()
                data["n_species"] = bdry.nSpecies()

                caught_xin = False
                try:
                    bdry.setMoleFractions("H2:1")
                except NotImplementedError:
                    caught_xin = True

                caught_mass = False
                try:
                    bdry.massFraction(0)
                except NotImplementedError:
                    caught_mass = True

                caught_spread = False
                try:
                    bdry.setSpreadRate(10.0)
                except NotImplementedError:
                    caught_spread = True

                caught_getspread = False
                try:
                    bdry.spreadRate()
                except NotImplementedError:
                    caught_getspread = True

                data["caught_xin"] = caught_xin
                data["caught_mass"] = caught_mass
                data["caught_spread"] = caught_spread
                data["caught_getspread"] = caught_getspread

            elif case_id == "boundary1d_02_empty1d_lifecycle":
                sol = ct.Solution(gas_mech)
                flow = Flow1D(phase=sol)
                empty = Empty1D(phase=sol, name="empty")
                container = OneDim([flow, empty])
                flow.setupGrid([0.0, 0.01])
                container.resize()
                container.init()
                container.resize()

                data["domain_type"] = empty.domainType()
                data["is_connector"] = empty.isConnector()
                data["n_points"] = empty.nPoints()
                data["n_components"] = empty.nComponents()

                sz = container.size()
                xg = np.ones(sz, dtype=float)
                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                empty.eval(-1, xg, rg, diag, 0.0)

                data["eval_rg_sum"] = float(np.sum(np.abs(rg)))

            elif case_id == "boundary1d_03_inlet1d_properties_gas":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="fuel_inlet")
                data["domain_type"] = inlet.domainType()
                data["is_connector"] = inlet.isConnector()
                data["n_species"] = inlet.nSpecies()
                inlet.setTemperature(400.0)
                data["temperature"] = inlet.temperature()
                inlet.setMdot(2.5)
                data["mdot"] = inlet.mdot()
                inlet.setSpreadRate(12.3)
                data["spread_rate"] = inlet.spreadRate()
                inlet.setMoleFractions("H2:0.8, O2:0.2")

                data["mass_fractions"] = [inlet.massFraction(k) for k in range(inlet.nSpecies())]

            elif case_id == "boundary1d_04_inlet1d_properties_span":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="ox_inlet")
                inlet.setTemperature(320.0)
                inlet.setMdot(0.8)
                xin = [0.4, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2]
                inlet.setMoleFractions(xin)

                data["temperature"] = inlet.temperature()
                data["mdot"] = inlet.mdot()
                data["mass_fractions"] = [inlet.massFraction(k) for k in range(inlet.nSpecies())]

            elif case_id == "boundary1d_05_inlet1d_unstrained_eval":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="inlet")
                flow = UnstrainedFlow(phase=sol)
                outlet = Outlet1D(phase=sol, name="outlet")
                container = OneDim([inlet, flow, outlet])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()

                inlet.setTemperature(350.0)
                inlet.setMdot(0.15)
                inlet.setMoleFractions("H2:0.1, O2:0.9")
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                nsp = sol.n_species

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_U] = 0.10 + 0.01 * j
                    xg[pt + c_offset_V] = 0.0
                    xg[pt + c_offset_T] = 320.0 + 10.0 * j
                    xg[pt + c_offset_L] = 0.0
                    xg[pt + c_offset_Y] = 0.1
                    xg[pt + c_offset_Y + 3] = 0.9

                flow.setGas(xg[fl_loc : fl_loc + flow.size()], 0)

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                inlet.eval(-1, xg, rg, diag, 0.0)

                data["r_u"] = float(rg[fl_loc + c_offset_U])
                data["r_T"] = float(rg[fl_loc + c_offset_T])
                data["r_Y"] = [float(rg[fl_loc + c_offset_Y + k]) for k in range(nsp)]

            elif case_id == "boundary1d_06_inlet1d_free_flow_eval":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="inlet")
                flow = FreeFlow(phase=sol)
                outlet = Outlet1D(phase=sol, name="outlet")
                container = OneDim([inlet, flow, outlet])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()

                inlet.setTemperature(300.0)
                inlet.setMoleFractions("H2:0.25, O2:0.25, AR:0.5")
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                nsp = sol.n_species

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_U] = 0.25
                    xg[pt + c_offset_V] = 0.0
                    xg[pt + c_offset_T] = 310.0
                    xg[pt + c_offset_L] = 0.0
                    xg[pt + c_offset_Y] = 0.05
                    xg[pt + c_offset_Y + 3] = 0.40
                    xg[pt + c_offset_Y + 8] = 0.55

                flow.setGas(xg[fl_loc : fl_loc + flow.size()], 0)

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                inlet.eval(-1, xg, rg, diag, 0.0)

                data["computed_mdot"] = float(inlet.mdot())
                data["r_T"] = float(rg[fl_loc + c_offset_T])
                data["r_Y"] = [float(rg[fl_loc + c_offset_Y + k]) for k in range(nsp)]

            elif case_id == "boundary1d_07_inlet1d_strained_flow_left_eval":
                sol = ct.Solution(gas_mech)
                inlet_l = Inlet1D(phase=sol, name="inlet_left")
                flow = AxisymmetricFlow(phase=sol)
                inlet_r = Inlet1D(phase=sol, name="inlet_right")
                container = OneDim([inlet_l, flow, inlet_r])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()

                inlet_l.setTemperature(300.0)
                inlet_l.setMdot(0.5)
                inlet_l.setSpreadRate(20.0)
                inlet_l.setMoleFractions("H2:0.5, AR:0.5")

                inlet_r.setTemperature(450.0)
                inlet_r.setMdot(0.75)
                inlet_r.setSpreadRate(15.0)
                inlet_r.setMoleFractions("O2:0.21, AR:0.79")

                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                nsp = sol.n_species

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_U] = 0.1
                    xg[pt + c_offset_V] = 5.0
                    xg[pt + c_offset_T] = 350.0
                    xg[pt + c_offset_L] = 0.0
                    xg[pt + c_offset_Y + 8] = 1.0

                flow.setGas(xg[fl_loc : fl_loc + flow.size()], 0)

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                inlet_l.eval(-1, xg, rg, diag, 0.0)

                data["r_L"] = float(rg[fl_loc + c_offset_L])
                data["r_V"] = float(rg[fl_loc + c_offset_V])
                data["r_T"] = float(rg[fl_loc + c_offset_T])
                data["r_Y"] = [float(rg[fl_loc + c_offset_Y + k]) for k in range(nsp)]

            elif case_id == "boundary1d_08_inlet1d_strained_flow_right_eval":
                sol = ct.Solution(gas_mech)
                inlet_l = Inlet1D(phase=sol, name="inlet_left")
                flow = AxisymmetricFlow(phase=sol)
                inlet_r = Inlet1D(phase=sol, name="inlet_right")
                container = OneDim([inlet_l, flow, inlet_r])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()

                inlet_l.setTemperature(300.0)
                inlet_l.setMdot(0.5)
                inlet_l.setSpreadRate(20.0)
                inlet_l.setMoleFractions("H2:0.5, AR:0.5")

                inlet_r.setTemperature(450.0)
                inlet_r.setMdot(0.75)
                inlet_r.setSpreadRate(15.0)
                inlet_r.setMoleFractions("O2:0.21, AR:0.79")

                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                nsp = sol.n_species
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_U] = -0.2
                    xg[pt + c_offset_V] = 4.0
                    xg[pt + c_offset_T] = 400.0
                    xg[pt + c_offset_L] = 0.0
                    xg[pt + c_offset_Y + 8] = 1.0

                flow.setGas(xg[fl_loc : fl_loc + flow.size()], last_j)

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                inlet_r.eval(-1, xg, rg, diag, 0.0)

                data["r_U"] = float(rg[last_pt + c_offset_U])
                data["r_V"] = float(rg[last_pt + c_offset_V])
                data["r_T"] = float(rg[last_pt + c_offset_T])
                data["r_Y"] = [float(rg[last_pt + c_offset_Y + k]) for k in range(nsp)]

            elif case_id == "boundary1d_09_outlet1d_unstrained_eval":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="inlet")
                flow = Flow1D(phase=sol)
                outlet = Outlet1D(phase=sol, name="outlet")
                container = OneDim([inlet, flow, outlet])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                nsp = sol.n_species
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                prev_pt = fl_loc + flow.index(0, last_j - 1)

                xg[prev_pt + c_offset_T] = 1500.0
                xg[last_pt + c_offset_T] = 1550.0
                for k in range(nsp):
                    xg[prev_pt + c_offset_Y + k] = 0.05 * (k + 1)
                    xg[last_pt + c_offset_Y + k] = 0.05 * (k + 1) + 0.002 * (k + 1)

                rg = np.zeros(sz, dtype=float)
                diag = np.ones(sz, dtype=int)
                outlet.eval(-1, xg, rg, diag, 0.0)

                data["r_T"] = float(rg[last_pt + c_offset_T])
                data["r_Y"] = [float(rg[last_pt + c_offset_Y + k]) for k in range(nsp)]
                data["diag_Y"] = [int(diag[last_pt + c_offset_Y + k]) for k in range(nsp)]

            elif case_id == "boundary1d_10_outletres1d_unstrained_eval":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="inlet")
                flow = Flow1D(phase=sol)
                outlet_res = OutletRes1D(phase=sol, name="outlet-reservoir")
                container = OneDim([inlet, flow, outlet_res])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()
                outlet_res.setMoleFractions("H2O:0.2, AR:0.8")
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                nsp = sol.n_species
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                prev_pt = fl_loc + flow.index(0, last_j - 1)

                xg[prev_pt + c_offset_T] = 1200.0
                xg[last_pt + c_offset_T] = 1250.0
                for k in range(nsp):
                    xg[last_pt + c_offset_Y + k] = 0.08 * (k + 1)

                rg = np.zeros(sz, dtype=float)
                diag = np.ones(sz, dtype=int)
                outlet_res.eval(-1, xg, rg, diag, 0.0)

                data["r_T"] = float(rg[last_pt + c_offset_T])
                data["r_Y"] = [float(rg[last_pt + c_offset_Y + k]) for k in range(nsp)]
                data["diag_Y"] = [int(diag[last_pt + c_offset_Y + k]) for k in range(nsp)]

            elif case_id == "boundary1d_11_symm1d_right_eval":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="inlet")
                flow = AxisymmetricFlow(phase=sol)
                symm = Symm1D(phase=sol, name="symmetry-plane")
                container = OneDim([inlet, flow, symm])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                prev_pt = fl_loc + flow.index(0, last_j - 1)

                xg[prev_pt + c_offset_V] = 2.0
                xg[last_pt + c_offset_V] = 2.5
                xg[prev_pt + c_offset_T] = 1200.0
                xg[last_pt + c_offset_T] = 1220.0

                rg = np.zeros(sz, dtype=float)
                diag = np.ones(sz, dtype=int)
                symm.eval(-1, xg, rg, diag, 0.0)

                data["r_V"] = float(rg[last_pt + c_offset_V])
                data["diag_V"] = int(diag[last_pt + c_offset_V])
                data["r_T"] = float(rg[last_pt + c_offset_T])
                data["diag_T"] = int(diag[last_pt + c_offset_T])

            elif case_id == "boundary1d_12_symm1d_left_eval":
                sol = ct.Solution(gas_mech)
                symm = Symm1D(phase=sol, name="symmetry-plane")
                flow = AxisymmetricFlow(phase=sol)
                outlet = Outlet1D(phase=sol, name="outlet")
                container = OneDim([symm, flow, outlet])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                nc = flow.nComponents()
                p0 = fl_loc + flow.index(0, 0)
                p1 = fl_loc + flow.index(0, 1)

                xg[p0 + c_offset_V] = 1.0
                xg[p1 + c_offset_V] = 1.2
                xg[p0 + c_offset_T] = 300.0
                xg[p1 + c_offset_T] = 310.0

                rg = np.zeros(sz, dtype=float)
                diag = np.ones(sz, dtype=int)
                symm.eval(-1, xg, rg, diag, 0.0)

                data["r_V"] = float(rg[p0 + c_offset_V])
                data["diag_V"] = int(diag[p0 + c_offset_V])
                data["r_T"] = float(rg[p0 + c_offset_T])
                data["diag_T"] = int(diag[p0 + c_offset_T])

            elif case_id == "boundary1d_13_surf1d_left_eval":
                sol = ct.Solution(gas_mech)
                surf = Surf1D(phase=sol, name="surface")
                flow = Flow1D(phase=sol)
                outlet = Outlet1D(phase=sol, name="outlet")
                container = OneDim([surf, flow, outlet])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()
                surf.setTemperature(800.0)
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                p0 = fl_loc + flow.index(0, 0)
                xg[p0 + c_offset_T] = 750.0

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                surf.eval(-1, xg, rg, diag, 0.0)

                data["r_T"] = float(rg[p0 + c_offset_T])

            elif case_id == "boundary1d_14_surf1d_right_eval":
                sol = ct.Solution(gas_mech)
                inlet = Inlet1D(phase=sol, name="inlet")
                flow = Flow1D(phase=sol)
                surf = Surf1D(phase=sol, name="surface")
                container = OneDim([inlet, flow, surf])
                flow.setupGrid([0.0, 0.005, 0.01, 0.015, 0.02])
                container.resize()
                surf.setTemperature(650.0)
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                xg[last_pt + c_offset_T] = 680.0

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                surf.eval(-1, xg, rg, diag, 0.0)

                data["r_T"] = float(rg[last_pt + c_offset_T])

            elif case_id == "boundary1d_15_reactingsurf1d_properties":
                sol_surf = ct.Interface(surf_mech, "Pt_surf")
                rsurf = ReactingSurf1D(phase=sol_surf, name="reacting-surface")

                data["domain_type"] = rsurf.domainType()
                data["is_connector"] = rsurf.isConnector()
                data["cov_enabled_init"] = rsurf.coverageEnabled()
                rsurf.enableCoverageEquations(False)
                data["cov_enabled_false"] = rsurf.coverageEnabled()
                rsurf.enableCoverageEquations(True)
                data["cov_enabled_true"] = rsurf.coverageEnabled()

                nsp = sol_surf.n_species
                data["species_names"] = [rsurf.componentName(k) for k in range(nsp)]
                data["species_indices"] = [rsurf.componentIndex(rsurf.componentName(k)) for k in range(nsp)]

            elif case_id == "boundary1d_16_reactingsurf1d_bounds_resetbadvalues":
                sol_surf = ct.Interface(surf_mech, "Pt_surf")
                rsurf = TestReactingSurf1D(phase=sol_surf, name="test_rsurf")
                rsurf.testInit()

                nsp = sol_surf.n_species
                data["lower_bounds"] = [float(rsurf.lowerBound(k)) for k in range(nsp)]
                data["upper_bounds"] = [float(rsurf.upperBound(k)) for k in range(nsp)]

                cov = np.full(nsp, 0.5, dtype=float)
                rsurf.resetBadValues(cov)
                data["normalized_cov"] = [float(c) for c in cov]

            elif case_id == "boundary1d_17_reactingsurf1d_eval_steady":
                sol_gas = ct.Solution(surf_mech, "gas")
                sol_surf = ct.Interface(surf_mech, "Pt_surf", [sol_gas])
                flow = Flow1D(phase=sol_gas)
                rsurf = ReactingSurf1D(phase=sol_surf, name="reacting-surface")
                container = OneDim([flow, rsurf])
                flow.setupGrid([0.0, 0.01])
                container.resize()
                rsurf.setTemperature(900.0)
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                rs_loc = rsurf.loc()
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                nsp_gas = sol_gas.n_species

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_T] = 880.0
                    for k in range(nsp_gas):
                        xg[pt + c_offset_Y + k] = 1.0 / nsp_gas

                cov = [0.30, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10]
                for k in range(len(cov)):
                    xg[rs_loc + k] = cov[k]

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                rsurf.eval(-1, xg, rg, diag, 0.0)

                data["r_cov"] = [float(rg[rs_loc + k]) for k in range(len(cov))]
                data["diag_cov"] = [int(diag[rs_loc + k]) for k in range(len(cov))]
                data["r_gas_T"] = float(rg[last_pt + c_offset_T])
                data["r_gas_Y"] = [float(rg[last_pt + c_offset_Y + k]) for k in range(nsp_gas)]

            elif case_id == "boundary1d_18_reactingsurf1d_eval_transient":
                sol_gas = ct.Solution(surf_mech, "gas")
                sol_surf = ct.Interface(surf_mech, "Pt_surf", [sol_gas])
                flow = Flow1D(phase=sol_gas)
                rsurf = ReactingSurf1D(phase=sol_surf, name="reacting-surface")
                container = OneDim([flow, rsurf])
                flow.setupGrid([0.0, 0.01])
                container.resize()
                rsurf.setTemperature(900.0)
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                rs_loc = rsurf.loc()
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                nsp_gas = sol_gas.n_species

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_T] = 880.0
                    for k in range(nsp_gas):
                        xg[pt + c_offset_Y + k] = 1.0 / nsp_gas

                cov = [0.32, 0.10, 0.05, 0.05, 0.08, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10]
                for k in range(len(cov)):
                    xg[rs_loc + k] = cov[k]

                prev_cov = [0.30, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10]
                prev_x = np.zeros(sz, dtype=float)
                for k in range(len(prev_cov)):
                    prev_x[rs_loc + k] = prev_cov[k]
                rsurf.initTimeInteg(1.0 / 1000.0, prev_x)

                rg = np.zeros(sz, dtype=float)
                diag = np.zeros(sz, dtype=int)
                rsurf.eval(-1, xg, rg, diag, 1000.0)

                data["r_cov"] = [float(rg[rs_loc + k]) for k in range(len(cov))]
                data["diag_cov"] = [int(diag[rs_loc + k]) for k in range(len(cov))]

            elif case_id == "boundary1d_19_reactingsurf1d_eval_fixed_coverage":
                sol_gas = ct.Solution(surf_mech, "gas")
                sol_surf = ct.Interface(surf_mech, "Pt_surf", [sol_gas])
                flow = Flow1D(phase=sol_gas)
                rsurf = ReactingSurf1D(phase=sol_surf, name="reacting-surface")
                container = OneDim([flow, rsurf])
                flow.setupGrid([0.0, 0.01])
                container.resize()
                rsurf.setTemperature(850.0)
                rsurf.enableCoverageEquations(False)
                container.init()
                container.resize()

                sz = container.size()
                xg = np.zeros(sz, dtype=float)
                fl_loc = flow.loc()
                rs_loc = rsurf.loc()
                last_j = flow.nPoints() - 1
                last_pt = fl_loc + flow.index(0, last_j)
                nsp_gas = sol_gas.n_species

                for j in range(flow.nPoints()):
                    pt = fl_loc + flow.index(0, j)
                    xg[pt + c_offset_T] = 850.0
                    for k in range(nsp_gas):
                        xg[pt + c_offset_Y + k] = 1.0 / nsp_gas

                cov = [0.35, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10]
                for k in range(len(cov)):
                    xg[rs_loc + k] = cov[k]

                rg = np.zeros(sz, dtype=float)
                diag = np.ones(sz, dtype=int)
                rsurf.eval(-1, xg, rg, diag, 0.0)

                data["r_cov"] = [float(rg[rs_loc + k]) for k in range(len(cov))]
                data["diag_cov"] = [int(diag[rs_loc + k]) for k in range(len(cov))]

            elif case_id == "boundary1d_20_topology_error_handling":
                err1, err2, err3, err4, err5 = "", "", "", "", ""
                try:
                    bdry = TestBoundary1D()
                    bdry._init(0)
                except CanteraError as e:
                    err1 = str(e)

                try:
                    sol = ct.Solution(gas_mech)
                    bdry1 = Inlet1D(phase=sol, name="inlet")
                    bdry2 = Outlet1D(phase=sol, name="outlet")
                    container = OneDim([bdry1, bdry2])
                    container.init()
                except CanteraError as e:
                    err2 = str(e)

                try:
                    sol = ct.Solution(gas_mech)
                    flow = Flow1D(phase=sol)
                    inlet = Inlet1D(phase=sol, name="inlet")
                    container = OneDim([flow, inlet])
                    flow.setupGrid([0.0, 0.01])
                    container.resize()
                    container.init()
                except CanteraError as e:
                    err3 = str(e)

                try:
                    sol = ct.Solution(gas_mech)
                    outlet = Outlet1D(phase=sol, name="outlet")
                    flow = Flow1D(phase=sol)
                    container = OneDim([outlet, flow])
                    flow.setupGrid([0.0, 0.01])
                    container.resize()
                    container.init()
                except CanteraError as e:
                    err4 = str(e)

                try:
                    sol = ct.Solution(gas_mech)
                    outlet_res = OutletRes1D(phase=sol, name="outlet-reservoir")
                    flow = Flow1D(phase=sol)
                    container = OneDim([outlet_res, flow])
                    flow.setupGrid([0.0, 0.01])
                    container.resize()
                    container.init()
                except CanteraError as e:
                    err5 = str(e)

                data["err_uninstalled"] = err1
                data["err_connected_connector"] = err2
                data["err_right_inlet_unstrained"] = err3
                data["err_left_outlet"] = err4
                data["err_left_outlet_res"] = err5

            results.append({
                "case_id": case_id,
                "success": True,
                "data": data,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, indent=2)

    print(f"Executed {len(results)} Python cases. Results in {output_path}")


if __name__ == "__main__":
    run_port()
