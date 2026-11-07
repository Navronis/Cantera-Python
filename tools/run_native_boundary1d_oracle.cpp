/**
 * Pinned native Cantera Boundary1D differential validation oracle.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_boundary1d.jsonl, writing exact oracle outputs to
 * audit/native_boundary1d_oracle_results.json.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <cmath>

#include "cantera/oneD/Boundary1D.h"
#include "cantera/oneD/Flow1D.h"
#include "cantera/oneD/DomainFactory.h"
#include "cantera/oneD/OneDim.h"
#include "cantera/base/Solution.h"
#include "cantera/base/AnyMap.h"
#include "cantera/base/ctexceptions.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/kinetics/InterfaceKinetics.h"

using namespace Cantera;
using namespace std;

class TestBoundary1D : public Boundary1D
{
public:
    using Boundary1D::_init;
    using Boundary1D::m_temp;
    using Boundary1D::m_mdot;

    TestBoundary1D() {
        m_index = npos;
    }
};

class TestReactingSurf1D : public ReactingSurf1D
{
public:
    using ReactingSurf1D::_init;
    using ReactingSurf1D::m_sphase;
    using ReactingSurf1D::m_fixed_cov;
    using ReactingSurf1D::m_work;

    explicit TestReactingSurf1D(shared_ptr<Solution> phase, const string& id="")
        : ReactingSurf1D(phase, id)
    {
    }

    void testInit() {
        m_nv = m_nsp;
        resize(m_nsp, 1);
        m_fixed_cov.resize(m_nsp, 0.0);
        m_fixed_cov[0] = 1.0;
        m_work.resize(m_kin->nTotalSpecies(), 0.0);
        for (size_t n = 0; n < m_nsp; n++) {
            setBounds(n, -1.0e-5, 2.0);
        }
    }
};

static string escapeJsonString(const string& input) {
    ostringstream ss;
    for (char c : input) {
        if (c == '"') ss << "\\\"";
        else if (c == '\\') ss << "\\\\";
        else if (c == '\n') ss << "\\n";
        else if (c == '\r') ss << "\\r";
        else if (c == '\t') ss << "\\t";
        else ss << c;
    }
    return ss.str();
}

int main(int argc, char** argv) {
    string corpus_path = "audit/corpus_boundary1d.jsonl";
    string output_path = "audit/native_boundary1d_oracle_results.json";

    if (argc > 1) {
        corpus_path = argv[1];
    }
    if (argc > 2) {
        output_path = argv[2];
    }

    cout << "Loading corpus from " << corpus_path << endl;
    ifstream corpus_file(corpus_path);
    if (!corpus_file.is_open()) {
        cerr << "Failed to open corpus file: " << corpus_path << endl;
        return 1;
    }

    string gas_mech = "h2o2.yaml";
    string surf_mech = "ptcombust.yaml";

    ofstream out(output_path);
    out << "{\n  \"cases\": [\n";

    string line;
    bool first_case = true;
    size_t case_count = 0;

    while (getline(corpus_file, line)) {
        if (line.empty()) continue;
        case_count++;

        // Parse case_id from line
        string case_id;
        size_t id_pos = line.find("\"case_id\": \"");
        if (id_pos != string::npos) {
            size_t start = id_pos + 12;
            size_t end = line.find("\"", start);
            case_id = line.substr(start, end - start);
        } else {
            case_id = "case_" + to_string(case_count);
        }

        cout << "Running native case: " << case_id << endl;

        if (!first_case) {
            out << ",\n";
        }
        first_case = false;

        out << "    {\n";
        out << "      \"case_id\": \"" << case_id << "\",\n";
        out << "      \"success\": true,\n";
        out << "      \"data\": {\n";

        if (case_id == "boundary1d_01_base_defaults") {
            TestBoundary1D bdry;
            out << "        \"domain_type\": \"" << bdry.domainType() << "\",\n";
            out << "        \"is_connector\": " << (bdry.isConnector() ? "true" : "false") << ",\n";
            out << "        \"n_points\": " << bdry.nPoints() << ",\n";
            out << "        \"n_components\": " << bdry.nComponents() << ",\n";
            out << "        \"init_temp\": " << bdry.temperature() << ",\n";
            bdry.setTemperature(500.0);
            out << "        \"set_temp\": " << bdry.temperature() << ",\n";
            out << "        \"init_mdot\": " << bdry.mdot() << ",\n";
            bdry.setMdot(1.25);
            out << "        \"set_mdot\": " << bdry.mdot() << ",\n";
            out << "        \"n_species\": " << bdry.nSpecies() << ",\n";

            bool caught_xin = false, caught_mass = false, caught_spread = false, caught_getspread = false;
            try { bdry.setMoleFractions("H2:1"); } catch (const NotImplementedError&) { caught_xin = true; }
            try { bdry.massFraction(0); } catch (const NotImplementedError&) { caught_mass = true; }
            try { bdry.setSpreadRate(10.0); } catch (const NotImplementedError&) { caught_spread = true; }
            try { bdry.spreadRate(); } catch (const NotImplementedError&) { caught_getspread = true; }

            out << "        \"caught_xin\": " << (caught_xin ? "true" : "false") << ",\n";
            out << "        \"caught_mass\": " << (caught_mass ? "true" : "false") << ",\n";
            out << "        \"caught_spread\": " << (caught_spread ? "true" : "false") << ",\n";
            out << "        \"caught_getspread\": " << (caught_getspread ? "true" : "false") << "\n";
        }
        else if (case_id == "boundary1d_02_empty1d_lifecycle") {
            auto sol = newSolution(gas_mech);
            auto flow = newFlow1D("gas-flow", sol);
            auto empty = newDomain<Empty1D>("empty", sol);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(empty);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            container.init();
            container.resize();

            out << "        \"domain_type\": \"" << empty->domainType() << "\",\n";
            out << "        \"is_connector\": " << (empty->isConnector() ? "true" : "false") << ",\n";
            out << "        \"n_points\": " << empty->nPoints() << ",\n";
            out << "        \"n_components\": " << empty->nComponents() << ",\n";

            vector<double> xg(container.size(), 1.0);
            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            empty->eval(npos, xg, rg, diag, 0.0);

            double sum_rg = 0.0;
            for (double v : rg) sum_rg += std::abs(v);
            out << "        \"eval_rg_sum\": " << sum_rg << "\n";
        }
        else if (case_id == "boundary1d_03_inlet1d_properties_gas") {
            auto sol = newSolution(gas_mech);
            Inlet1D inlet(sol, "fuel_inlet");
            out << "        \"domain_type\": \"" << inlet.domainType() << "\",\n";
            out << "        \"is_connector\": " << (inlet.isConnector() ? "true" : "false") << ",\n";
            out << "        \"n_species\": " << inlet.nSpecies() << ",\n";
            inlet.setTemperature(400.0);
            out << "        \"temperature\": " << inlet.temperature() << ",\n";
            inlet.setMdot(2.5);
            out << "        \"mdot\": " << inlet.mdot() << ",\n";
            inlet.setSpreadRate(12.3);
            out << "        \"spread_rate\": " << inlet.spreadRate() << ",\n";
            inlet.setMoleFractions("H2:0.8, O2:0.2");

            out << "        \"mass_fractions\": [";
            for (size_t k = 0; k < inlet.nSpecies(); k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << inlet.massFraction(k);
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_04_inlet1d_properties_span") {
            auto sol = newSolution(gas_mech);
            Inlet1D inlet(sol, "ox_inlet");
            inlet.setTemperature(320.0);
            inlet.setMdot(0.8);
            vector<double> xin = {0.4, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2};
            inlet.setMoleFractions(xin);

            out << "        \"temperature\": " << inlet.temperature() << ",\n";
            out << "        \"mdot\": " << inlet.mdot() << ",\n";
            out << "        \"mass_fractions\": [";
            for (size_t k = 0; k < inlet.nSpecies(); k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << inlet.massFraction(k);
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_05_inlet1d_unstrained_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet = newDomain<Inlet1D>("inlet", sol);
            auto flow = newFlow1D("unstrained-flow", sol);
            auto outlet = newDomain<Outlet1D>("outlet", sol);
            OneDim container;
            container.addDomain(inlet);
            container.addDomain(flow);
            container.addDomain(outlet);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();

            inlet->setTemperature(350.0);
            inlet->setMdot(0.15);
            inlet->setMoleFractions("H2:0.1, O2:0.9");
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_U] = 0.10 + 0.01 * j;
                xg[pt + c_offset_V] = 0.0;
                xg[pt + c_offset_T] = 320.0 + 10.0 * j;
                xg[pt + c_offset_L] = 0.0;
                xg[pt + c_offset_Y] = 0.1;
                xg[pt + c_offset_Y + 3] = 0.9;
            }
            flow->setGas(span<const double>(xg.data() + fl_loc, flow->size()), 0);

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            inlet->eval(npos, xg, rg, diag, 0.0);
            

            out << "        \"r_u\": " << setprecision(16) << rg[fl_loc + c_offset_U] << ",\n";
            out << "        \"r_T\": " << setprecision(16) << rg[fl_loc + c_offset_T] << ",\n";
            out << "        \"r_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[fl_loc + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_06_inlet1d_free_flow_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet = newDomain<Inlet1D>("inlet", sol);
            auto flow = newFlow1D("free-flow", sol);
            auto outlet = newDomain<Outlet1D>("outlet", sol);
            OneDim container;
            container.addDomain(inlet);
            container.addDomain(flow);
            container.addDomain(outlet);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();

            inlet->setTemperature(300.0);
            inlet->setMoleFractions("H2:0.25, O2:0.25, AR:0.5");
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_U] = 0.25;
                xg[pt + c_offset_V] = 0.0;
                xg[pt + c_offset_T] = 310.0;
                xg[pt + c_offset_L] = 0.0;
                xg[pt + c_offset_Y] = 0.05;
                xg[pt + c_offset_Y + 3] = 0.40;
                xg[pt + c_offset_Y + 8] = 0.55;
            }
            flow->setGas(span<const double>(xg.data() + fl_loc, flow->size()), 0);

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            inlet->eval(npos, xg, rg, diag, 0.0);

            out << "        \"computed_mdot\": " << setprecision(16) << inlet->mdot() << ",\n";
            out << "        \"r_T\": " << setprecision(16) << rg[fl_loc + c_offset_T] << ",\n";
            out << "        \"r_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[fl_loc + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_07_inlet1d_strained_flow_left_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet_l = newDomain<Inlet1D>("inlet", sol, "inlet_left");
            auto flow = newFlow1D("axisymmetric-flow", sol);
            auto inlet_r = newDomain<Inlet1D>("inlet", sol, "inlet_right");
            OneDim container;
            container.addDomain(inlet_l);
            container.addDomain(flow);
            container.addDomain(inlet_r);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();

            inlet_l->setTemperature(300.0);
            inlet_l->setMdot(0.5);
            inlet_l->setSpreadRate(20.0);
            inlet_l->setMoleFractions("H2:0.5, AR:0.5");

            inlet_r->setTemperature(450.0);
            inlet_r->setMdot(0.75);
            inlet_r->setSpreadRate(15.0);
            inlet_r->setMoleFractions("O2:0.21, AR:0.79");

            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_U] = 0.1;
                xg[pt + c_offset_V] = 5.0;
                xg[pt + c_offset_T] = 350.0;
                xg[pt + c_offset_L] = 0.0;
                xg[pt + c_offset_Y + 8] = 1.0;
            }
            flow->setGas(span<const double>(xg.data() + fl_loc, flow->size()), 0);

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            inlet_l->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_L\": " << setprecision(16) << rg[fl_loc + c_offset_L] << ",\n";
            out << "        \"r_V\": " << setprecision(16) << rg[fl_loc + c_offset_V] << ",\n";
            out << "        \"r_T\": " << setprecision(16) << rg[fl_loc + c_offset_T] << ",\n";
            out << "        \"r_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[fl_loc + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_08_inlet1d_strained_flow_right_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet_l = newDomain<Inlet1D>("inlet", sol, "inlet_left");
            auto flow = newFlow1D("axisymmetric-flow", sol);
            auto inlet_r = newDomain<Inlet1D>("inlet", sol, "inlet_right");
            OneDim container;
            container.addDomain(inlet_l);
            container.addDomain(flow);
            container.addDomain(inlet_r);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();

            inlet_l->setTemperature(300.0);
            inlet_l->setMdot(0.5);
            inlet_l->setSpreadRate(20.0);
            inlet_l->setMoleFractions("H2:0.5, AR:0.5");

            inlet_r->setTemperature(450.0);
            inlet_r->setMdot(0.75);
            inlet_r->setSpreadRate(15.0);
            inlet_r->setMoleFractions("O2:0.21, AR:0.79");

            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_U] = -0.2;
                xg[pt + c_offset_V] = 4.0;
                xg[pt + c_offset_T] = 400.0;
                xg[pt + c_offset_L] = 0.0;
                xg[pt + c_offset_Y + 8] = 1.0;
            }
            flow->setGas(span<const double>(xg.data() + fl_loc, flow->size()), last_j);

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            inlet_r->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_U\": " << setprecision(16) << rg[last_pt + c_offset_U] << ",\n";
            out << "        \"r_V\": " << setprecision(16) << rg[last_pt + c_offset_V] << ",\n";
            out << "        \"r_T\": " << setprecision(16) << rg[last_pt + c_offset_T] << ",\n";
            out << "        \"r_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[last_pt + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_09_outlet1d_unstrained_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet = newDomain<Inlet1D>("inlet", sol);
            auto flow = newFlow1D("gas-flow", sol);
            auto outlet = newDomain<Outlet1D>("outlet", sol);
            OneDim container;
            container.addDomain(inlet);
            container.addDomain(flow);
            container.addDomain(outlet);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t prev_pt = fl_loc + flow->index(0, last_j - 1);

            xg[prev_pt + c_offset_T] = 1500.0;
            xg[last_pt + c_offset_T] = 1550.0;
            for (size_t k = 0; k < nsp; k++) {
                xg[prev_pt + c_offset_Y + k] = 0.05 * (k + 1);
                xg[last_pt + c_offset_Y + k] = 0.05 * (k + 1) + 0.002 * (k + 1);
            }

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 1);
            outlet->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_T\": " << setprecision(16) << rg[last_pt + c_offset_T] << ",\n";
            out << "        \"r_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[last_pt + c_offset_Y + k];
            }
            out << "],\n";
            out << "        \"diag_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << diag[last_pt + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_10_outletres1d_unstrained_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet = newDomain<Inlet1D>("inlet", sol);
            auto flow = newFlow1D("gas-flow", sol);
            auto outlet_res = newDomain<OutletRes1D>("outlet-reservoir", sol);
            OneDim container;
            container.addDomain(inlet);
            container.addDomain(flow);
            container.addDomain(outlet_res);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();
            outlet_res->setMoleFractions("H2O:0.2, AR:0.8");
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t prev_pt = fl_loc + flow->index(0, last_j - 1);

            xg[prev_pt + c_offset_T] = 1200.0;
            xg[last_pt + c_offset_T] = 1250.0;
            for (size_t k = 0; k < nsp; k++) {
                xg[last_pt + c_offset_Y + k] = 0.08 * (k + 1);
            }

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 1);
            outlet_res->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_T\": " << setprecision(16) << rg[last_pt + c_offset_T] << ",\n";
            out << "        \"r_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[last_pt + c_offset_Y + k];
            }
            out << "],\n";
            out << "        \"diag_Y\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << diag[last_pt + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_11_symm1d_right_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet = newDomain<Inlet1D>("inlet", sol);
            auto flow = newFlow1D("axisymmetric-flow", sol);
            auto symm = newDomain<Symm1D>("symmetry-plane", sol);
            OneDim container;
            container.addDomain(inlet);
            container.addDomain(flow);
            container.addDomain(symm);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t prev_pt = fl_loc + flow->index(0, last_j - 1);

            xg[prev_pt + c_offset_V] = 2.0;
            xg[last_pt + c_offset_V] = 2.5;
            xg[prev_pt + c_offset_T] = 1200.0;
            xg[last_pt + c_offset_T] = 1220.0;

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 1);
            symm->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_V\": " << setprecision(16) << rg[last_pt + c_offset_V] << ",\n";
            out << "        \"diag_V\": " << diag[last_pt + c_offset_V] << ",\n";
            out << "        \"r_T\": " << setprecision(16) << rg[last_pt + c_offset_T] << ",\n";
            out << "        \"diag_T\": " << diag[last_pt + c_offset_T] << "\n";
        }
        else if (case_id == "boundary1d_12_symm1d_left_eval") {
            auto sol = newSolution(gas_mech);
            auto symm = newDomain<Symm1D>("symmetry-plane", sol);
            auto flow = newFlow1D("axisymmetric-flow", sol);
            auto outlet = newDomain<Outlet1D>("outlet", sol);
            OneDim container;
            container.addDomain(symm);
            container.addDomain(flow);
            container.addDomain(outlet);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t nc = flow->nComponents();
            size_t p0 = fl_loc + flow->index(0, 0);
            size_t p1 = fl_loc + flow->index(0, 1);

            xg[p0 + c_offset_V] = 1.0;
            xg[p1 + c_offset_V] = 1.2;
            xg[p0 + c_offset_T] = 300.0;
            xg[p1 + c_offset_T] = 310.0;

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 1);
            symm->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_V\": " << setprecision(16) << rg[p0 + c_offset_V] << ",\n";
            out << "        \"diag_V\": " << diag[p0 + c_offset_V] << ",\n";
            out << "        \"r_T\": " << setprecision(16) << rg[p0 + c_offset_T] << ",\n";
            out << "        \"diag_T\": " << diag[p0 + c_offset_T] << "\n";
        }
        else if (case_id == "boundary1d_13_surf1d_left_eval") {
            auto sol = newSolution(gas_mech);
            auto surf = newDomain<Surf1D>("surface", sol);
            auto flow = newFlow1D("gas-flow", sol);
            auto outlet = newDomain<Outlet1D>("outlet", sol);
            OneDim container;
            container.addDomain(surf);
            container.addDomain(flow);
            container.addDomain(outlet);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();
            surf->setTemperature(800.0);
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t p0 = fl_loc + flow->index(0, 0);
            xg[p0 + c_offset_T] = 750.0;

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            surf->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_T\": " << setprecision(16) << rg[p0 + c_offset_T] << "\n";
        }
        else if (case_id == "boundary1d_14_surf1d_right_eval") {
            auto sol = newSolution(gas_mech);
            auto inlet = newDomain<Inlet1D>("inlet", sol);
            auto flow = newFlow1D("gas-flow", sol);
            auto surf = newDomain<Surf1D>("surface", sol);
            OneDim container;
            container.addDomain(inlet);
            container.addDomain(flow);
            container.addDomain(surf);
            vector<double> z = {0.0, 0.005, 0.01, 0.015, 0.02};
            flow->setupGrid(z);
            container.resize();
            surf->setTemperature(650.0);
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            xg[last_pt + c_offset_T] = 680.0;

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            surf->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_T\": " << setprecision(16) << rg[last_pt + c_offset_T] << "\n";
        }
        else if (case_id == "boundary1d_15_reactingsurf1d_properties") {
            auto sol_surf = newSolution(surf_mech, "Pt_surf");
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);

            out << "        \"domain_type\": \"" << rsurf->domainType() << "\",\n";
            out << "        \"is_connector\": " << (rsurf->isConnector() ? "true" : "false") << ",\n";
            out << "        \"cov_enabled_init\": " << (rsurf->coverageEnabled() ? "true" : "false") << ",\n";
            rsurf->enableCoverageEquations(false);
            out << "        \"cov_enabled_false\": " << (rsurf->coverageEnabled() ? "true" : "false") << ",\n";
            rsurf->enableCoverageEquations(true);
            out << "        \"cov_enabled_true\": " << (rsurf->coverageEnabled() ? "true" : "false") << ",\n";

            auto surf_phase = dynamic_pointer_cast<SurfPhase>(sol_surf->thermo());
            size_t nsp = surf_phase->nSpecies();
            out << "        \"species_names\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << "\"" << rsurf->componentName(k) << "\"";
            }
            out << "],\n";
            out << "        \"species_indices\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << rsurf->componentIndex(rsurf->componentName(k));
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_16_reactingsurf1d_bounds_resetbadvalues") {
            auto sol_surf = newSolution(surf_mech, "Pt_surf");
            TestReactingSurf1D rsurf(sol_surf, "test_rsurf");
            rsurf.testInit();

            auto surf_phase = dynamic_pointer_cast<SurfPhase>(sol_surf->thermo());
            size_t nsp = surf_phase->nSpecies();

            out << "        \"lower_bounds\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rsurf.lowerBound(k);
            }
            out << "],\n";
            out << "        \"upper_bounds\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rsurf.upperBound(k);
            }
            out << "],\n";

            vector<double> cov(nsp, 0.5);
            rsurf.resetBadValues(cov);

            out << "        \"normalized_cov\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << cov[k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_17_reactingsurf1d_eval_steady") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "Pt_surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(900.0);
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t nsp_gas = sol_gas->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 880.0;
                for (size_t k = 0; k < nsp_gas; k++) {
                    xg[pt + c_offset_Y + k] = 1.0 / nsp_gas;
                }
            }

            vector<double> cov = {0.30, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};
            for (size_t k = 0; k < cov.size(); k++) {
                xg[rs_loc + k] = cov[k];
            }

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            rsurf->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_cov\": [";
            for (size_t k = 0; k < cov.size(); k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[rs_loc + k];
            }
            out << "],\n";
            out << "        \"diag_cov\": [";
            for (size_t k = 0; k < cov.size(); k++) {
                if (k > 0) out << ", ";
                out << diag[rs_loc + k];
            }
            out << "],\n";
            out << "        \"r_gas_T\": " << setprecision(16) << rg[last_pt + c_offset_T] << ",\n";
            out << "        \"r_gas_Y\": [";
            for (size_t k = 0; k < sol_gas->thermo()->nSpecies(); k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[last_pt + c_offset_Y + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_18_reactingsurf1d_eval_transient") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "Pt_surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(900.0);
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t nsp_gas = sol_gas->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 880.0;
                for (size_t k = 0; k < nsp_gas; k++) {
                    xg[pt + c_offset_Y + k] = 1.0 / nsp_gas;
                }
            }

            vector<double> cov = {0.32, 0.10, 0.05, 0.05, 0.08, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};
            for (size_t k = 0; k < cov.size(); k++) {
                xg[rs_loc + k] = cov[k];
            }

            // Set previous solution
            vector<double> prev_cov = {0.30, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};
            vector<double> prev_x(container.size(), 0.0);
            for (size_t k = 0; k < prev_cov.size(); k++) {
                prev_x[rs_loc + k] = prev_cov[k];
            }
            rsurf->initTimeInteg(1.0/1000.0, prev_x);

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 0);
            rsurf->eval(npos, xg, rg, diag, 1000.0);

            out << "        \"r_cov\": [";
            for (size_t k = 0; k < cov.size(); k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[rs_loc + k];
            }
            out << "],\n";
            out << "        \"diag_cov\": [";
            for (size_t k = 0; k < cov.size(); k++) {
                if (k > 0) out << ", ";
                out << diag[rs_loc + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_19_reactingsurf1d_eval_fixed_coverage") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "Pt_surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(850.0);
            rsurf->enableCoverageEquations(false);
            container.init();
            container.resize();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t nsp_gas = sol_gas->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 850.0;
                for (size_t k = 0; k < nsp_gas; k++) {
                    xg[pt + c_offset_Y + k] = 1.0 / nsp_gas;
                }
            }
            vector<double> cov = {0.35, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};
            for (size_t k = 0; k < cov.size(); k++) {
                xg[rs_loc + k] = cov[k];
            }

            vector<double> rg(container.size(), 0.0);
            vector<int> diag(container.size(), 1);
            rsurf->eval(npos, xg, rg, diag, 0.0);

            out << "        \"r_cov\": [";
            for (size_t k = 0; k < cov.size(); k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rg[rs_loc + k];
            }
            out << "],\n";
            out << "        \"diag_cov\": [";
            for (size_t k = 0; k < cov.size(); k++) {
                if (k > 0) out << ", ";
                out << diag[rs_loc + k];
            }
            out << "]\n";
        }
        else if (case_id == "boundary1d_20_topology_error_handling") {
            string err1 = "", err2 = "", err3 = "", err4 = "", err5 = "";
            try {
                TestBoundary1D bdry;
                bdry._init(0);
            } catch (const CanteraError& e) {
                err1 = e.getMessage();
            }

            try {
                auto sol = newSolution(gas_mech);
                auto bdry1 = newDomain<Inlet1D>("inlet", sol);
                auto bdry2 = newDomain<Outlet1D>("outlet", sol);
                OneDim container;
                container.addDomain(bdry1);
                container.addDomain(bdry2);
                container.init();
            container.resize();
            } catch (const CanteraError& e) {
                err2 = e.getMessage();
            }

            try {
                auto sol = newSolution(gas_mech);
                auto flow = newFlow1D("gas-flow", sol);
                auto inlet = newDomain<Inlet1D>("inlet", sol);
                OneDim container;
                container.addDomain(flow);
                container.addDomain(inlet);
                vector<double> z = {0.0, 0.01};
                flow->setupGrid(z);
            container.resize();
                container.init();
            container.resize();
            } catch (const CanteraError& e) {
                err3 = e.getMessage();
            }

            try {
                auto sol = newSolution(gas_mech);
                auto outlet = newDomain<Outlet1D>("outlet", sol);
                auto flow = newFlow1D("gas-flow", sol);
                OneDim container;
                container.addDomain(outlet);
                container.addDomain(flow);
                vector<double> z = {0.0, 0.01};
                flow->setupGrid(z);
            container.resize();
                container.init();
            container.resize();
            } catch (const CanteraError& e) {
                err4 = e.getMessage();
            }

            try {
                auto sol = newSolution(gas_mech);
                auto outlet_res = newDomain<OutletRes1D>("outlet-reservoir", sol);
                auto flow = newFlow1D("gas-flow", sol);
                OneDim container;
                container.addDomain(outlet_res);
                container.addDomain(flow);
                vector<double> z = {0.0, 0.01};
                flow->setupGrid(z);
            container.resize();
                container.init();
            container.resize();
            } catch (const CanteraError& e) {
                err5 = e.getMessage();
            }

            out << "        \"err_uninstalled\": \"" << escapeJsonString(err1) << "\",\n";
            out << "        \"err_connected_connector\": \"" << escapeJsonString(err2) << "\",\n";
            out << "        \"err_right_inlet_unstrained\": \"" << escapeJsonString(err3) << "\",\n";
            out << "        \"err_left_outlet\": \"" << escapeJsonString(err4) << "\",\n";
            out << "        \"err_left_outlet_res\": \"" << escapeJsonString(err5) << "\"\n";
        }

        out << "      }\n";
        out << "    }";
    }

    out << "\n  ]\n}\n";
    out.close();

    cout << "Finished running all " << case_count << " native cases. Results in " << output_path << endl;
    return 0;
}
