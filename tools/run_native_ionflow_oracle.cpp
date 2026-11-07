/**
 * Pinned native Cantera IonFlow differential validation oracle.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_ionflow.jsonl, writing exact oracle outputs to
 * audit/native_ionflow_oracle_results.json.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <cmath>
#include <map>

#include "cantera/oneD/IonFlow.h"
#include "cantera/oneD/Boundary1D.h"
#include "cantera/oneD/OneDim.h"
#include "cantera/base/Solution.h"
#include "cantera/base/ctexceptions.h"

using namespace Cantera;
using namespace std;

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
    string corpus_path = "audit/corpus_ionflow.jsonl";
    string output_path = "audit/native_ionflow_oracle_results.json";

    if (argc > 1) corpus_path = argv[1];
    if (argc > 2) output_path = argv[2];

    cout << "Loading IonFlow corpus from " << corpus_path << endl;
    ifstream infile(corpus_path);
    if (!infile.is_open()) {
        cerr << "Failed to open corpus: " << corpus_path << endl;
        return 1;
    }

    ofstream outfile(output_path);
    if (!outfile.is_open()) {
        cerr << "Failed to open output: " << output_path << endl;
        return 1;
    }

    outfile << "{\n  \"cases\": [\n";

    string line;
    bool first_case = true;
    size_t case_count = 0;

    while (getline(infile, line)) {
        if (line.empty()) continue;

        try {
            AnyMap case_map = AnyMap::fromYamlString(line);
            string case_id = case_map["case_id"].asString();
            string mech = case_map["mechanism"].asString();
            string domain_type = case_map["domain_type"].asString();
            vector<double> grid = case_map["grid"].asVector<double>();
            double pressure = case_map["pressure"].asDouble();
            bool energy_enabled = case_map["energy_enabled"].asBool();
            bool efield_enabled = case_map["electric_field_enabled"].asBool();
            bool has_electron_transport = case_map.hasKey("electron_transport") && case_map["electron_transport"].asBool();
            double rdt = case_map["rdt"].asDouble();

            size_t n_pts = grid.size();

            string trans_model = (mech == "ch4_ion.yaml") ? "ionized-gas" : "mixture-averaged";
            auto sol = newSolution(mech, "", trans_model);
            auto inlet = make_shared<Inlet1D>(sol, "inlet");
            auto flow = make_shared<IonFlow>(sol, "ion_flame", n_pts);
            auto outlet = make_shared<Outlet1D>(sol, "outlet");

            flow->setupGrid(grid);
            flow->setPressure(pressure);

            if (domain_type == "free-flow") {
                flow->setFreeFlow();
            } else if (domain_type == "axisymmetric-flow") {
                flow->setAxisymmetricFlow();
            } else if (domain_type == "unstrained-flow") {
                flow->setUnstrainedFlow();
            }

            if (energy_enabled) {
                flow->solveEnergyEqn();
            } else {
                flow->fixTemperature();
            }

            if (efield_enabled) {
                flow->solveElectricField();
            } else {
                flow->fixElectricField();
            }

            if (has_electron_transport) {
                vector<double> tfix = case_map["tfix"].asVector<double>();
                vector<double> diff_e = case_map["diff_e"].asVector<double>();
                vector<double> mobi_e = case_map["mobi_e"].asVector<double>();
                flow->setElectronTransport(tfix, diff_e, mobi_e);
            }

            vector<shared_ptr<Domain1D>> domains = {inlet, flow, outlet};
            OneDim sim(domains);
            sim.init();

            vector<double> x_global(sim.size(), 0.0);
            auto pts_data = case_map["state_profile"].asVector<AnyMap>();

            size_t flow_loc = flow->loc();
            size_t n_comp = flow->nComponents();
            size_t nsp = sol->thermo()->nSpecies();

            for (size_t j = 0; j < n_pts; j++) {
                auto& pt = pts_data[j];
                double u = pt["u"].asDouble();
                double V = pt["V"].asDouble();
                double T = pt["T"].asDouble();
                double L = pt["Lambda"].asDouble();
                double E_val = pt["eField"].asDouble();
                double Uo = pt["Uo"].asDouble();
                auto Y_map = pt["Y"].as<AnyMap>();

                x_global[flow_loc + flow->index(c_offset_U, j)] = u;
                x_global[flow_loc + flow->index(c_offset_V, j)] = V;
                x_global[flow_loc + flow->index(c_offset_T, j)] = T;
                x_global[flow_loc + flow->index(c_offset_L, j)] = L;
                x_global[flow_loc + flow->index(c_offset_E, j)] = E_val;
                x_global[flow_loc + flow->index(c_offset_Uo, j)] = Uo;

                for (size_t k = 0; k < nsp; k++) {
                    string sp_name = sol->thermo()->speciesName(k);
                    double y_val = Y_map.hasKey(sp_name) ? Y_map[sp_name].asDouble() : 0.0;
                    x_global[flow_loc + flow->index(c_offset_Y + k, j)] = y_val;
                }
            }

            inlet->setTemperature(pts_data[0]["T"].asDouble());
            inlet->setMdot(0.5);

            vector<double> rsd_global(sim.size(), 0.0);
            vector<int> diag_global(sim.size(), 0);

            flow->eval(npos, x_global, rsd_global, diag_global, rdt);

            if (!first_case) {
                outfile << ",\n";
            }
            first_case = false;

            outfile << "    {\n";
            outfile << "      \"case_id\": \"" << escapeJsonString(case_id) << "\",\n";
            outfile << "      \"flow_size\": " << flow->size() << ",\n";
            outfile << "      \"flow_loc\": " << flow_loc << ",\n";
            outfile << "      \"residuals\": [\n";

            for (size_t j = 0; j < n_pts; j++) {
                outfile << "        {\n";
                outfile << "          \"point\": " << j << ",\n";
                outfile << "          \"z\": " << grid[j] << ",\n";
                outfile << "          \"u\": " << setprecision(16) << rsd_global[flow_loc + flow->index(c_offset_U, j)] << ",\n";
                outfile << "          \"V\": " << setprecision(16) << rsd_global[flow_loc + flow->index(c_offset_V, j)] << ",\n";
                outfile << "          \"T\": " << setprecision(16) << rsd_global[flow_loc + flow->index(c_offset_T, j)] << ",\n";
                outfile << "          \"Lambda\": " << setprecision(16) << rsd_global[flow_loc + flow->index(c_offset_L, j)] << ",\n";
                outfile << "          \"eField\": " << setprecision(16) << rsd_global[flow_loc + flow->index(c_offset_E, j)] << ",\n";
                outfile << "          \"Uo\": " << setprecision(16) << rsd_global[flow_loc + flow->index(c_offset_Uo, j)] << ",\n";
                outfile << "          \"diag_u\": " << diag_global[flow_loc + flow->index(c_offset_U, j)] << ",\n";
                outfile << "          \"diag_T\": " << diag_global[flow_loc + flow->index(c_offset_T, j)] << ",\n";
                outfile << "          \"diag_E\": " << diag_global[flow_loc + flow->index(c_offset_E, j)] << ",\n";
                outfile << "          \"Y\": {\n";
                for (size_t k = 0; k < nsp; k++) {
                    string sp = sol->thermo()->speciesName(k);
                    double r_yk = rsd_global[flow_loc + flow->index(c_offset_Y + k, j)];
                    outfile << "            \"" << sp << "\": " << setprecision(16) << r_yk;
                    if (k + 1 < nsp) outfile << ",";
                    outfile << "\n";
                }
                outfile << "          }\n";
                outfile << "        }" << (j + 1 < n_pts ? "," : "") << "\n";
            }

            outfile << "      ]\n";
            outfile << "    }";
            case_count++;
        } catch (const exception& e) {
            cerr << "Error running case: " << e.what() << endl;
            return 1;
        }
    }

    outfile << "\n  ],\n";
    outfile << "  \"total_cases\": " << case_count << "\n";
    outfile << "}\n";

    cout << "Executed " << case_count << " cases successfully. Oracle results written to " << output_path << endl;
    return 0;
}
