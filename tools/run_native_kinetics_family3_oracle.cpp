/**
 * Pinned native Cantera Kinetics Family 3 differential validation oracle.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_kinetics_family3.jsonl, writing exact native oracle outputs to
 * audit/native_kinetics_family3_oracle_results.json.
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
#include <span>

#include "cantera/base/Interface.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/kinetics/InterfaceKinetics.h"
#include "cantera/base/ctexceptions.h"

using namespace Cantera;
using namespace std;

static string escapeJson(const string& s) {
    ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else o << c;
    }
    return o.str();
}

static void writeDoubleVector(ostream& os, const vector<double>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ", ";
        if (std::isnan(vec[i])) {
            os << "\"NaN\"";
        } else if (std::isinf(vec[i])) {
            os << (vec[i] > 0 ? "\"Inf\"" : "\"-Inf\"");
        } else {
            os << setprecision(17) << vec[i];
        }
    }
    os << "]";
}

struct CorpusCase {
    string case_id;
    string mechanism;
    string phase;
    double temperature = 300.0;
    double pressure = 101325.0;
    string gas_phase = "gas";
    string gas_mole_fractions;
    string coverages;
    double electric_potential = 0.0;
    map<size_t, double> multipliers;
    string description;
};

static string extractJsonField(const string& line, const string& key) {
    string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == string::npos) return "";
    pos += search.length();
    while (pos < line.length() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.length()) return "";

    if (line[pos] == '"') {
        size_t end = line.find('"', pos + 1);
        if (end == string::npos) return "";
        return line.substr(pos + 1, end - pos - 1);
    } else {
        size_t end = line.find_first_of(",}", pos);
        if (end == string::npos) return "";
        string val = line.substr(pos, end - pos);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
        return val;
    }
}

static CorpusCase parseLine(const string& line) {
    CorpusCase c;
    c.case_id = extractJsonField(line, "case_id");
    c.mechanism = extractJsonField(line, "mechanism");
    c.phase = extractJsonField(line, "phase");
    string t_str = extractJsonField(line, "temperature");
    if (!t_str.empty()) c.temperature = stod(t_str);
    string p_str = extractJsonField(line, "pressure");
    if (!p_str.empty()) c.pressure = stod(p_str);
    c.gas_phase = extractJsonField(line, "gas_phase");
    if (c.gas_phase.empty()) c.gas_phase = "gas";
    c.gas_mole_fractions = extractJsonField(line, "gas_mole_fractions");
    c.coverages = extractJsonField(line, "coverages");
    string phi_str = extractJsonField(line, "electric_potential");
    if (!phi_str.empty()) c.electric_potential = stod(phi_str);
    c.description = extractJsonField(line, "description");

    size_t mpos = line.find("\"multipliers\":");
    if (mpos != string::npos) {
        size_t bstart = line.find('{', mpos);
        size_t bend = line.find('}', bstart);
        if (bstart != string::npos && bend != string::npos) {
            string mcontent = line.substr(bstart + 1, bend - bstart - 1);
            stringstream mss(mcontent);
            string pair;
            while (getline(mss, pair, ',')) {
                size_t colon = pair.find(':');
                if (colon != string::npos) {
                    string k = pair.substr(0, colon);
                    string v = pair.substr(colon + 1);
                    auto trim = [](string& str) {
                        while (!str.empty() && (str.front() == ' ' || str.front() == '"')) str.erase(0, 1);
                        while (!str.empty() && (str.back() == ' ' || str.back() == '"')) str.pop_back();
                    };
                    trim(k);
                    trim(v);
                    if (!k.empty() && !v.empty()) {
                        try {
                            c.multipliers[stoul(k)] = stod(v);
                        } catch (...) {}
                    }
                }
            }
        }
    }
    return c;
}

int main(int argc, char** argv) {
    string corpus_path = "audit/corpus_kinetics_family3.jsonl";
    string output_path = "audit/native_kinetics_family3_oracle_results.json";

    if (argc > 1) corpus_path = argv[1];
    if (argc > 2) output_path = argv[2];

    cout << "Executing Pinned Native Kinetics Family 3 Oracle against " << corpus_path << endl;

    ifstream infile(corpus_path);
    if (!infile.is_open()) {
        cerr << "Failed to open corpus file: " << corpus_path << endl;
        return 1;
    }

    ofstream outfile(output_path);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << output_path << endl;
        return 1;
    }

    outfile << "{\n  \"cases\": [\n";

    map<pair<string, string>, shared_ptr<Interface>> ifaces;

    string line;
    size_t case_num = 0;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        CorpusCase c = parseLine(line);
        if (c.case_id.empty()) continue;

        if (case_num > 0) {
            outfile << ",\n";
        }

        cout << "  [" << (case_num + 1) << "] " << c.case_id << " (" << c.mechanism
             << " phase=" << c.phase
             << " T=" << c.temperature << " P=" << c.pressure << ")..." << flush;

        auto iface_key = make_pair(c.mechanism, c.phase);
        if (ifaces.find(iface_key) == ifaces.end()) {
            ifaces[iface_key] = newInterface(c.mechanism, c.phase);
        }
        auto iface = ifaces[iface_key];
        auto thermo = iface->thermo();
        auto kin = iface->kinetics();

        // Reset multipliers to 1.0
        for (size_t i = 0; i < kin->nReactions(); ++i) {
            kin->setMultiplier(i, 1.0);
        }
        // Apply custom multipliers if any
        for (const auto& [idx, val] : c.multipliers) {
            if (idx < kin->nReactions()) {
                kin->setMultiplier(idx, val);
            }
        }

        // Set gas phase state
        if (!c.gas_phase.empty()) {
            try {
                auto gas = iface->adjacent(c.gas_phase);
                if (gas && !c.gas_mole_fractions.empty()) {
                    gas->thermo()->setState_TPX(c.temperature, c.pressure, c.gas_mole_fractions);
                }
            } catch (...) {}
        }

        // Set surface phase state
        thermo->setState_TP(c.temperature, c.pressure);
        if (!c.coverages.empty()) {
            thermo->setCoveragesByName(c.coverages);
        }
        thermo->setElectricPotential(c.electric_potential);

        size_t nRxn = kin->nReactions();
        size_t nSp = kin->nTotalSpecies();
        size_t nSurfSp = thermo->nSpecies();

        vector<double> kf(nRxn), kr(nRxn), kc(nRxn);
        vector<double> ropf(nRxn), ropr(nRxn), ropnet(nRxn);
        vector<double> cdot(nSp), ddot(nSp), wdot(nSp);
        vector<double> deltaG(nRxn), deltaH(nRxn), deltaS(nRxn);
        vector<double> deltaSSG(nRxn), deltaSSH(nRxn), deltaSSS(nRxn);
        vector<double> surf_conc(nSurfSp), surf_cov(nSurfSp);

        kin->getFwdRateConstants(kf);
        kin->getRevRateConstants(kr);
        kin->getEquilibriumConstants(kc);
        kin->getFwdRatesOfProgress(ropf);
        kin->getRevRatesOfProgress(ropr);
        kin->getNetRatesOfProgress(ropnet);
        kin->getCreationRates(cdot);
        kin->getDestructionRates(ddot);
        kin->getNetProductionRates(wdot);

        kin->getDeltaGibbs(deltaG);
        kin->getDeltaEnthalpy(deltaH);
        kin->getDeltaEntropy(deltaS);
        kin->getDeltaSSGibbs(deltaSSG);
        kin->getDeltaSSEnthalpy(deltaSSH);
        kin->getDeltaSSEntropy(deltaSSS);

        thermo->getConcentrations(surf_conc);
        thermo->getCoverages(surf_cov);

        double cur0 = kin->interfaceCurrent(0);

        outfile << "    {\n";
        outfile << "      \"case_id\": \"" << escapeJson(c.case_id) << "\",\n";
        outfile << "      \"mechanism\": \"" << escapeJson(c.mechanism) << "\",\n";
        outfile << "      \"phase\": \"" << escapeJson(c.phase) << "\",\n";
        outfile << "      \"temperature\": " << setprecision(17) << c.temperature << ",\n";
        outfile << "      \"pressure\": " << setprecision(17) << c.pressure << ",\n";
        outfile << "      \"gas_mole_fractions\": \"" << escapeJson(c.gas_mole_fractions) << "\",\n";
        outfile << "      \"coverages\": \"" << escapeJson(c.coverages) << "\",\n";
        outfile << "      \"electric_potential\": " << setprecision(17) << c.electric_potential << ",\n";
        outfile << "      \"n_reactions\": " << nRxn << ",\n";
        outfile << "      \"n_total_species\": " << nSp << ",\n";
        outfile << "      \"n_surface_species\": " << nSurfSp << ",\n";

        outfile << "      \"forward_rate_constants\": "; writeDoubleVector(outfile, kf); outfile << ",\n";
        outfile << "      \"reverse_rate_constants\": "; writeDoubleVector(outfile, kr); outfile << ",\n";
        outfile << "      \"equilibrium_constants\": "; writeDoubleVector(outfile, kc); outfile << ",\n";
        outfile << "      \"forward_rates_of_progress\": "; writeDoubleVector(outfile, ropf); outfile << ",\n";
        outfile << "      \"reverse_rates_of_progress\": "; writeDoubleVector(outfile, ropr); outfile << ",\n";
        outfile << "      \"net_rates_of_progress\": "; writeDoubleVector(outfile, ropnet); outfile << ",\n";
        outfile << "      \"species_creation_rates\": "; writeDoubleVector(outfile, cdot); outfile << ",\n";
        outfile << "      \"species_destruction_rates\": "; writeDoubleVector(outfile, ddot); outfile << ",\n";
        outfile << "      \"net_species_production_rates\": "; writeDoubleVector(outfile, wdot); outfile << ",\n";

        outfile << "      \"delta_gibbs\": "; writeDoubleVector(outfile, deltaG); outfile << ",\n";
        outfile << "      \"delta_enthalpy\": "; writeDoubleVector(outfile, deltaH); outfile << ",\n";
        outfile << "      \"delta_entropy\": "; writeDoubleVector(outfile, deltaS); outfile << ",\n";
        outfile << "      \"delta_standard_gibbs\": "; writeDoubleVector(outfile, deltaSSG); outfile << ",\n";
        outfile << "      \"delta_standard_enthalpy\": "; writeDoubleVector(outfile, deltaSSH); outfile << ",\n";
        outfile << "      \"delta_standard_entropy\": "; writeDoubleVector(outfile, deltaSSS); outfile << ",\n";

        outfile << "      \"surface_concentrations\": "; writeDoubleVector(outfile, surf_conc); outfile << ",\n";
        outfile << "      \"surface_coverages\": "; writeDoubleVector(outfile, surf_cov); outfile << ",\n";
        outfile << "      \"interface_current\": " << setprecision(17) << cur0 << "\n";

        outfile << "    }";
        cout << " done." << endl;
        case_num++;
    }

    outfile << "\n  ],\n";
    outfile << "  \"total_cases\": " << case_num << "\n}\n";

    cout << "Finished executing native kinetics Family 3 oracle for " << case_num << " cases." << endl;
    return 0;
}
