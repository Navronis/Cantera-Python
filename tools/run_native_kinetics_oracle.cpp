/**
 * Pinned native Cantera Kinetics differential validation oracle.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_kinetics.jsonl, writing exact native oracle outputs to
 * audit/native_kinetics_oracle_results.json.
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

#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/kinetics/BulkKinetics.h"
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

// Simple JSON parser for single-line JSONL corpus
struct CorpusCase {
    string case_id;
    string mechanism;
    double temperature = 300.0;
    double pressure = 101325.0;
    string mole_fractions;
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
    string t_str = extractJsonField(line, "temperature");
    if (!t_str.empty()) c.temperature = stod(t_str);
    string p_str = extractJsonField(line, "pressure");
    if (!p_str.empty()) c.pressure = stod(p_str);
    c.mole_fractions = extractJsonField(line, "mole_fractions");
    c.description = extractJsonField(line, "description");

    // parse multipliers if present
    size_t mpos = line.find("\"multipliers\":");
    if (mpos != string::npos) {
        size_t brace1 = line.find('{', mpos);
        size_t brace2 = line.find('}', brace1);
        if (brace1 != string::npos && brace2 != string::npos && brace2 > brace1 + 1) {
            string inner = line.substr(brace1 + 1, brace2 - brace1 - 1);
            stringstream ss(inner);
            string pair_str;
            while (getline(ss, pair_str, ',')) {
                size_t col = pair_str.find(':');
                if (col != string::npos) {
                    string k = pair_str.substr(0, col);
                    string v = pair_str.substr(col + 1);
                    // trim quotes and whitespace
                    while (!k.empty() && (k.front() == ' ' || k.front() == '"')) k.erase(k.begin());
                    while (!k.empty() && (k.back() == ' ' || k.back() == '"')) k.pop_back();
                    while (!v.empty() && (v.front() == ' ' || v.front() == '"')) v.erase(v.begin());
                    while (!v.empty() && (v.back() == ' ' || v.back() == '"')) v.pop_back();
                    if (!k.empty() && !v.empty()) {
                        c.multipliers[stoul(k)] = stod(v);
                    }
                }
            }
        }
    }
    return c;
}

int main(int argc, char** argv) {
    string corpus_path = "audit/corpus_kinetics.jsonl";
    string output_path = "audit/native_kinetics_oracle_results.json";

    if (argc > 1) corpus_path = argv[1];
    if (argc > 2) output_path = argv[2];

    cout << "Executing Pinned Native Kinetics Oracle against " << corpus_path << endl;

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

    map<string, shared_ptr<Solution>> solutions;

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
             << " T=" << c.temperature << " P=" << c.pressure << ")..." << flush;

        if (solutions.find(c.mechanism) == solutions.end()) {
            solutions[c.mechanism] = newSolution(c.mechanism);
        }
        auto sol = solutions[c.mechanism];
        auto thermo = sol->thermo();
        auto kin = sol->kinetics();

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

        thermo->setState_TPX(c.temperature, c.pressure, c.mole_fractions);

        size_t nRxn = kin->nReactions();
        size_t nSp = kin->nTotalSpecies();

        vector<double> kf(nRxn), kr(nRxn), kc(nRxn);
        vector<double> ropf(nRxn), ropr(nRxn), ropnet(nRxn);
        vector<double> cdot(nSp), ddot(nSp), wdot(nSp);
        vector<double> deltaG(nRxn), deltaH(nRxn), deltaS(nRxn);
        vector<double> deltaSSG(nRxn), deltaSSH(nRxn), deltaSSS(nRxn);
        vector<double> concm(nRxn, NAN);

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

        try {
            kin->getThirdBodyConcentrations(concm);
        } catch (...) {
            // Not all mechanisms/reactions have third bodies
        }

        outfile << "    {\n";
        outfile << "      \"case_id\": \"" << escapeJson(c.case_id) << "\",\n";
        outfile << "      \"mechanism\": \"" << escapeJson(c.mechanism) << "\",\n";
        outfile << "      \"temperature\": " << setprecision(17) << c.temperature << ",\n";
        outfile << "      \"pressure\": " << setprecision(17) << c.pressure << ",\n";
        outfile << "      \"n_reactions\": " << nRxn << ",\n";
        outfile << "      \"n_species\": " << nSp << ",\n";

        outfile << "      \"fwd_rate_constants\": ";
        writeDoubleVector(outfile, kf);
        outfile << ",\n";

        outfile << "      \"rev_rate_constants\": ";
        writeDoubleVector(outfile, kr);
        outfile << ",\n";

        outfile << "      \"equilibrium_constants\": ";
        writeDoubleVector(outfile, kc);
        outfile << ",\n";

        outfile << "      \"fwd_rates_of_progress\": ";
        writeDoubleVector(outfile, ropf);
        outfile << ",\n";

        outfile << "      \"rev_rates_of_progress\": ";
        writeDoubleVector(outfile, ropr);
        outfile << ",\n";

        outfile << "      \"net_rates_of_progress\": ";
        writeDoubleVector(outfile, ropnet);
        outfile << ",\n";

        outfile << "      \"creation_rates\": ";
        writeDoubleVector(outfile, cdot);
        outfile << ",\n";

        outfile << "      \"destruction_rates\": ";
        writeDoubleVector(outfile, ddot);
        outfile << ",\n";

        outfile << "      \"net_production_rates\": ";
        writeDoubleVector(outfile, wdot);
        outfile << ",\n";

        outfile << "      \"delta_gibbs\": ";
        writeDoubleVector(outfile, deltaG);
        outfile << ",\n";

        outfile << "      \"delta_enthalpy\": ";
        writeDoubleVector(outfile, deltaH);
        outfile << ",\n";

        outfile << "      \"delta_entropy\": ";
        writeDoubleVector(outfile, deltaS);
        outfile << ",\n";

        outfile << "      \"delta_standard_gibbs\": ";
        writeDoubleVector(outfile, deltaSSG);
        outfile << ",\n";

        outfile << "      \"delta_standard_enthalpy\": ";
        writeDoubleVector(outfile, deltaSSH);
        outfile << ",\n";

        outfile << "      \"delta_standard_entropy\": ";
        writeDoubleVector(outfile, deltaSSS);
        outfile << ",\n";

        outfile << "      \"third_body_concentrations\": ";
        writeDoubleVector(outfile, concm);
        outfile << "\n";

        outfile << "    }";
        cout << " done." << endl;
        case_num++;
    }

    outfile << "\n  ],\n";
    outfile << "  \"total_cases\": " << case_num << "\n";
    outfile << "}\n";

    cout << "Finished executing native kinetics oracle for " << case_num << " cases." << endl;
    return 0;
}
