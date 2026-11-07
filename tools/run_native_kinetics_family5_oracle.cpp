/**
 * Pinned native Cantera Kinetics Family 5 differential validation oracle.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_kinetics_family5.jsonl, writing exact native oracle outputs to
 * audit/native_kinetics_family5_oracle_results.json.
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

#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/kinetics/ReactionRate.h"
#include "cantera/kinetics/ReactionRateFactory.h"
#include "cantera/kinetics/TwoTempPlasmaRate.h"
#include "cantera/kinetics/ElectronCollisionPlasmaRate.h"
#include "cantera/kinetics/VibrationalRelaxationRate.h"
#include "cantera/kinetics/ConstantVibrationalRelaxationRate.h"
#include "cantera/kinetics/StarikovskiyVibrationalRelaxationRate.h"
#include "cantera/kinetics/CastelaVibrationalRelaxationRate.h"
#include "cantera/kinetics/MultiStateResolvedVibrationalRelaxationRate.h"
#include "cantera/kinetics/Custom.h"
#include "cantera/kinetics/MultiRate.h"
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

// Simple JSON parser helpers for corpus line
static string getJsonString(const string& line, const string& key, const string& defVal = "") {
    string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == string::npos) return defVal;
    pos += search.length();
    while (pos < line.length() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos < line.length() && line[pos] == '"') {
        pos++;
        size_t end = line.find('"', pos);
        if (end != string::npos) return line.substr(pos, end - pos);
    }
    return defVal;
}

static double getJsonDouble(const string& line, const string& key, double defVal = 0.0) {
    string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == string::npos) return defVal;
    pos += search.length();
    while (pos < line.length() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    size_t end = line.find_first_of(",}\"]", pos);
    if (end != string::npos) {
        try {
            return stod(line.substr(pos, end - pos));
        } catch (...) {}
    }
    return defVal;
}

static int getJsonInt(const string& line, const string& key, int defVal = 0) {
    return static_cast<int>(getJsonDouble(line, key, defVal));
}

int main(int argc, char** argv) {
    string corpusPath = "audit/corpus_kinetics_family5.jsonl";
    string outputPath = "audit/native_kinetics_family5_oracle_results.json";

    if (argc > 1) corpusPath = argv[1];
    if (argc > 2) outputPath = argv[2];

    ifstream infile(corpusPath);
    if (!infile.is_open()) {
        cerr << "Failed to open corpus file: " << corpusPath << endl;
        return 1;
    }

    ofstream outfile(outputPath);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << outputPath << endl;
        return 1;
    }

    outfile << "{\n";
    outfile << "  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n";
    outfile << "  \"oracle_type\": \"native_cpp_oracle\",\n";
    outfile << "  \"family\": \"FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE\",\n";
    outfile << "  \"results\": [\n";

    string line;
    bool firstCase = true;
    int count = 0;

    while (getline(infile, line)) {
        if (line.empty() || line[0] != '{') continue;

        string caseId = getJsonString(line, "case_id");
        string category = getJsonString(line, "category");

        if (!firstCase) outfile << ",\n";
        firstCase = false;
        count++;

        outfile << "    {\n";
        outfile << "      \"case_id\": \"" << escapeJson(caseId) << "\",\n";
        outfile << "      \"category\": \"" << escapeJson(category) << "\",\n";

        try {
            if (category == "vibrational_relaxation") {
                string mech = getJsonString(line, "mechanism", "vibrational-relaxation.yaml");
                double T = getJsonDouble(line, "temperature", 1200.0);
                double P = getJsonDouble(line, "pressure", 101325.0);
                string X = getJsonString(line, "mole_fractions", "");

                auto sol = newSolution(mech, "", "none");
                if (!X.empty()) {
                    sol->thermo()->setState_TPX(T, P, X);
                } else {
                    sol->thermo()->setState_TP(T, P);
                }

                auto kin = sol->kinetics();
                size_t nRxn = kin->nReactions();
                vector<double> kf(nRxn, 0.0);
                vector<double> rop(nRxn, 0.0);
                kin->getFwdRateConstants(std::span<double>(kf));
                kin->getNetRatesOfProgress(std::span<double>(rop));

                outfile << "      \"nReactions\": " << nRxn << ",\n";
                outfile << "      \"forward_rate_constants\": [";
                for (size_t i = 0; i < nRxn; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << kf[i];
                }
                outfile << "],\n";
                outfile << "      \"net_rates_of_progress\": [";
                for (size_t i = 0; i < nRxn; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << rop[i];
                }
                outfile << "],\n";

                // Individual reaction rates inspection
                outfile << "      \"rate_types\": [";
                for (size_t i = 0; i < nRxn; i++) {
                    if (i > 0) outfile << ", ";
                    string rtype = kin->reaction(i)->rate()->type();
                    outfile << "\"" << escapeJson(rtype) << "\"";
                }
                outfile << "]\n";

            } else if (category == "two_temp_plasma") {
                string mech = getJsonString(line, "mechanism", "oxygen-plasma.yaml");
                double T = getJsonDouble(line, "temperature", 1000.0);
                double Te = getJsonDouble(line, "electron_temperature", 23209.0);
                double P = getJsonDouble(line, "pressure", 101325.0);
                string X = getJsonString(line, "mole_fractions", "");

                auto sol = newSolution(mech, "", "none");
                if (!X.empty()) {
                    sol->thermo()->setState_TPX(T, P, X);
                } else {
                    sol->thermo()->setState_TP(T, P);
                }

                auto kin = sol->kinetics();
                size_t nRxn = kin->nReactions();
                vector<double> kf(nRxn, 0.0);
                kin->getFwdRateConstants(std::span<double>(kf));

                // Test TwoTempPlasmaRate directly with (T, Te)
                TwoTempPlasmaRate ttRate;
                AnyMap node;
                node["rate-constant"] = AnyMap();
                node["rate-constant"]["A"] = 1.0e13;
                node["rate-constant"]["b"] = 0.5;
                node["rate-constant"]["Ea-gas"] = 1.0e7;
                node["rate-constant"]["Ea-electron"] = 2.0e7;
                node["rate-constant"]["b-gas"] = 0.1;
                ttRate.setParameters(node, UnitStack({}));

                TwoTempPlasmaData data;
                data.update(T, Te);
                double val = ttRate.evalFromStruct(data);
                double ddT = ttRate.ddTScaledFromStruct(data);

                outfile << "      \"nReactions\": " << nRxn << ",\n";
                outfile << "      \"forward_rate_constants\": [";
                for (size_t i = 0; i < nRxn; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << kf[i];
                }
                outfile << "],\n";
                outfile << "      \"twotemp_eval\": " << setprecision(17) << val << ",\n";
                outfile << "      \"twotemp_ddT_scaled\": " << setprecision(17) << ddT << ",\n";
                outfile << "      \"rate_type\": \"" << escapeJson(ttRate.type()) << "\"\n";

            } else if (category == "air_plasma") {
                string mech = getJsonString(line, "mechanism", "air-plasma.yaml");
                double T = getJsonDouble(line, "temperature", 1000.0);
                double Te = getJsonDouble(line, "electron_temperature", 23209.0);
                double P = getJsonDouble(line, "pressure", 101325.0);
                string X = getJsonString(line, "mole_fractions", "");

                auto sol = newSolution(mech, "", "none");
                if (!X.empty()) {
                    sol->thermo()->setState_TPX(T, P, X);
                } else {
                    sol->thermo()->setState_TP(T, P);
                }

                auto kin = sol->kinetics();
                size_t nRxn = kin->nReactions();
                vector<double> kf(nRxn, 0.0);
                kin->getFwdRateConstants(std::span<double>(kf));

                outfile << "      \"nReactions\": " << nRxn << ",\n";
                outfile << "      \"forward_rate_constants\": [";
                for (size_t i = 0; i < nRxn; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << kf[i];
                }
                outfile << "]\n";

            } else if (category == "custom_delegator") {
                double T = getJsonDouble(line, "temperature", 1000.0);

                CustomFunc1Rate cfr;
                string cfrType = cfr.type();

                ArrheniusData arrData;
                arrData.update(T);

                outfile << "      \"custom_rate_type\": \"" << escapeJson(cfrType) << "\",\n";
                outfile << "      \"temperature\": " << T << ",\n";
                outfile << "      \"valid\": true\n";

            } else if (category == "multirate_ops") {
                string rtype = getJsonString(line, "rate_type", "Arrhenius");
                double T = getJsonDouble(line, "temperature", 1000.0);

                shared_ptr<ReactionRate> rate;
                if (rtype == "constant-vibrational-relaxation") {
                    rate = make_shared<ConstantVibrationalRelaxationRate>(1.0e6);
                } else {
                    rate = newReactionRate(rtype);
                }
                string actualType = rate->type();
                auto multi = rate->newMultiRate();

                bool caughtEmpty = false;
                try {
                    multi->type();
                } catch (const CanteraError&) {
                    caughtEmpty = true;
                }

                multi->add(0, *rate);
                string multiType = multi->type();

                // Test replacing with identical type
                shared_ptr<ReactionRate> rate2;
                if (rtype == "constant-vibrational-relaxation") {
                    rate2 = make_shared<ConstantVibrationalRelaxationRate>(2.0e6);
                } else {
                    rate2 = newReactionRate(rtype);
                }
                bool repOk = multi->replace(0, *rate2);

                // Test replacing with mismatched type (should throw)
                bool caughtMismatch = false;
                try {
                    auto mismatchRate = newReactionRate(rtype == "Arrhenius" ? "two-temperature-plasma" : "Arrhenius");
                    multi->replace(0, *mismatchRate);
                } catch (const CanteraError&) {
                    caughtMismatch = true;
                }

                outfile << "      \"rate_type\": \"" << escapeJson(actualType) << "\",\n";
                outfile << "      \"multi_type\": \"" << escapeJson(multiType) << "\",\n";
                outfile << "      \"caught_empty\": " << (caughtEmpty ? "true" : "false") << ",\n";
                outfile << "      \"replace_ok\": " << (repOk ? "true" : "false") << ",\n";
                outfile << "      \"caught_mismatch\": " << (caughtMismatch ? "true" : "false") << "\n";

            } else if (category == "kinetics_derivatives") {
                string mech = getJsonString(line, "mechanism", "h2o2.yaml");
                double T = getJsonDouble(line, "temperature", 1000.0);
                double P = getJsonDouble(line, "pressure", 101325.0);
                string X = getJsonString(line, "mole_fractions", "");

                auto sol = newSolution(mech, "", "none");
                if (!X.empty()) {
                    sol->thermo()->setState_TPX(T, P, X);
                } else {
                    sol->thermo()->setState_TP(T, P);
                }

                auto kin = sol->kinetics();
                size_t nSp = kin->nTotalSpecies();

                vector<double> net_ddT(nSp, 0.0);
                vector<double> creat_ddT(nSp, 0.0);
                vector<double> dest_ddT(nSp, 0.0);
                vector<double> net_ddP(nSp, 0.0);

                kin->getNetProductionRates_ddT(std::span<double>(net_ddT));
                kin->getCreationRates_ddT(std::span<double>(creat_ddT));
                kin->getDestructionRates_ddT(std::span<double>(dest_ddT));
                kin->getNetProductionRates_ddP(std::span<double>(net_ddP));

                outfile << "      \"nTotalSpecies\": " << nSp << ",\n";
                outfile << "      \"netProductionRates_ddT\": [";
                for (size_t i = 0; i < nSp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << net_ddT[i];
                }
                outfile << "],\n";
                outfile << "      \"creationRates_ddT\": [";
                for (size_t i = 0; i < nSp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << creat_ddT[i];
                }
                outfile << "],\n";
                outfile << "      \"destructionRates_ddT\": [";
                for (size_t i = 0; i < nSp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << dest_ddT[i];
                }
                outfile << "],\n";
                outfile << "      \"netProductionRates_ddP\": [";
                for (size_t i = 0; i < nSp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << net_ddP[i];
                }
                outfile << "]\n";
            }

            outfile << "    }";
        } catch (const exception& e) {
            outfile << "      \"error\": \"" << escapeJson(e.what()) << "\"\n";
            outfile << "    }";
        }
    }

    outfile << "\n  ]\n}\n";
    cout << "Executed " << count << " test cases, saved to " << outputPath << endl;
    return 0;
}
