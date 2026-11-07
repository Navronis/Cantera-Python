/**
 * Pinned native Cantera Kinetics Family 4 differential validation oracle.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_kinetics_family4.jsonl, writing exact native oracle outputs to
 * audit/native_kinetics_family4_oracle_results.json.
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
#include "cantera/base/Interface.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/kinetics/InterfaceKinetics.h"
#include "cantera/kinetics/EdgeKinetics.h"
#include "cantera/kinetics/Reaction.h"
#include "cantera/kinetics/ReactionPath.h"
#include "cantera/kinetics/Group.h"
#include "cantera/kinetics/KineticsFactory.h"
#include "cantera/kinetics/ReactionRateFactory.h"
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

static string trim(const string& s) {
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == string::npos) return "";
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, (last - first + 1));
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

static vector<string> getJsonStringArray(const string& line, const string& key) {
    vector<string> res;
    string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == string::npos) return res;
    pos = line.find('[', pos);
    if (pos == string::npos) return res;
    size_t end = line.find(']', pos);
    if (end == string::npos) return res;
    string inner = line.substr(pos + 1, end - pos - 1);
    stringstream ss(inner);
    string token;
    while (getline(ss, token, ',')) {
        string s = trim(token);
        if (s.length() >= 2 && s.front() == '"' && s.back() == '"') {
            res.push_back(s.substr(1, s.length() - 2));
        }
    }
    return res;
}

static vector<int> getJsonIntArray(const string& line, const string& key) {
    vector<int> res;
    string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == string::npos) return res;
    pos = line.find('[', pos);
    if (pos == string::npos) return res;
    size_t end = line.find(']', pos);
    if (end == string::npos) return res;
    string inner = line.substr(pos + 1, end - pos - 1);
    stringstream ss(inner);
    string token;
    while (getline(ss, token, ',')) {
        string s = trim(token);
        if (!s.empty()) {
            try {
                res.push_back(stoi(s));
            } catch (...) {}
        }
    }
    return res;
}

int main(int argc, char* argv[]) {
    string corpusPath = "audit/corpus_kinetics_family4.jsonl";
    string outputPath = "audit/native_kinetics_family4_oracle_results.json";

    if (argc >= 2) corpusPath = argv[1];
    if (argc >= 3) outputPath = argv[2];

    ifstream infile(corpusPath);
    if (!infile.is_open()) {
        cerr << "Failed to open corpus file: " << corpusPath << endl;
        return 1;
    }

    cout << "Executing Pinned Native Kinetics Family 4 Oracle against " << corpusPath << endl;

    ofstream outfile(outputPath);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << outputPath << endl;
        return 1;
    }

    outfile << "{\n";
    outfile << "  \"schema_version\": \"1.0.0\",\n";
    outfile << "  \"subsystem\": \"kinetics\",\n";
    outfile << "  \"closure_family\": \"FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES\",\n";
    outfile << "  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n";
    outfile << "  \"results\": [\n";

    string line;
    int caseIdx = 0;
    bool firstCase = true;

    while (getline(infile, line)) {
        if (trim(line).empty()) continue;
        caseIdx++;

        string caseId = getJsonString(line, "case_id");
        string category = getJsonString(line, "category");

        cout << "  [" << caseIdx << "] " << caseId << " (" << category << ")..." << flush;

        if (!firstCase) outfile << ",\n";
        firstCase = false;

        outfile << "    {\n";
        outfile << "      \"case_id\": \"" << escapeJson(caseId) << "\",\n";
        outfile << "      \"category\": \"" << escapeJson(category) << "\",\n";

        if (category == "reaction_path_diagram") {
            string mech = getJsonString(line, "mechanism");
            string phaseName = getJsonString(line, "phase", "");
            string element = getJsonString(line, "element", "H");
            double T = getJsonDouble(line, "temperature", 1000.0);
            double P = getJsonDouble(line, "pressure", 101325.0);
            string X = getJsonString(line, "mole_fractions", "");
            string flowType = getJsonString(line, "flow_type", "NetFlow");
            double scale = getJsonDouble(line, "scale", -1.0);
            double threshold = getJsonDouble(line, "threshold", 0.005);
            vector<string> inc = getJsonStringArray(line, "included");
            vector<string> exc = getJsonStringArray(line, "excluded");
            int dispOnly = getJsonInt(line, "display_only", -1);

            shared_ptr<Solution> sol;
            if (!phaseName.empty()) {
                sol = newSolution(mech, phaseName);
            } else {
                sol = newSolution(mech);
            }

            if (!X.empty()) {
                sol->thermo()->setState_TPX(T, P, X);
            } else {
                sol->thermo()->setState_TP(T, P);
            }

            auto diag = newReactionPathDiagram(sol->kinetics(), element);
            diag->setFlowType(flowType);
            diag->scale = scale;
            diag->threshold = threshold;
            for (const auto& s : inc) diag->include(s);
            for (const auto& s : exc) diag->exclude(s);
            if (dispOnly >= 0) diag->displayOnly(dispOnly);

            diag->build();

            double maxFlow = diag->maxFlow();
            size_t nNodes = diag->nNodes();
            size_t nPaths = diag->nPaths();
            vector<size_t> sp = diag->species();
            vector<int> rxns = diag->reactions();
            string dataStr = diag->getData();
            string dotStr = diag->getDot();

            outfile << "      \"maxFlow\": " << setprecision(17) << maxFlow << ",\n";
            outfile << "      \"nNodes\": " << nNodes << ",\n";
            outfile << "      \"nPaths\": " << nPaths << ",\n";
            outfile << "      \"species\": [";
            for (size_t i = 0; i < sp.size(); i++) {
                if (i > 0) outfile << ", ";
                outfile << sp[i];
            }
            outfile << "],\n";
            outfile << "      \"reactions\": [";
            for (size_t i = 0; i < rxns.size(); i++) {
                if (i > 0) outfile << ", ";
                outfile << rxns[i];
            }
            outfile << "],\n";
            outfile << "      \"data\": \"" << escapeJson(dataStr) << "\",\n";
            outfile << "      \"dot\": \"" << escapeJson(dotStr) << "\",\n";

            // Pairs flow
            outfile << "      \"pairwise_flows\": [\n";
            bool firstPair = true;
            for (size_t i1 = 0; i1 < sp.size(); i1++) {
                size_t k1 = sp[i1];
                for (size_t i2 = i1 + 1; i2 < sp.size(); i2++) {
                    size_t k2 = sp[i2];
                    double f1 = diag->flow(k1, k2);
                    double f2 = diag->flow(k2, k1);
                    double net = diag->netFlow(k1, k2);
                    if (!firstPair) outfile << ",\n";
                    firstPair = false;
                    outfile << "        {\"k1\": " << k1 << ", \"k2\": " << k2
                            << ", \"flow_fwd\": " << setprecision(17) << f1
                            << ", \"flow_rev\": " << setprecision(17) << f2
                            << ", \"netFlow\": " << setprecision(17) << net << "}";
                }
            }
            outfile << "\n      ]\n";

        } else if (category == "reaction_inspection") {
            string mech = getJsonString(line, "mechanism");
            string phaseName = getJsonString(line, "phase", "");
            int rxnIdx = getJsonInt(line, "reaction_index", 0);

            shared_ptr<Solution> sol;
            if (!phaseName.empty()) {
                sol = newSolution(mech, phaseName);
            } else {
                sol = newSolution(mech);
            }

            auto kin = sol->kinetics();
            auto rxn = kin->reaction(rxnIdx);

            string eq = rxn->equation();
            string rStr = rxn->reactantString();
            string pStr = rxn->productString();
            string rType = rxn->type();
            bool rev = rxn->reversible;
            bool dup = rxn->duplicate;
            bool val = rxn->valid();
            bool tbody = rxn->usesThirdBody();
            bool electro = rxn->usesElectrochemistry(*kin);

            outfile << "      \"equation\": \"" << escapeJson(eq) << "\",\n";
            outfile << "      \"reactantString\": \"" << escapeJson(rStr) << "\",\n";
            outfile << "      \"productString\": \"" << escapeJson(pStr) << "\",\n";
            outfile << "      \"type\": \"" << escapeJson(rType) << "\",\n";
            outfile << "      \"reversible\": " << (rev ? "true" : "false") << ",\n";
            outfile << "      \"duplicate\": " << (dup ? "true" : "false") << ",\n";
            outfile << "      \"valid\": " << (val ? "true" : "false") << ",\n";
            outfile << "      \"usesThirdBody\": " << (tbody ? "true" : "false") << ",\n";
            outfile << "      \"usesElectrochemistry\": " << (electro ? "true" : "false") << ",\n";

            outfile << "      \"reactants\": {";
            bool firstR = true;
            for (const auto& [name, stoich] : rxn->reactants) {
                if (!firstR) outfile << ", ";
                firstR = false;
                outfile << "\"" << escapeJson(name) << "\": " << setprecision(17) << stoich;
            }
            outfile << "},\n";

            outfile << "      \"products\": {";
            bool firstP = true;
            for (const auto& [name, stoich] : rxn->products) {
                if (!firstP) outfile << ", ";
                firstP = false;
                outfile << "\"" << escapeJson(name) << "\": " << setprecision(17) << stoich;
            }
            outfile << "}\n";

        } else if (category == "group_algebra") {
            vector<int> comp1 = getJsonIntArray(line, "comp1");
            vector<int> comp2 = getJsonIntArray(line, "comp2");
            vector<string> elemSyms = getJsonStringArray(line, "elements");

            Group g1;
            if (!comp1.empty()) {
                g1 = Group(comp1);
            }

            ostringstream fmt1;
            g1.fmt(fmt1, elemSyms);

            outfile << "      \"g1_valid\": " << (g1.valid() ? "true" : "false") << ",\n";
            outfile << "      \"g1_sign\": " << g1.sign() << ",\n";
            outfile << "      \"g1_size\": " << g1.size() << ",\n";
            outfile << "      \"g1_nAtoms\": " << g1.nAtoms() << ",\n";
            outfile << "      \"g1_fmt\": \"" << escapeJson(fmt1.str()) << "\"";

            if (!comp2.empty()) {
                Group g2(comp2);
                Group gSum = g1 + g2;
                Group gDiff = g1 - g2;

                ostringstream fmtSum, fmtDiff;
                gSum.fmt(fmtSum, elemSyms);
                gDiff.fmt(fmtDiff, elemSyms);

                outfile << ",\n";
                outfile << "      \"sum_valid\": " << (gSum.valid() ? "true" : "false") << ",\n";
                outfile << "      \"sum_sign\": " << gSum.sign() << ",\n";
                outfile << "      \"sum_nAtoms\": " << gSum.nAtoms() << ",\n";
                outfile << "      \"sum_fmt\": \"" << escapeJson(fmtSum.str()) << "\",\n";
                outfile << "      \"diff_valid\": " << (gDiff.valid() ? "true" : "false") << ",\n";
                outfile << "      \"diff_sign\": " << gDiff.sign() << ",\n";
                outfile << "      \"diff_nAtoms\": " << gDiff.nAtoms() << ",\n";
                outfile << "      \"diff_fmt\": \"" << escapeJson(fmtDiff.str()) << "\"";
            }
            outfile << "\n";

        } else if (category == "kinetics_factory") {
            vector<string> models = getJsonStringArray(line, "models");
            outfile << "      \"models\": {\n";
            for (size_t i = 0; i < models.size(); i++) {
                if (i > 0) outfile << ",\n";
                string m = models[i];
                auto kin = newKinetics(m);
                string ktype = kin ? kin->kineticsType() : "null";
                outfile << "        \"" << escapeJson(m) << "\": \"" << escapeJson(ktype) << "\"";
            }
            outfile << "\n      }\n";

        } else if (category == "rate_factory") {
            vector<string> rtypes = getJsonStringArray(line, "rate_types");
            outfile << "      \"rate_types\": {\n";
            for (size_t i = 0; i < rtypes.size(); i++) {
                if (i > 0) outfile << ",\n";
                string rt = rtypes[i];
                auto rate = newReactionRate(rt);
                string rname = rate ? rate->type() : "null";
                outfile << "        \"" << escapeJson(rt) << "\": \"" << escapeJson(rname) << "\"";
            }
            outfile << "\n      }\n";
        }

        outfile << "    }";
        cout << " done." << endl;
    }

    outfile << "\n  ]\n}\n";
    cout << "Finished executing native kinetics Family 4 oracle for " << caseIdx << " cases." << endl;
    return 0;
}
