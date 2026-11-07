#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <span>

#include "cantera/base/Solution.h"
#include "cantera/transport/TransportFactory.h"
#include "cantera/transport/WaterTransport.h"
#include "cantera/transport/UnityLewisTransport.h"
#include "cantera/transport/IonGasTransport.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/base/global.h"

using namespace Cantera;
using namespace std;

string escapeJson(const string& s) {
    ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else o << c;
    }
    return o.str();
}

string getJsonString(const string& json, const string& key, const string& defVal="") {
    string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == string::npos) return defVal;
    pos += pattern.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos < json.length() && json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        if (end != string::npos) {
            return json.substr(pos + 1, end - pos - 1);
        }
    }
    return defVal;
}

double getJsonDouble(const string& json, const string& key, double defVal=0.0) {
    string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == string::npos) return defVal;
    pos += pattern.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    size_t end = json.find_first_of(",}\n\r", pos);
    if (end != string::npos) {
        string sub = json.substr(pos, end - pos);
        return stod(sub);
    }
    return defVal;
}

int main(int argc, char** argv) {
    string corpusPath = "audit/corpus_transport_remaining.jsonl";
    string outputPath = "audit/native_transport_remaining_oracle_results.json";

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
    outfile << "  \"oracle_type\": \"native_transport_remaining_oracle\",\n";
    outfile << "  \"results\": [\n";

    string line;
    bool firstCase = true;
    int count = 0;

    while (getline(infile, line)) {
        if (line.empty()) continue;
        string caseId = getJsonString(line, "case_id");
        string category = getJsonString(line, "category");
        if (caseId.empty()) continue;

        if (!firstCase) outfile << ",\n";
        firstCase = false;
        count++;

        outfile << "    {\n";
        outfile << "      \"case_id\": \"" << escapeJson(caseId) << "\",\n";
        outfile << "      \"category\": \"" << escapeJson(category) << "\",\n";

        try {
            if (category == "water_transport") {
                double T = getJsonDouble(line, "temperature");
                double P = getJsonDouble(line, "pressure");
                auto phase = newThermo("thermo-models.yaml", "liquid-water");
                auto tran = newTransport(phase, "default");
                phase->setState_TP(T, P);

                double mu = tran->viscosity();
                double lam = tran->thermalConductivity();
                double bulk_mu = tran->bulkViscosity();
                string model = tran->transportModel();

                outfile << "      \"transport_model\": \"" << escapeJson(model) << "\",\n";
                outfile << "      \"viscosity\": " << setprecision(17) << mu << ",\n";
                outfile << "      \"thermal_conductivity\": " << setprecision(17) << lam << ",\n";
                outfile << "      \"bulk_viscosity\": " << setprecision(17) << bulk_mu << ",\n";
                outfile << "      \"density\": " << setprecision(17) << phase->density() << ",\n";
                outfile << "      \"isothermal_compressibility\": " << setprecision(17) << phase->isothermalCompressibility() << ",\n";
                outfile << "      \"thermal_expansion_coeff\": " << setprecision(17) << phase->thermalExpansionCoeff() << "\n";

            } else if (category == "unity_lewis_transport") {
                string input_file = getJsonString(line, "input_file", "gri30.yaml");
                string phase_name = getJsonString(line, "phase", "gri30");
                double T = getJsonDouble(line, "temperature", 1500.0);
                double P = getJsonDouble(line, "pressure", 101325.0);
                string X = getJsonString(line, "mole_fractions", "");

                auto sol = newSolution(input_file, phase_name, "unity-Lewis-number");
                if (!X.empty()) {
                    sol->thermo()->setState_TPX(T, P, X);
                } else {
                    sol->thermo()->setState_TP(T, P);
                }

                auto tran = sol->transport();
                size_t nsp = sol->thermo()->nSpecies();
                vector<double> d(nsp, 0.0);
                vector<double> dt(nsp, 0.0);
                tran->getMixDiffCoeffs(span<double>(d));
                tran->getThermalDiffCoeffs(span<double>(dt));

                double mu = tran->viscosity();
                double lam = tran->thermalConductivity();
                string model = tran->transportModel();

                outfile << "      \"transport_model\": \"" << escapeJson(model) << "\",\n";
                outfile << "      \"viscosity\": " << setprecision(17) << mu << ",\n";
                outfile << "      \"thermal_conductivity\": " << setprecision(17) << lam << ",\n";
                outfile << "      \"mix_diff_coeffs\": [";
                for (size_t i = 0; i < nsp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << d[i];
                }
                outfile << "],\n";
                outfile << "      \"thermal_diff_coeffs\": [";
                for (size_t i = 0; i < nsp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << dt[i];
                }
                outfile << "]\n";

            } else if (category == "ion_gas_transport") {
                string input_file = getJsonString(line, "input_file", "ch4_ion.yaml");
                string phase_name = getJsonString(line, "phase", "gas");
                double T = getJsonDouble(line, "temperature", 1200.0);
                double P = getJsonDouble(line, "pressure", 101325.0);
                string X = getJsonString(line, "mole_fractions", "");

                auto sol = newSolution(input_file, phase_name, "ionized-gas");
                if (!X.empty()) {
                    sol->thermo()->setState_TPX(T, P, X);
                } else {
                    sol->thermo()->setState_TP(T, P);
                }

                auto tran = sol->transport();
                size_t nsp = sol->thermo()->nSpecies();
                vector<double> mobi(nsp, 0.0);
                vector<double> d(nsp, 0.0);
                tran->getMobilities(span<double>(mobi));
                tran->getMixDiffCoeffs(span<double>(d));

                double mu = tran->viscosity();
                double lam = tran->thermalConductivity();
                double sigma = tran->electricalConductivity();
                string model = tran->transportModel();

                outfile << "      \"transport_model\": \"" << escapeJson(model) << "\",\n";
                outfile << "      \"viscosity\": " << setprecision(17) << mu << ",\n";
                outfile << "      \"thermal_conductivity\": " << setprecision(17) << lam << ",\n";
                outfile << "      \"electrical_conductivity\": " << setprecision(17) << sigma << ",\n";
                outfile << "      \"mobilities\": [";
                for (size_t i = 0; i < nsp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << mobi[i];
                }
                outfile << "],\n";
                outfile << "      \"mix_diff_coeffs\": [";
                for (size_t i = 0; i < nsp; i++) {
                    if (i > 0) outfile << ", ";
                    outfile << setprecision(17) << d[i];
                }
                outfile << "]\n";

            } else if (category == "transport_factory") {
                string model_req = getJsonString(line, "model");
                string input_file = getJsonString(line, "input_file", "gri30.yaml");
                string phase_name = getJsonString(line, "phase", "gri30");

                auto sol = newSolution(input_file, phase_name, "none");
                TransportFactory* fac = TransportFactory::factory();
                auto tr = fac->newTransport(model_req, sol->thermo());
                string model_actual = tr->transportModel();
                delete tr;

                outfile << "      \"requested_model\": \"" << escapeJson(model_req) << "\",\n";
                outfile << "      \"actual_model\": \"" << escapeJson(model_actual) << "\"\n";
            }
        } catch (const exception& e) {
            outfile << "      \"error\": \"" << escapeJson(e.what()) << "\"\n";
        }
        outfile << "    }";
    }

    outfile << "\n  ]\n}\n";
    cout << "Executed " << count << " native oracle transport cases." << endl;
    return 0;
}
