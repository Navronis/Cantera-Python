#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include <cantera/core.h>
#include <cantera/thermo/PengRobinson.h>
#include <cantera/thermo/RedlichKwongMFTP.h>

using namespace Cantera;

static double extractDouble(const std::string& line, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t kpos = line.find(needle);
    if (kpos == std::string::npos) return 0.0;
    size_t start = kpos + needle.size();
    while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) start++;
    size_t end = line.find_first_of(",}", start);
    return std::stod(line.substr(start, end - start));
}

static std::string extractString(const std::string& line, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t kpos = line.find(needle);
    if (kpos == std::string::npos) return "";
    size_t q1 = line.find("\"", kpos + needle.size());
    if (q1 == std::string::npos) return "";
    size_t q2 = line.find("\"", q1 + 1);
    if (q2 == std::string::npos) return "";
    return line.substr(q1 + 1, q2 - q1 - 1);
}

int main(int argc, char** argv) {
    std::string root_dir = "/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs";
    std::string corpus_path = root_dir + "/cantera-python/audit/corpus_wave2.jsonl";
    std::string out_path = root_dir + "/cantera-python/audit/raw_native_wave2_internal.jsonl";
    std::string test_data = root_dir + "/cantera-original/test/data";

    if (argc > 1) corpus_path = argv[1];
    if (argc > 2) out_path = argv[2];

    std::cout << "Loading models...\n";
    auto solPR = newSolution(test_data + "/thermo-models.yaml", "CO2-PR");
    auto pr = std::dynamic_pointer_cast<PengRobinson>(solPR->thermo());
    auto solRK = newSolution(test_data + "/co2_RK_example.yaml");
    auto rk = std::dynamic_pointer_cast<RedlichKwongMFTP>(solRK->thermo());

    std::ifstream infile(corpus_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open corpus: " << corpus_path << "\n";
        return 1;
    }

    std::ofstream outfile(out_path);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output: " << out_path << "\n";
        return 1;
    }

    std::string line;
    size_t count = 0;
    std::cout << "Running Native Wave 2 Internal Cubic Probe...\n";

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::string cid = extractString(line, "case_id");
        std::string model = extractString(line, "model");
        double T = extractDouble(line, "temperature");
        double P = extractDouble(line, "pressure");

        std::ostringstream oss;
        oss << std::setprecision(17);
        oss << "{\"case_id\": \"" << cid << "\", \"wave\": 2, \"model\": \"" << model << "\"";

        try {
            int nsol = 0;
            double vroot[3] = {0.0, 0.0, 0.0};
            double mv = 0.0;
            double rho = 0.0;

            if (model == "CO2-PR") {
                pr->setState_TP(T, P);
                mv = pr->molarVolume();
                rho = pr->density();
                double a = 0.0, b = 0.0, aAlpha = 0.0;
                pr->calculateAB(a, b, aAlpha);
                nsol = pr->solveCubic(T, P, a, b, aAlpha, vroot);
            } else if (model == "CO2-RK") {
                rk->setState_TP(T, P);
                mv = rk->molarVolume();
                rho = rk->density();
                double a = 0.0, b = 0.0;
                rk->calculateAB(T, a, b);
                nsol = rk->solveCubic(T, P, a, b, vroot);
            } else {
                throw CanteraError("main", "Unknown model: " + model);
            }

            oss << ", \"status\": \"SUCCESS\""
                << ", \"nsol\": " << nsol
                << ", \"vroots\": [" << vroot[0] << ", " << vroot[1] << ", " << vroot[2] << "]"
                << ", \"molar_volume\": " << mv
                << ", \"density\": " << rho
                << "}\n";
        } catch (std::exception& e) {
            std::string msg = e.what();
            for (char& c : msg) {
                if (c == '"' || c == '\\') c = '\'';
                else if (c == '\n' || c == '\r') c = ' ';
            }
            oss << ", \"status\": \"FAILURE\""
                << ", \"error_type\": \"CanteraError\""
                << ", \"error_msg\": \"" << msg << "\""
                << "}\n";
        }

        outfile << oss.str();
        count++;
    }

    std::cout << "Native Wave 2 Internal Cubic Probe completed: " << count << " records written to " << out_path << "\n";
    return 0;
}
