#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <limits>

#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

using namespace Cantera;

// Minimal JSON parser helper for string values
static std::string extractString(const std::string& line, const std::string& key) {
    size_t kpos = line.find("\"" + key + "\":");
    if (kpos == std::string::npos) return "";
    size_t q1 = line.find('"', kpos + key.size() + 2);
    if (q1 == std::string::npos) return "";
    size_t q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return line.substr(q1 + 1, q2 - q1 - 1);
}

// Minimal JSON parser helper for double values (including nan, inf)
static double extractDouble(const std::string& line, const std::string& key) {
    size_t kpos = line.find("\"" + key + "\":");
    if (kpos == std::string::npos) return 0.0;
    size_t start = kpos + key.size() + 2;
    while (start < line.size() && (line[start] == ' ' || line[start] == ':')) start++;
    if (line.substr(start, 3) == "NaN" || line.substr(start, 3) == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (line.substr(start, 8) == "Infinity" || line.substr(start, 3) == "inf") {
        return std::numeric_limits<double>::infinity();
    }
    if (line.substr(start, 9) == "-Infinity" || line.substr(start, 4) == "-inf") {
        return -std::numeric_limits<double>::infinity();
    }
    size_t end = line.find_first_of(",}", start);
    return std::stod(line.substr(start, end - start));
}

static int extractInt(const std::string& line, const std::string& key) {
    size_t kpos = line.find("\"" + key + "\":");
    if (kpos == std::string::npos) return 0;
    size_t start = kpos + key.size() + 2;
    while (start < line.size() && (line[start] == ' ' || line[start] == ':')) start++;
    size_t end = line.find_first_of(",}", start);
    return std::stoi(line.substr(start, end - start));
}

// Escape JSON string
static std::string escapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string input_path = (argc > 1) ? argv[1] : "audit/corpus_wave4_failures.jsonl";
    std::string output_path = (argc > 2) ? argv[2] : "audit/native_failure_raw.jsonl";

    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open failure input: " << input_path << std::endl;
        return 1;
    }
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open failure output: " << output_path << std::endl;
        return 1;
    }

    std::cout << "Executing Native Cantera C++ Failure-Semantics Suite..." << std::endl;

    std::string line;
    int count = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        int case_id = extractInt(line, "case_id");
        std::string name = extractString(line, "name");
        std::string api_target = extractString(line, "api_target");
        std::string mech = extractString(line, "mech");
        if (mech.empty()) mech = "h2o2.yaml";

        std::string native_status = "SUCCESS";
        std::string exception_type = "None";
        std::string exception_msg = "";

        try {
            if (api_target == "transport_model") {
                // Testing invalid transport model creation
                std::string model = extractString(line, "model");
                auto soln = newSolution(mech, "", "none");
                auto tr = newTransport(soln->thermo(), model);
                (void)tr;
            } else if (api_target == "state_tpx") {
                // Testing invalid thermodynamic state directly in Cantera
                double T = extractDouble(line, "T");
                double P = extractDouble(line, "P");
                auto soln = newSolution(mech, "", "none");
                auto thermo = soln->thermo();
                auto multi = std::dynamic_pointer_cast<MultiTransport>(newTransport(thermo, "multicomponent"));

                // Parse composition
                size_t xpos = line.find("\"X\": {");
                if (xpos != std::string::npos) {
                    size_t start = xpos + 5;
                    size_t end = line.find("}", start);
                    std::string body = line.substr(start, end - start);
                    std::stringstream ss(body);
                    std::string token;
                    std::map<std::string, double> comp;
                    while (std::getline(ss, token, ',')) {
                        size_t cpos = token.find(':');
                        if (cpos == std::string::npos) continue;
                        size_t q1 = token.find('"');
                        size_t q2 = token.find('"', q1 + 1);
                        if (q1 != std::string::npos && q2 != std::string::npos) {
                            std::string sp = token.substr(q1 + 1, q2 - q1 - 1);
                            std::string valstr = token.substr(cpos + 1);
                            double v = 0.0;
                            if (valstr.find("nan") != std::string::npos || valstr.find("NaN") != std::string::npos) {
                                v = std::numeric_limits<double>::quiet_NaN();
                            } else {
                                v = std::stod(valstr);
                            }
                            comp[sp] = v;
                        }
                    }
                    thermo->setMoleFractionsByName(comp);
                }
                thermo->setState_TP(T, P);
                // Also trigger transport evaluations
                multi->thermalConductivity();
            } else if (api_target == "flux_dimensions") {
                // Testing invalid dimensions in getSpeciesFluxes
                auto soln = newSolution(mech, "", "none");
                auto thermo = soln->thermo();
                auto multi = std::dynamic_pointer_cast<MultiTransport>(newTransport(thermo, "multicomponent"));
                size_t nsp = thermo->nSpecies();
                thermo->setState_TPX(1000.0, 101325.0, "H2:0.5, O2:0.5");

                int ndir = extractInt(line, "ndir");
                int ldx = extractInt(line, "ldx");
                int ldf = extractInt(line, "ldf");

                std::vector<double> grad_T(std::max(1, ndir), 0.0);
                std::vector<double> grad_X(std::max(1, ldx * ndir), 0.0);
                std::vector<double> fluxes(std::max(1, ldf * ndir), 0.0);

                multi->getSpeciesFluxes(ndir, grad_T, ldx, grad_X, ldf, fluxes);
            }
        } catch (const CanteraError& e) {
            native_status = "FAILED";
            exception_type = "CanteraError";
            exception_msg = e.what();
        } catch (const std::exception& e) {
            native_status = "FAILED";
            exception_type = "std::exception";
            exception_msg = e.what();
        } catch (...) {
            native_status = "FAILED";
            exception_type = "UnknownException";
            exception_msg = "Unknown C++ exception";
        }

        outfile << "{\"case_id\":" << case_id
                << ",\"name\":\"" << name << "\""
                << ",\"native_status\":\"" << native_status << "\""
                << ",\"native_exception_type\":\"" << exception_type << "\""
                << ",\"native_message\":\"" << escapeJson(exception_msg) << "\"}\n";

        std::cout << "  Native Case " << case_id << " (" << name << "): "
                  << native_status << " [" << exception_type << "]" << std::endl;
        count++;
    }

    std::cout << "Successfully evaluated " << count << " native failure test cases!" << std::endl;
    return 0;
}
