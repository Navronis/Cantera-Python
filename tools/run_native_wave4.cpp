#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

using namespace Cantera;

static const std::string NATIVE_ORACLE_SHA256 = "41b22f6cf4e0e0ec7f1fe015e9c44be568c526e24a0c1f0f5a2966549cf4cff2";

// Minimal json string parser helper for simple flat dicts
static std::map<std::string, double> parseMap(const std::string& line, const std::string& key) {
    std::map<std::string, double> result;
    size_t kpos = line.find("\"" + key + "\": {");
    if (kpos == std::string::npos) return result;
    size_t start = kpos + key.size() + 5;
    size_t end = line.find("}", start);
    if (end == std::string::npos) return result;
    std::string body = line.substr(start, end - start);
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t cpos = token.find(':');
        if (cpos == std::string::npos) continue;
        size_t q1 = token.find('"');
        size_t q2 = token.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            std::string name = token.substr(q1 + 1, q2 - q1 - 1);
            std::string valstr = token.substr(cpos + 1);
            double val = std::stod(valstr);
            result[name] = val;
        }
    }
    return result;
}

static double extractDouble(const std::string& line, const std::string& key) {
    size_t kpos = line.find("\"" + key + "\":");
    if (kpos == std::string::npos) return 0.0;
    size_t start = kpos + key.size() + 3;
    size_t end = line.find_first_of(",}", start);
    return std::stod(line.substr(start, end - start));
}

static int extractInt(const std::string& line, const std::string& key) {
    size_t kpos = line.find("\"" + key + "\":");
    if (kpos == std::string::npos) return 0;
    size_t start = kpos + key.size() + 3;
    size_t end = line.find_first_of(",}", start);
    return std::stoi(line.substr(start, end - start));
}

static std::string extractString(const std::string& line, const std::string& key) {
    std::string prefix = "\"" + key + "\": \"";
    size_t kpos = line.find(prefix);
    if (kpos == std::string::npos) return "";
    size_t start = kpos + prefix.size();
    size_t end = line.find('"', start);
    if (end == std::string::npos) return "";
    return line.substr(start, end - start);
}

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    std::string input_path = (argc > 1) ? argv[1] : "audit/corpus_wave4.jsonl";
    std::string output_path = (argc > 2) ? argv[2] : "audit/native_wave4_results.jsonl";
    std::string wave4_run_id = (argc > 3) ? argv[3] : "wave4_run_verified";

    std::cout << "Loading mechanisms for native evaluation..." << std::endl;
    auto soln_gri = newSolution("gri30.yaml", "", "none");
    auto thermo_gri = soln_gri->thermo();
    auto multi_gri = std::dynamic_pointer_cast<MultiTransport>(newTransport(thermo_gri, "multicomponent"));

    auto soln_h2o2 = newSolution("h2o2.yaml", "", "none");
    auto thermo_h2o2 = soln_h2o2->thermo();
    auto multi_h2o2 = std::dynamic_pointer_cast<MultiTransport>(newTransport(thermo_h2o2, "multicomponent"));

    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open input: " << input_path << std::endl;
        return 1;
    }
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output: " << output_path << std::endl;
        return 1;
    }

    std::string line;
    size_t count = 0;
    std::cout << "Processing Wave 4 cases (with FULL binary & multicomponent diffusion matrices)..." << std::endl;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        int case_id = extractInt(line, "case_id");
        std::string mech = extractString(line, "mech");
        double T = extractDouble(line, "T");
        double P = extractDouble(line, "P");
        double grad_T_val = extractDouble(line, "grad_T");
        auto comp = parseMap(line, "X");
        auto grad_X_map = parseMap(line, "grad_X");

        auto thermo = (mech == "h2o2.yaml") ? thermo_h2o2 : thermo_gri;
        auto multi = (mech == "h2o2.yaml") ? multi_h2o2 : multi_gri;

        size_t nsp = thermo->nSpecies();
        std::vector<double> X(nsp, 0.0);
        std::vector<double> grad_X(nsp, 0.0);
        for (size_t k = 0; k < nsp; ++k) {
            std::string sp = thermo->speciesName(k);
            if (comp.count(sp)) X[k] = comp[sp];
            if (grad_X_map.count(sp)) grad_X[k] = grad_X_map[sp];
        }

        // Set state
        thermo->setState_TPX(T, P, X);

        // 1. Viscosity
        double visc = multi->viscosity();

        // 2. Thermal conductivity
        double cond = multi->thermalConductivity();

        // 3. Species viscosities
        std::vector<double> sp_visc(nsp, 0.0);
        multi->getSpeciesViscosities(sp_visc);

        // 4. Thermal diffusion coefficients
        std::vector<double> dt(nsp, 0.0);
        multi->getThermalDiffCoeffs(dt);

        // 5. Binary diffusion coefficients (full N x N)
        Array2D bdiff(nsp, nsp, 0.0);
        multi->getBinaryDiffCoeffs(nsp, bdiff.data());

        // 6. Multicomponent diffusion coefficients (full N x N)
        Array2D dmulti(nsp, nsp, 0.0);
        multi->getMultiDiffCoeffs(nsp, dmulti.data());

        // 7. Diffusive species mass fluxes
        std::vector<double> grad_T = {grad_T_val};
        std::vector<double> fluxes(nsp, 0.0);
        multi->getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes);

        // Compute net flux sum (mass conservation check)
        double flux_sum = 0.0;
        for (double fl : fluxes) flux_sum += fl;

        // Write compact JSON record with full matrices & native oracle identity SHA256 & wave4_run_id
        outfile << "{\"case_id\":" << case_id
                << ",\"wave4_run_id\":\"" << wave4_run_id << "\""
                << ",\"native_oracle_identity_sha256\":\"" << NATIVE_ORACLE_SHA256 << "\""
                << ",\"viscosity\":" << std::setprecision(16) << visc
                << ",\"thermal_conductivity\":" << cond
                << ",\"flux_sum\":" << flux_sum
                << ",\"sp_visc\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << sp_visc[k] << (k + 1 < nsp ? "," : "");
        }
        outfile << "],\"dt\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << dt[k] << (k + 1 < nsp ? "," : "");
        }
        outfile << "],\"fluxes\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << fluxes[k] << (k + 1 < nsp ? "," : "");
        }
        outfile << "],\"molar_fluxes\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << (fluxes[k] / thermo->molecularWeight(k)) << (k + 1 < nsp ? "," : "");
        }
        // Full N x N binary diffusion matrix
        outfile << "],\"bdiff\":[";
        for (size_t i = 0; i < nsp; ++i) {
            outfile << "[";
            for (size_t j = 0; j < nsp; ++j) {
                outfile << bdiff(i, j) << (j + 1 < nsp ? "," : "");
            }
            outfile << "]" << (i + 1 < nsp ? "," : "");
        }
        // Full N x N multicomponent diffusion matrix
        outfile << "],\"dmulti\":[";
        for (size_t i = 0; i < nsp; ++i) {
            outfile << "[";
            for (size_t j = 0; j < nsp; ++j) {
                outfile << dmulti(i, j) << (j + 1 < nsp ? "," : "");
            }
            outfile << "]" << (i + 1 < nsp ? "," : "");
        }
        // Legacy row0 / diagonal for fast backward-compatibility
        outfile << "],\"dmulti_diag\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << dmulti(k, k) << (k + 1 < nsp ? "," : "");
        }
        outfile << "],\"dmulti_row0\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << dmulti(0, k) << (k + 1 < nsp ? "," : "");
        }
        outfile << "],\"bdiff_row0\":[";
        for (size_t k = 0; k < nsp; ++k) {
            outfile << bdiff(0, k) << (k + 1 < nsp ? "," : "");
        }
        outfile << "]}\n";

        count++;
        if (count % 2000 == 0) {
            std::cout << "Evaluated " << count << " / 10000 cases..." << std::endl;
        }
    }

    std::cout << "Successfully evaluated all " << count << " native cases!" << std::endl;
    return 0;
}
