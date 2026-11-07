#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "cantera/oneD/OneDim.h"
#include "cantera/oneD/Domain1D.h"
#include "cantera/base/ctexceptions.h"
#include "cantera/oneD/MultiJac.h"

class TestOneDimDomain : public Cantera::Domain1D
{
public:
    std::string m_domain_type;

    TestOneDimDomain(const std::string& name, const std::string& type, size_t nv, size_t np, size_t bw)
        : Cantera::Domain1D(nv, np), m_domain_type(type)
    {
        m_id = name;
        setBandwidth(static_cast<int>(bw));
        for (size_t i = 0; i < nv; ++i) {
            setComponentName(i, name + "_var_" + std::to_string(i));
            setBounds(i, -100.0 - 5.0 * static_cast<double>(i), 10000.0 + 10.0 * static_cast<double>(i));
        }
    }

    void setTolerances(const std::vector<double>& rtol, const std::vector<double>& atol) {
        for (size_t i = 0; i < m_nv; ++i) {
            m_rtol_ss[i] = rtol[i % rtol.size()];
            m_atol_ss[i] = atol[i % atol.size()];
        }
    }

    void eval(size_t j, Cantera::span<const double> x, Cantera::span<double> r,
              Cantera::span<int> mask, double rdt=0.0) override {
        size_t j0 = 0;
        size_t j1 = nPoints() - 1;
        if (j != Cantera::npos) {
            if (j < firstPoint() || j > lastPoint()) {
                // If point j is not in this domain or immediate boundary stencil, still check boundaries
                if (j + 1 < firstPoint() || (firstPoint() > 0 && j > lastPoint() + 1)) {
                    return;
                }
            }
            if (j >= firstPoint() && j <= lastPoint()) {
                size_t jloc = j - firstPoint();
                j0 = (jloc > 0) ? jloc - 1 : 0;
                j1 = std::min(nPoints() - 1, jloc + 1);
            } else if (j + 1 == firstPoint()) {
                j0 = 0;
                j1 = 0;
            } else if (j == lastPoint() + 1) {
                j0 = nPoints() - 1;
                j1 = nPoints() - 1;
            }
        }

        for (size_t pt = j0; pt <= j1; ++pt) {
            for (size_t k = 0; k < m_nv; ++k) {
                size_t idx = index(k, pt);
                size_t g_idx = loc() + idx;
                double val = x[g_idx];
                double lapl = 0.0;
                if (pt > 0 && pt < nPoints() - 1) {
                    double v_left = x[loc() + index(k, pt - 1)];
                    double v_right = x[loc() + index(k, pt + 1)];
                    lapl = v_right - 2.0 * val + v_left;
                } else if (pt == 0 && m_left) {
                    double v_left = x[m_left->loc() + m_left->index(k % m_left->nComponents(), m_left->nPoints() - 1)];
                    double v_right = (nPoints() > 1) ? x[loc() + index(k, 1)] : val;
                    lapl = v_right - 2.0 * val + v_left;
                } else if (pt == nPoints() - 1 && m_right) {
                    double v_left = (nPoints() > 1) ? x[loc() + index(k, pt - 1)] : val;
                    double v_right = x[m_right->loc() + m_right->index(k % m_right->nComponents(), 0)];
                    lapl = v_right - 2.0 * val + v_left;
                }
                r[g_idx] = -0.5 * lapl + 0.1 * val * val - 0.05 * val;
                if (rdt > 0.0) {
                    r[g_idx] += rdt * val;
                    mask[g_idx] = 1;
                }
            }
        }
    }
};

static std::string jsonEscapeString(const std::string& s) {
    std::ostringstream o;
    o << "\"";
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (static_cast<unsigned char>(c) <= 0x1f) {
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        } else {
            o << c;
        }
    }
    o << "\"";
    return o.str();
}

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string getJsonString(const std::string& line, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return "";
    size_t start = line.find("\"", pos + search.size());
    if (start == std::string::npos) return "";
    size_t end = line.find("\"", start + 1);
    return line.substr(start + 1, end - start - 1);
}

static double getJsonDouble(const std::string& line, const std::string& key, double def = 0.0) {
    std::string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return def;
    size_t start = pos + search.size();
    while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) start++;
    size_t end = line.find_first_of(",}\n\r", start);
    if (end == std::string::npos) end = line.size();
    try {
        return std::stod(line.substr(start, end - start));
    } catch (...) {
        return def;
    }
}

static int getJsonInt(const std::string& line, const std::string& key, int def = 0) {
    std::string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return def;
    size_t start = pos + search.size();
    while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) start++;
    size_t end = line.find_first_of(",}\n\r", start);
    if (end == std::string::npos) end = line.size();
    try {
        return std::stoi(line.substr(start, end - start));
    } catch (...) {
        return def;
    }
}

static std::vector<double> getJsonDoubleVector(const std::string& line, const std::string& key) {
    std::vector<double> result;
    std::string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return result;
    size_t openBracket = line.find("[", pos);
    if (openBracket == std::string::npos) return result;
    size_t closeBracket = line.find("]", openBracket);
    if (closeBracket == std::string::npos) return result;

    std::string arrayContent = line.substr(openBracket + 1, closeBracket - openBracket - 1);
    std::stringstream ss(arrayContent);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            try {
                result.push_back(std::stod(item));
            } catch (...) {}
        }
    }
    return result;
}

int main(int argc, char** argv) {
    std::string corpusFile = "audit/corpus_onedim.jsonl";
    std::string outputFile = "audit/native_onedim_oracle_results.json";

    if (argc > 1) corpusFile = argv[1];
    if (argc > 2) outputFile = argv[2];

    std::ifstream inFile(corpusFile);
    if (!inFile.is_open()) {
        std::cerr << "Error opening corpus file: " << corpusFile << std::endl;
        return 1;
    }

    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }

    outFile << "{\n  \"cases\": [\n";

    std::string line;
    bool firstCase = true;
    size_t caseIndex = 0;

    while (std::getline(inFile, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::string caseId = getJsonString(line, "case_id");
        double rdt = getJsonDouble(line, "rdt", 0.0);
        int evalPoint = getJsonInt(line, "eval_point", -1);
        std::vector<double> state = getJsonDoubleVector(line, "state");
        std::vector<double> step = getJsonDoubleVector(line, "step");
        std::vector<double> rtol = getJsonDoubleVector(line, "rtol");
        std::vector<double> atol = getJsonDoubleVector(line, "atol");
        int queryDomainIndex = getJsonInt(line, "query_domain_index", -1);
        int queryComponentIndex = getJsonInt(line, "query_component_index", -1);

        if (!firstCase) outFile << ",\n";
        firstCase = false;

        outFile << "    {\n";
        outFile << "      \"case_id\": \"" << caseId << "\",\n";
        outFile << "      \"index\": " << caseIndex++ << ",\n";

        try {
            auto oned = std::make_shared<Cantera::OneDim>();

            // Parse domains from JSON line
            size_t domPos = line.find("\"domains\":");
            size_t openB = line.find("[", domPos);
            size_t closeB = line.find("]", openB);
            std::string domStr = line.substr(openB, closeB - openB + 1);

            // Simple domain parser
            size_t curr = 0;
            while ((curr = domStr.find("{", curr)) != std::string::npos) {
                size_t endBrace = domStr.find("}", curr);
                std::string dObj = domStr.substr(curr, endBrace - curr + 1);

                std::string dName = getJsonString(dObj, "name");
                std::string dType = getJsonString(dObj, "type");
                int dNv = getJsonInt(dObj, "n_components", 1);
                int dNp = getJsonInt(dObj, "n_points", 1);
                int dBw = getJsonInt(dObj, "bandwidth", 1);

                auto dom = std::make_shared<TestOneDimDomain>(dName, dType, dNv, dNp, dBw);
                dom->setTolerances(rtol, atol);
                oned->addDomain(dom);
                curr = endBrace + 1;
            }

            oned->init();
            oned->resize();

            // Set initial state
            std::vector<double> x0 = state;
            if (x0.size() < oned->size()) {
                x0.resize(oned->size(), 0.0);
            }
            oned->setInitialGuess(x0);

            // Exceptions tests
            if (queryDomainIndex >= 0) {
                oned->checkDomainIndex(queryDomainIndex);
            }
            if (queryComponentIndex >= 0) {
                oned->component(queryComponentIndex);
            }

            // Normal outputs
            outFile << "      \"success\": true,\n";
            outFile << "      \"exception_thrown\": false,\n";
            outFile << "      \"error_message\": \"\",\n";
            outFile << "      \"n_domains\": " << oned->nDomains() << ",\n";
            outFile << "      \"points\": " << oned->points() << ",\n";
            outFile << "      \"size\": " << oned->size() << ",\n";
            outFile << "      \"bandwidth\": " << oned->bandwidth() << ",\n";

            double wnorm = oned->weightedNorm(step);
            outFile << "      \"weighted_norm\": " << std::setprecision(15) << wnorm << ",\n";

            // Residual evaluation
            std::vector<double> res(oned->size(), 0.0);
            size_t evalJ = (evalPoint >= 0) ? static_cast<size_t>(evalPoint) : Cantera::npos;
            oned->eval(evalJ, x0, res, rdt, 1);

            outFile << "      \"residual\": [";
            for (size_t k = 0; k < res.size(); ++k) {
                if (k > 0) outFile << ", ";
                outFile << std::setprecision(15) << res[k];
            }
            outFile << "],\n";

            // Finite-difference Jacobian evaluation
            oned->evalJacobian(x0);
            auto jsys = oned->linearSolver();
            auto mjac = std::dynamic_pointer_cast<Cantera::MultiJac>(jsys);

            outFile << "      \"jacobian_diag\": [";
            if (mjac) {
                for (size_t k = 0; k < oned->size(); ++k) {
                    if (k > 0) outFile << ", ";
                    outFile << std::setprecision(15) << mjac->value(k, k);
                }
            }
            outFile << "],\n";

            // Component name and labels
            outFile << "      \"first_component_name\": " << jsonEscapeString(oned->componentName(0)) << ",\n";
            outFile << "      \"first_upper_bound\": " << oned->upperBound(0) << ",\n";
            outFile << "      \"first_lower_bound\": " << oned->lowerBound(0) << "\n";

        } catch (const Cantera::CanteraError& e) {
            outFile << "      \"success\": true,\n";
            outFile << "      \"exception_thrown\": true,\n";
            outFile << "      \"error_message\": " << jsonEscapeString(e.what()) << ",\n";
            outFile << "      \"n_domains\": 0,\n";
            outFile << "      \"points\": 0,\n";
            outFile << "      \"size\": 0,\n";
            outFile << "      \"bandwidth\": 0,\n";
            outFile << "      \"weighted_norm\": 0.0,\n";
            outFile << "      \"residual\": [],\n";
            outFile << "      \"jacobian_diag\": [],\n";
            outFile << "      \"first_component_name\": \"\",\n";
            outFile << "      \"first_upper_bound\": 0.0,\n";
            outFile << "      \"first_lower_bound\": 0.0\n";
        } catch (const std::exception& e) {
            outFile << "      \"success\": false,\n";
            outFile << "      \"exception_thrown\": true,\n";
            outFile << "      \"error_message\": " << jsonEscapeString(e.what()) << ",\n";
            outFile << "      \"n_domains\": 0,\n";
            outFile << "      \"points\": 0,\n";
            outFile << "      \"size\": 0,\n";
            outFile << "      \"bandwidth\": 0,\n";
            outFile << "      \"weighted_norm\": 0.0,\n";
            outFile << "      \"residual\": [],\n";
            outFile << "      \"jacobian_diag\": [],\n";
            outFile << "      \"first_component_name\": \"\",\n";
            outFile << "      \"first_upper_bound\": 0.0,\n";
            outFile << "      \"first_lower_bound\": 0.0\n";
        }

        outFile << "    }";
    }

    outFile << "\n  ]\n}\n";
    outFile.close();
    inFile.close();

    std::cout << "Native OneDim Oracle finished processing cases into " << outputFile << std::endl;
    return 0;
}
