#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/refine.h"
#include "cantera/base/ctexceptions.h"

class TestDomain : public Cantera::Domain1D
{
public:
    TestDomain(size_t nv, size_t points) : Cantera::Domain1D(nv, points) {}
    void resize(size_t nv, size_t np) {
        Cantera::Domain1D::resize(nv, np);
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

static std::vector<bool> getJsonBoolVector(const std::string& line, const std::string& key) {
    std::vector<bool> result;
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
        if (item == "true" || item == "True" || item == "1") {
            result.push_back(true);
        } else if (item == "false" || item == "False" || item == "0") {
            result.push_back(false);
        }
    }
    return result;
}

int main(int argc, char** argv) {
    std::string corpusFile = "audit/corpus_refiner.jsonl";
    std::string outputFile = "audit/native_refiner_oracle_results.json";

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
        int nPoints = getJsonInt(line, "n_points", 0);
        int nComponents = getJsonInt(line, "n_components", 1);
        int nCall = getJsonInt(line, "n_call", nPoints);
        std::vector<double> z = getJsonDoubleVector(line, "z");
        std::vector<double> x = getJsonDoubleVector(line, "x");
        std::vector<double> criteria = getJsonDoubleVector(line, "criteria");
        std::vector<bool> active = getJsonBoolVector(line, "active");
        double gridmin = getJsonDouble(line, "gridmin", 1e-10);
        int npmax = getJsonInt(line, "npmax", 1000);

        if (!firstCase) {
            outFile << ",\n";
        }
        firstCase = false;

        outFile << "    {\n";
        outFile << "      \"case_id\": \"" << caseId << "\",\n";
        outFile << "      \"index\": " << caseIndex++ << ",\n";

        try {
            TestDomain dom(nComponents, nPoints);
            Cantera::Refiner& ref = dom.refiner();

            if (criteria.size() >= 4) {
                ref.setCriteria(criteria[0], criteria[1], criteria[2], criteria[3]);
            }
            ref.setGridMin(gridmin);
            ref.setMaxPoints(npmax);

            for (size_t c = 0; c < active.size(); ++c) {
                ref.setActive(static_cast<int>(c), active[c]);
            }

            int nNew = ref.analyze(nCall, z, x);

            outFile << "      \"success\": true,\n";
            outFile << "      \"exception_thrown\": false,\n";
            outFile << "      \"error_message\": \"\",\n";
            outFile << "      \"n_new_points\": " << nNew << ",\n";
            outFile << "      \"max_ratio\": " << ref.maxRatio() << ",\n";
            outFile << "      \"max_delta\": " << ref.maxDelta() << ",\n";
            outFile << "      \"max_slope\": " << ref.maxSlope() << ",\n";
            outFile << "      \"prune\": " << ref.prune() << ",\n";

            auto critOut = ref.getCriteria();
            outFile << "      \"criteria\": [" << critOut[0] << ", " << critOut[1] << ", "
                    << critOut[2] << ", " << critOut[3] << "],\n";

            // List of insertion points
            outFile << "      \"insert_points\": [";
            bool firstIns = true;
            for (int j = 0; j < nCall; ++j) {
                if (ref.newPointNeeded(j)) {
                    if (!firstIns) outFile << ", ";
                    firstIns = false;
                    outFile << j;
                }
            }
            outFile << "],\n";

            // Status of keepPoint for each grid point
            outFile << "      \"keep_points\": [";
            for (int j = 0; j < nCall; ++j) {
                if (j > 0) outFile << ", ";
                outFile << (ref.keepPoint(j) ? "true" : "false");
            }
            outFile << "]\n";

        } catch (const Cantera::CanteraError& e) {
            outFile << "      \"success\": true,\n";
            outFile << "      \"exception_thrown\": true,\n";
            outFile << "      \"error_message\": " << jsonEscapeString(e.what()) << ",\n";
            outFile << "      \"n_new_points\": -1,\n";
            outFile << "      \"insert_points\": [],\n";
            outFile << "      \"keep_points\": []\n";
        } catch (const std::exception& e) {
            outFile << "      \"success\": false,\n";
            outFile << "      \"exception_thrown\": true,\n";
            outFile << "      \"error_message\": " << jsonEscapeString(e.what()) << ",\n";
            outFile << "      \"n_new_points\": -1,\n";
            outFile << "      \"insert_points\": [],\n";
            outFile << "      \"keep_points\": []\n";
        }

        outFile << "    }";
    }

    outFile << "\n  ]\n}\n";
    outFile.close();
    inFile.close();

    std::cout << "Native Refiner Oracle finished processing cases into " << outputFile << std::endl;
    return 0;
}
