#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <cmath>

#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/OneDim.h"
#include "cantera/oneD/Sim1D.h"
#include "cantera/oneD/MultiJac.h"
#include "cantera/base/AnyMap.h"
#include "cantera/base/global.h"

static std::string jsonEscapeString(const std::string& input) {
    std::ostringstream ss;
    for (char c : input) {
        switch (c) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    ss << c;
                }
        }
    }
    return ss.str();
}

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

class TestSim1DDomain : public Cantera::Domain1D
{
public:
    std::string m_domain_type;

    TestSim1DDomain(const std::string& name, const std::string& type, size_t nv, size_t np, size_t bw,
                    const std::vector<double>& grid)
        : Cantera::Domain1D(nv, np), m_domain_type(type)
    {
        m_id = name;
        setBandwidth(static_cast<int>(bw));
        for (size_t i = 0; i < nv; ++i) {
            setComponentName(i, name + "_var_" + std::to_string(i));
            setBounds(i, -100.0 - 5.0 * static_cast<double>(i), 10000.0 + 10.0 * static_cast<double>(i));
        }
        if (!grid.empty()) {
            setupGrid(grid);
        }
    }

    double initialValue(size_t n, size_t j) override {
        return 1.0;
    }

    void eval(size_t j, Cantera::span<const double> x, Cantera::span<double> r,
              Cantera::span<int> mask, double rdt=0.0) override {
        size_t j0 = 0;
        size_t j1 = nPoints() - 1;
        if (j != Cantera::npos) {
            if (j < firstPoint() || j > lastPoint()) {
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

class TestSim1D : public Cantera::Sim1D
{
public:
    using Cantera::Sim1D::Sim1D;
    using Cantera::Sim1D::_setValue;
    using Cantera::Sim1D::_value;
    using Cantera::Sim1D::_workValue;
};

int main(int argc, char** argv) {
    std::string corpusPath = "audit/corpus_sim1d.jsonl";
    std::string outputPath = "audit/native_sim1d_oracle_results.json";

    std::ifstream inFile(corpusPath);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open " << corpusPath << std::endl;
        return 1;
    }

    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << outputPath << std::endl;
        return 1;
    }

    outFile << "{\n  \"cases\": [\n";

    std::string line;
    bool firstCase = true;
    size_t caseIndex = 0;

    while (std::getline(inFile, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (!firstCase) outFile << ",\n";
        firstCase = false;

        outFile << "    {\n";

        try {
            auto caseMap = Cantera::AnyMap::fromYamlString(line);
            std::string caseId = caseMap["case_id"].as<std::string>();

            outFile << "      \"case_id\": \"" << caseId << "\",\n";
            outFile << "      \"index\": " << caseIndex++ << ",\n";

            // Parse domains
            std::vector<std::shared_ptr<Cantera::Domain1D>> doms;
            auto& domains = caseMap["domains"].asVector<Cantera::AnyMap>();
            for (auto& dMap : domains) {
                std::string dName = dMap["name"].as<std::string>();
                std::string dType = dMap.getString("type", "flow");
                size_t dNv = dMap.getInt("n_components", 1);
                size_t dNp = dMap.getInt("n_points", 1);
                size_t dBw = dMap.getInt("bandwidth", 1);
                std::vector<double> dGrid;
                if (dMap.hasKey("grid")) {
                    dGrid = dMap["grid"].asVector<double>();
                }
                auto dom = std::make_shared<TestSim1DDomain>(dName, dType, dNv, dNp, dBw, dGrid);
                doms.push_back(dom);
            }

            TestSim1D sim(doms);

            // Populate initial state by domain
            if (caseMap.hasKey("state_by_domain")) {
                auto sbd = caseMap["state_by_domain"].asVector<std::vector<double>>();
                for (size_t d = 0; d < sbd.size(); ++d) {
                    const auto& vals = sbd[d];
                    size_t nv = sim.domain(d).nComponents();
                    size_t np = sim.domain(d).nPoints();
                    for (size_t k = 0; k < vals.size() && k < nv * np; ++k) {
                        size_t pt = k / nv;
                        size_t c = k % nv;
                        sim._setValue(d, c, pt, vals[k]);
                    }
                }
            }

            // Parse refine_criteria if present
            if (caseMap.hasKey("refine_criteria")) {
                auto& rcs = caseMap["refine_criteria"].asVector<Cantera::AnyMap>();
                for (auto& rc : rcs) {
                    int d = rc.getInt("domain", 0);
                    double ratio = rc.getDouble("ratio", 10.0);
                    double slope = rc.getDouble("slope", 0.8);
                    double curve = rc.getDouble("curve", 0.8);
                    double prune = rc.getDouble("prune", -0.1);
                    sim.setRefineCriteria(d, ratio, slope, curve, prune);
                }
            }

            // Execute operations
            int refineResult = 0;
            double queriedValue = 0.0;
            std::vector<double> queriedCriteria;
            int queriedMaxPoints = 0;
            int queriedTsRegridMax = 0;
            std::vector<double> adjointLambda;
            std::vector<double> jacDiag;

            if (caseMap.hasKey("operations")) {
                auto& ops = caseMap["operations"].asVector<Cantera::AnyMap>();
                for (auto& op : ops) {
                    std::string opName = op["op"].as<std::string>();

                    if (opName == "set_value") {
                        int d = op.getInt("domain", 0);
                        int c = op.getInt("comp", 0);
                        int pt = op.getInt("point", 0);
                        double val = op.getDouble("val", 0.0);
                        sim._setValue(d, c, pt, val);
                    } else if (opName == "get_value") {
                        int d = op.getInt("domain", 0);
                        int c = op.getInt("comp", 0);
                        int pt = op.getInt("point", 0);
                        queriedValue = sim._value(d, c, pt);
                    } else if (opName == "refine") {
                        int llevel = op.getInt("loglevel", 0);
                        refineResult = sim.refine(llevel);
                    } else if (opName == "set_criteria") {
                        int d = op.getInt("domain", -1);
                        double ratio = op.getDouble("ratio", 10.0);
                        double slope = op.getDouble("slope", 0.8);
                        double curve = op.getDouble("curve", 0.8);
                        double prune = op.getDouble("prune", -0.1);
                        sim.setRefineCriteria(d, ratio, slope, curve, prune);
                    } else if (opName == "get_criteria") {
                        int d = op.getInt("domain", 0);
                        queriedCriteria = sim.getRefineCriteria(d);
                    } else if (opName == "set_gridmin") {
                        int d = op.getInt("domain", -1);
                        double gmin = op.getDouble("gridmin", 1e-5);
                        sim.setGridMin(d, gmin);
                    } else if (opName == "set_max_points") {
                        int d = op.getInt("domain", -1);
                        int npts = op.getInt("npoints", 1000);
                        sim.setMaxGridPoints(d, npts);
                    } else if (opName == "get_max_points") {
                        int d = op.getInt("domain", 0);
                        queriedMaxPoints = static_cast<int>(sim.maxGridPoints(d));
                    } else if (opName == "restore_steady") {
                        sim.restoreSteadySolution();
                    } else if (opName == "restore_timestepping") {
                        sim.restoreTimeSteppingSolution();
                    } else if (opName == "get_ts_regrid_max") {
                        queriedTsRegridMax = sim.timeStepRegridMax();
                    } else if (opName == "set_ts_regrid_max") {
                        int count = op.getInt("count", 10);
                        sim.setTimeStepRegridMax(count);
                    } else if (opName == "eval_ss_jacobian") {
                        sim.evalSSJacobian();
                        auto mjac = std::dynamic_pointer_cast<Cantera::MultiJac>(sim.linearSolver());
                        if (mjac) {
                            for (size_t k = 0; k < sim.size(); ++k) {
                                jacDiag.push_back(mjac->value(k, k));
                            }
                        }
                    } else if (opName == "solve_adjoint") {
                        std::vector<double> b = caseMap["adjoint_rhs"].asVector<double>();
                        adjointLambda.resize(sim.size(), 0.0);
                        sim.solveAdjoint(b, adjointLambda);
                    } else if (opName == "set_left_control") {
                        double temp = op.getDouble("temp", 1000.0);
                        sim.setLeftControlPoint(temp);
                    } else if (opName == "set_right_control") {
                        double temp = op.getDouble("temp", 1000.0);
                        sim.setRightControlPoint(temp);
                    }
                }
            }

            // Normal success response
            outFile << "      \"success\": true,\n";
            outFile << "      \"exception_thrown\": false,\n";
            outFile << "      \"error_message\": \"\",\n";
            outFile << "      \"n_domains\": " << sim.nDomains() << ",\n";
            outFile << "      \"points\": " << sim.points() << ",\n";
            outFile << "      \"size\": " << sim.size() << ",\n";
            outFile << "      \"bandwidth\": " << sim.bandwidth() << ",\n";
            outFile << "      \"refine_result\": " << refineResult << ",\n";
            outFile << "      \"queried_value\": " << std::setprecision(15) << queriedValue << ",\n";
            outFile << "      \"queried_max_points\": " << queriedMaxPoints << ",\n";
            outFile << "      \"queried_ts_regrid_max\": " << queriedTsRegridMax << ",\n";

            // Queried criteria
            outFile << "      \"queried_criteria\": [";
            for (size_t k = 0; k < queriedCriteria.size(); ++k) {
                if (k > 0) outFile << ", ";
                outFile << std::setprecision(15) << queriedCriteria[k];
            }
            outFile << "],\n";

            // Grid coordinates per domain
            outFile << "      \"domain_grids\": [";
            for (size_t d = 0; d < sim.nDomains(); ++d) {
                if (d > 0) outFile << ", ";
                outFile << "[";
                auto g = sim.domain(d).grid();
                for (size_t j = 0; j < g.size(); ++j) {
                    if (j > 0) outFile << ", ";
                    outFile << std::setprecision(15) << g[j];
                }
                outFile << "]";
            }
            outFile << "],\n";

            // Domain values
            outFile << "      \"domain_values\": [";
            for (size_t d = 0; d < sim.nDomains(); ++d) {
                if (d > 0) outFile << ", ";
                outFile << "[";
                size_t nv = sim.domain(d).nComponents();
                size_t np = sim.domain(d).nPoints();
                for (size_t j = 0; j < np; ++j) {
                    for (size_t c = 0; c < nv; ++c) {
                        if (j > 0 || c > 0) outFile << ", ";
                        outFile << std::setprecision(15) << sim._value(d, c, j);
                    }
                }
                outFile << "]";
            }
            outFile << "],\n";

            // Jacobian diag
            outFile << "      \"jacobian_diag\": [";
            for (size_t k = 0; k < jacDiag.size(); ++k) {
                if (k > 0) outFile << ", ";
                outFile << std::setprecision(15) << jacDiag[k];
            }
            outFile << "],\n";

            // Adjoint lambda
            outFile << "      \"adjoint_lambda\": [";
            for (size_t k = 0; k < adjointLambda.size(); ++k) {
                if (k > 0) outFile << ", ";
                outFile << std::setprecision(15) << adjointLambda[k];
            }
            outFile << "]\n";

        } catch (const Cantera::CanteraError& e) {
            outFile << "      \"success\": true,\n";
            outFile << "      \"exception_thrown\": true,\n";
            outFile << "      \"error_message\": \"" << jsonEscapeString(e.what()) << "\",\n";
            outFile << "      \"n_domains\": 0,\n";
            outFile << "      \"points\": 0,\n";
            outFile << "      \"size\": 0,\n";
            outFile << "      \"bandwidth\": 0,\n";
            outFile << "      \"refine_result\": 0,\n";
            outFile << "      \"queried_value\": 0.0,\n";
            outFile << "      \"queried_max_points\": 0,\n";
            outFile << "      \"queried_ts_regrid_max\": 0,\n";
            outFile << "      \"queried_criteria\": [],\n";
            outFile << "      \"domain_grids\": [],\n";
            outFile << "      \"domain_values\": [],\n";
            outFile << "      \"jacobian_diag\": [],\n";
            outFile << "      \"adjoint_lambda\": []\n";
        } catch (const std::exception& e) {
            outFile << "      \"success\": false,\n";
            outFile << "      \"exception_thrown\": true,\n";
            outFile << "      \"error_message\": \"" << jsonEscapeString(e.what()) << "\",\n";
            outFile << "      \"n_domains\": 0,\n";
            outFile << "      \"points\": 0,\n";
            outFile << "      \"size\": 0,\n";
            outFile << "      \"bandwidth\": 0,\n";
            outFile << "      \"refine_result\": 0,\n";
            outFile << "      \"queried_value\": 0.0,\n";
            outFile << "      \"queried_max_points\": 0,\n";
            outFile << "      \"queried_ts_regrid_max\": 0,\n";
            outFile << "      \"queried_criteria\": [],\n";
            outFile << "      \"domain_grids\": [],\n";
            outFile << "      \"domain_values\": [],\n";
            outFile << "      \"jacobian_diag\": [],\n";
            outFile << "      \"adjoint_lambda\": []\n";
        }

        outFile << "    }";
    }

    outFile << "\n  ]\n}\n";
    outFile.close();
    inFile.close();

    std::cout << "Native Sim1D Oracle processed " << caseIndex << " cases into " << outputPath << std::endl;
    return 0;
}
