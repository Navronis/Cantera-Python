#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "cantera/oneD/MultiNewton.h"
#include "cantera/oneD/MultiJac.h"
#include "cantera/numerics/SteadyStateSystem.h"
#include "cantera/base/ctexceptions.h"

// Simple JSON parser / formatter helpers for test runner
struct TestCase {
    std::string case_id;
    std::string description;
    std::string mode;
    size_t size = 0;
    size_t bandwidth = 1;
    std::string system_type;
    double diff = 1.0;
    double k = 1.0;
    double A = 1.0;
    double Ta = 1.0;
    double inter_coupling = 0.0;
    int zero_row = -1;
    std::vector<double> s;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    std::vector<double> atol;
    std::vector<double> rtol;
    double rdt = 0.0;
    std::vector<int> mask;
    std::vector<double> x0;
    std::vector<double> step0;
    int max_jac_age = 5;

    // multijac_ops params
    std::vector<std::vector<double>> mat_values;
    std::vector<double> rhs;
};

class TestSteadySystem : public Cantera::SteadyStateSystem
{
public:
    TestCase tc;
    std::vector<double> m_current_x;

    TestSteadySystem(const TestCase& c) : tc(c) {
        m_size = tc.size;
        m_bw = tc.bandwidth;
        m_rdt = tc.rdt;
        m_mask = tc.mask;
        if (m_mask.size() < m_size) {
            m_mask.assign(m_size, 1);
        }

        auto mjac = std::make_shared<Cantera::MultiJac>();
        mjac->initialize(m_size);
        mjac->setBandwidth(m_bw);
        m_jac = mjac;

        m_current_x = tc.x0;
        if (m_current_x.size() < m_size) {
            m_current_x.assign(m_size, 0.0);
        }
    }

    void eval(Cantera::span<const double> x, Cantera::span<double> r,
              double rdt = -1.0, int count = 1) override {
        size_t n = m_size;
        if (tc.system_type == "linear") {
            for (size_t i = 0; i < n; ++i) {
                double xl = (i > 0) ? x[i - 1] : 0.0;
                double xr = (i + 1 < n) ? x[i + 1] : 0.0;
                double d2x = xr - 2.0 * x[i] + xl;
                r[i] = -tc.diff * d2x + tc.k * x[i] - tc.s[i];
            }
        } else if (tc.system_type == "quadratic") {
            for (size_t i = 0; i < n; ++i) {
                double xl = (i > 0) ? x[i - 1] : 0.0;
                double xr = (i + 1 < n) ? x[i + 1] : 0.0;
                double d2x = xr - 2.0 * x[i] + xl;
                r[i] = -tc.diff * d2x + tc.k * x[i] * x[i] - tc.s[i];
            }
        } else if (tc.system_type == "arrhenius") {
            for (size_t i = 0; i < n; ++i) {
                double xl = (i > 0) ? x[i - 1] : 0.0;
                double xr = (i + 1 < n) ? x[i + 1] : 0.0;
                double d2x = xr - 2.0 * x[i] + xl;
                double T = std::max(x[i], 1e-4);
                double exp_t = std::exp(-tc.Ta / T);
                r[i] = -tc.diff * d2x + tc.A * T * exp_t - tc.s[i];
            }
        } else if (tc.system_type == "singular") {
            for (size_t i = 0; i < n; ++i) {
                if (static_cast<int>(i) == tc.zero_row) {
                    r[i] = 0.0;
                } else {
                    double xl = (i > 0) ? x[i - 1] : 0.0;
                    double xr = (i + 1 < n) ? x[i + 1] : 0.0;
                    double d2x = xr - 2.0 * x[i] + xl;
                    r[i] = -tc.diff * d2x + tc.k * x[i] - 1.0;
                }
            }
        } else if (tc.system_type == "coupled3") {
            size_t M = n / 3;
            for (size_t j = 0; j < M; ++j) {
                size_t iu = 3 * j;
                size_t iv = 3 * j + 1;
                size_t it = 3 * j + 2;

                double U = x[iu], V = x[iv], T = x[it];
                double Ul = (j > 0) ? x[3 * (j - 1)] : 0.0;
                double Ur = (j + 1 < M) ? x[3 * (j + 1)] : 0.0;
                double d2U = Ur - 2.0 * U + Ul;
                r[iu] = -tc.diff * d2U + 0.1 * V * U - tc.s[iu];

                double Vl = (j > 0) ? x[3 * (j - 1) + 1] : 0.0;
                double Vr = (j + 1 < M) ? x[3 * (j + 1) + 1] : 0.0;
                double d2V = Vr - 2.0 * V + Vl;
                r[iv] = -tc.diff * d2V + 0.1 * V * V - tc.s[iv];

                double Tl = (j > 0) ? x[3 * (j - 1) + 2] : 300.0;
                double Tr = (j + 1 < M) ? x[3 * (j + 1) + 2] : 1200.0;
                double d2T = Tr - 2.0 * T + Tl;
                r[it] = -tc.diff * d2T + tc.inter_coupling * 0.001 * U * T - tc.s[it];
            }
        }
    }

    void evalJacobian(Cantera::span<const double> x0) override {
        for (size_t i = 0; i < m_size; ++i) {
            m_current_x[i] = x0[i];
        }
        m_jac->reset();
        size_t n = m_size;

        if (tc.system_type == "linear") {
            for (size_t i = 0; i < n; ++i) {
                m_jac->setValue(i, i, 2.0 * tc.diff + tc.k);
                if (i > 0) {
                    m_jac->setValue(i, i - 1, -tc.diff);
                }
                if (i + 1 < n) {
                    m_jac->setValue(i, i + 1, -tc.diff);
                }
            }
        } else if (tc.system_type == "quadratic") {
            for (size_t i = 0; i < n; ++i) {
                m_jac->setValue(i, i, 2.0 * tc.diff + 2.0 * tc.k * x0[i]);
                if (i > 0) {
                    m_jac->setValue(i, i - 1, -tc.diff);
                }
                if (i + 1 < n) {
                    m_jac->setValue(i, i + 1, -tc.diff);
                }
            }
        } else if (tc.system_type == "arrhenius") {
            for (size_t i = 0; i < n; ++i) {
                double T0 = std::max(x0[i], 1e-4);
                double exp_t0 = std::exp(-tc.Ta / T0);
                double d_reac = tc.A * (1.0 + tc.Ta / T0) * exp_t0;
                m_jac->setValue(i, i, 2.0 * tc.diff + d_reac);
                if (i > 0) {
                    m_jac->setValue(i, i - 1, -tc.diff);
                }
                if (i + 1 < n) {
                    m_jac->setValue(i, i + 1, -tc.diff);
                }
            }
        } else if (tc.system_type == "singular") {
            for (size_t i = 0; i < n; ++i) {
                if (static_cast<int>(i) == tc.zero_row) {
                    m_jac->setValue(i, i, 0.0);
                } else {
                    m_jac->setValue(i, i, 2.0 * tc.diff + tc.k);
                    if (i > 0) m_jac->setValue(i, i - 1, -tc.diff);
                    if (i + 1 < n) m_jac->setValue(i, i + 1, -tc.diff);
                }
            }
        } else if (tc.system_type == "coupled3") {
            size_t M = n / 3;
            for (size_t j = 0; j < M; ++j) {
                size_t iu = 3 * j, iv = 3 * j + 1, it = 3 * j + 2;
                double U0 = x0[iu], V0 = x0[iv], T0 = x0[it];

                m_jac->setValue(iu, iu, 2.0 * tc.diff + 0.1 * V0);
                m_jac->setValue(iu, iv, 0.1 * U0);

                m_jac->setValue(iv, iv, 2.0 * tc.diff + 0.2 * V0);

                m_jac->setValue(it, it, 2.0 * tc.diff + tc.inter_coupling * 0.001 * U0);
                m_jac->setValue(it, iu, tc.inter_coupling * 0.001 * T0);

                if (j > 0) {
                    m_jac->setValue(iu, 3 * (j - 1), -tc.diff);
                    m_jac->setValue(iv, 3 * (j - 1) + 1, -tc.diff);
                    m_jac->setValue(it, 3 * (j - 1) + 2, -tc.diff);
                }
                if (j + 1 < M) {
                    m_jac->setValue(iu, 3 * (j + 1), -tc.diff);
                    m_jac->setValue(iv, 3 * (j + 1) + 1, -tc.diff);
                    m_jac->setValue(it, 3 * (j + 1) + 2, -tc.diff);
                }
            }
        }
    }

    double weightedNorm(Cantera::span<const double> step) const override {
        double sum = 0.0;
        for (size_t i = 0; i < m_size; ++i) {
            double ewt = tc.rtol[i] * std::abs(m_current_x[i]) + tc.atol[i];
            double val = step[i] / ewt;
            sum += val * val;
        }
        return std::sqrt(sum / m_size);
    }

    double upperBound(size_t i) const override {
        return tc.upper_bounds[i];
    }

    double lowerBound(size_t i) const override {
        return tc.lower_bounds[i];
    }

    std::string componentName(size_t i) const override {
        return "var_" + std::to_string(i);
    }
};

// Simple JSON extraction helpers
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
    try {
        return std::stoi(line.substr(start, end - start));
    } catch (...) {
        return def;
    }
}

static std::vector<double> getJsonDoubleArray(const std::string& line, const std::string& key) {
    std::vector<double> res;
    std::string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return res;
    size_t start = line.find("[", pos + search.size());
    if (start == std::string::npos) return res;
    size_t end = line.find("]", start);
    if (end == std::string::npos) return res;
    std::string content = line.substr(start + 1, end - start - 1);
    std::stringstream ss(content);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            try {
                res.push_back(std::stod(item));
            } catch (...) {}
        }
    }
    return res;
}

static std::vector<int> getJsonIntArray(const std::string& line, const std::string& key) {
    std::vector<int> res;
    std::string search = "\"" + key + "\":";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return res;
    size_t start = line.find("[", pos + search.size());
    if (start == std::string::npos) return res;
    size_t end = line.find("]", start);
    if (end == std::string::npos) return res;
    std::string content = line.substr(start + 1, end - start - 1);
    std::stringstream ss(content);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            try {
                res.push_back(std::stoi(item));
            } catch (...) {}
        }
    }
    return res;
}

static TestCase parseTestCase(const std::string& line) {
    TestCase c;
    c.case_id = getJsonString(line, "case_id");
    c.description = getJsonString(line, "description");
    c.mode = getJsonString(line, "mode");
    c.size = static_cast<size_t>(getJsonInt(line, "size", 1));
    c.bandwidth = static_cast<size_t>(getJsonInt(line, "bandwidth", 1));
    c.system_type = getJsonString(line, "system_type");
    c.diff = getJsonDouble(line, "diff", 1.0);
    c.k = getJsonDouble(line, "k", 1.0);
    c.A = getJsonDouble(line, "A", 1.0);
    c.Ta = getJsonDouble(line, "Ta", 1.0);
    c.inter_coupling = getJsonDouble(line, "inter_coupling", 0.0);
    c.zero_row = getJsonInt(line, "zero_row", -1);
    c.s = getJsonDoubleArray(line, "s");
    c.lower_bounds = getJsonDoubleArray(line, "lower_bounds");
    c.upper_bounds = getJsonDoubleArray(line, "upper_bounds");
    c.atol = getJsonDoubleArray(line, "atol");
    c.rtol = getJsonDoubleArray(line, "rtol");
    c.rdt = getJsonDouble(line, "rdt", 0.0);
    c.mask = getJsonIntArray(line, "mask");
    c.x0 = getJsonDoubleArray(line, "x0");
    c.step0 = getJsonDoubleArray(line, "step0");
    c.max_jac_age = getJsonInt(line, "max_jac_age", 5);
    c.rhs = getJsonDoubleArray(line, "rhs");

    if (c.s.size() < c.size) c.s.assign(c.size, 1.0);
    if (c.lower_bounds.size() < c.size) c.lower_bounds.assign(c.size, -100.0);
    if (c.upper_bounds.size() < c.size) c.upper_bounds.assign(c.size, 100.0);
    if (c.atol.size() < c.size) c.atol.assign(c.size, 1e-4);
    if (c.rtol.size() < c.size) c.rtol.assign(c.size, 1e-3);
    if (c.x0.size() < c.size) c.x0.assign(c.size, 0.0);
    if (c.mask.size() < c.size) c.mask.assign(c.size, 1);

    return c;
}

int main(int argc, char* argv[]) {
    std::string corpus_file = "audit/corpus_multi_newton.jsonl";
    std::string out_file = "audit/native_multi_newton_oracle_results.json";
    if (argc > 1) corpus_file = argv[1];
    if (argc > 2) out_file = argv[2];

    std::ifstream infile(corpus_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open " << corpus_file << std::endl;
        return 1;
    }

    std::ofstream outfile(out_file);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open " << out_file << std::endl;
        return 1;
    }

    outfile << "{\n  \"results\": [\n";

    std::string line;
    bool first_out = true;
    int count = 0;

    while (std::getline(infile, line)) {
        if (trim(line).empty()) continue;
        TestCase tc = parseTestCase(line);
        count++;

        if (!first_out) outfile << ",\n";
        first_out = false;

        outfile << "    {\n";
        outfile << "      \"case_id\": \"" << tc.case_id << "\",\n";
        outfile << "      \"mode\": \"" << tc.mode << "\",\n";

        TestSteadySystem sys(tc);
        Cantera::MultiNewton newt(tc.size);
        newt.resize(tc.size);
        newt.setOptions(tc.max_jac_age);

        if (tc.mode == "solve") {
            std::vector<double> x0 = tc.x0;
            std::vector<double> x1(tc.size, 0.0);
            int status = -999;
            int iters = 0;

            sys.evalJacobian(x0);
            try {
                sys.factorizeJacobian();
            } catch (Cantera::CanteraError& err) {
                // If steady factorization fails and rdt==0, status is -4
                if (tc.rdt == 0.0) {
                    status = -4;
                }
            }

            if (status != -4) {
                try {
                    status = newt.solve(x0, x1, sys, 0);
                    iters = newt.lastIterations();
                } catch (Cantera::CanteraError& err) {
                    status = -99;
                }
            }

            outfile << "      \"status\": " << status << ",\n";
            outfile << "      \"last_iterations\": " << iters << ",\n";
            outfile << "      \"x1\": [";
            for (size_t i = 0; i < x1.size(); ++i) {
                outfile << std::setprecision(16) << x1[i] << (i + 1 < x1.size() ? ", " : "");
            }
            outfile << "]\n";
        } else if (tc.mode == "step") {
            std::vector<double> x0 = tc.x0;
            std::vector<double> step_vec(tc.size, 0.0);
            int status = 0;

            sys.evalJacobian(x0);
            sys.factorizeJacobian();

            try {
                newt.step(x0, step_vec, sys, 0);
            } catch (Cantera::CanteraError&) {
                status = -1;
            }

            outfile << "      \"status\": " << status << ",\n";
            outfile << "      \"step\": [";
            for (size_t i = 0; i < step_vec.size(); ++i) {
                outfile << std::setprecision(16) << step_vec[i] << (i + 1 < step_vec.size() ? ", " : "");
            }
            outfile << "]\n";
        } else if (tc.mode == "boundStep") {
            std::vector<double> x0 = tc.x0;
            std::vector<double> step0 = tc.step0;
            if (step0.size() < tc.size) step0.assign(tc.size, 0.0);

            double fbound = newt.boundStep(x0, step0, sys, 0);
            outfile << "      \"fbound\": " << std::setprecision(16) << fbound << "\n";
        } else if (tc.mode == "dampStep") {
            std::vector<double> x0 = tc.x0;
            std::vector<double> step0 = tc.step0;
            if (step0.size() < tc.size) step0.assign(tc.size, 0.0);
            std::vector<double> x1(tc.size, 0.0);
            std::vector<double> step1(tc.size, 0.0);
            double s1 = 1e30;

            sys.evalJacobian(x0);
            sys.factorizeJacobian();

            int status = newt.dampStep(x0, step0, x1, step1, s1, sys, 0, false);
            outfile << "      \"status\": " << status << ",\n";
            outfile << "      \"s1\": " << std::setprecision(16) << s1 << ",\n";
            outfile << "      \"x1\": [";
            for (size_t i = 0; i < x1.size(); ++i) {
                outfile << std::setprecision(16) << x1[i] << (i + 1 < x1.size() ? ", " : "");
            }
            outfile << "]\n";
        } else if (tc.mode == "multijac_ops") {
            Cantera::MultiJac jac;
            jac.initialize(tc.size);
            jac.setBandwidth(tc.bandwidth);

            // set elements: [0, 0, 4.0], [0, 1, -1.0], [1, 0, -1.0], [1, 1, 5.0], ...
            jac.setValue(0, 0, 4.0);
            jac.setValue(0, 1, -1.0);
            jac.setValue(1, 0, -1.0);
            jac.setValue(1, 1, 5.0);
            jac.setValue(1, 2, -2.0);
            jac.setValue(2, 1, -2.0);
            jac.setValue(2, 2, 6.0);
            jac.setValue(2, 3, -1.0);
            jac.setValue(3, 2, -1.0);
            jac.setValue(3, 3, 4.0);

            double v11 = jac.value(1, 1);
            double v12 = jac.value(1, 2);

            // updateTransient
            std::vector<int> mask = {1, 1, 1, 0};
            jac.updateTransient(10.0, mask);

            // solve
            std::vector<double> rhs = {2.0, 3.0, 1.0, 5.0};
            std::vector<double> sol(tc.size, 0.0);
            jac.solve(rhs, sol);

            int info_val = jac.info();
            std::string jac_type = jac.type();

            outfile << "      \"type\": \"" << jac_type << "\",\n";
            outfile << "      \"value_1_1\": " << std::setprecision(16) << v11 << ",\n";
            outfile << "      \"value_1_2\": " << std::setprecision(16) << v12 << ",\n";
            outfile << "      \"info\": " << info_val << ",\n";
            outfile << "      \"sol\": [";
            for (size_t i = 0; i < sol.size(); ++i) {
                outfile << std::setprecision(16) << sol[i] << (i + 1 < sol.size() ? ", " : "");
            }
            outfile << "]\n";
        }

        outfile << "    }";
    }

    outfile << "\n  ]\n}\n";
    std::cout << "Successfully ran native MultiNewton oracle across " << count << " cases." << std::endl;
    return 0;
}
