#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <memory>

#include "cantera/numerics/CVodesIntegrator.h"
#include "cantera/numerics/FuncEval.h"
#include "cantera/base/ctexceptions.h"
#include "cantera/base/AnyMap.h"

using namespace Cantera;

class CorpusFunc : public FuncEval {
public:
    std::string problem_type;
    double lambda = 1.0;
    double omega = 1.0;
    double mu = 10.0;
    double fail_time = 1e9;
    std::vector<double> y_init;

    CorpusFunc(const std::string& type, const std::vector<double>& init_vals)
        : problem_type(type), y_init(init_vals) {
        if (problem_type == "param_exp") {
            m_sens_params = {2.0};
            m_paramScales = {1.0};
        } else if (problem_type == "multi_param_sens") {
            m_sens_params = {-1.0, -2.0};
            m_paramScales = {1.0, 1.0};
        }
    }

    size_t neq() const override {
        return y_init.size();
    }

    void getState(span<double> y) override {
        for (size_t i = 0; i < y_init.size(); ++i) {
            y[i] = y_init[i];
        }
    }

    void eval(double t, span<const double> y, span<double> ydot, span<const double> p) override {
        if (t >= fail_time) {
            throw CanteraError("CorpusFunc::eval", "Intentional RHS failure at t = {}", t);
        }
        if (problem_type == "decay") {
            ydot[0] = -lambda * y[0];
        } else if (problem_type == "param_exp") {
            double param = p.empty() ? m_sens_params[0] : p[0];
            ydot[0] = param * y[0];
        } else if (problem_type == "multi_param_sens") {
            double p0 = p.size() > 0 ? p[0] : m_sens_params[0];
            double p1 = p.size() > 1 ? p[1] : m_sens_params[1];
            ydot[0] = p0 * y[0] + p1 * y[1];
            ydot[1] = -p1 * y[1];
        } else if (problem_type == "coupled_linear") {
            ydot[0] = -1000.0 * y[0] + y[1];
            ydot[1] = -y[1];
        } else if (problem_type == "oscillator") {
            ydot[0] = y[1];
            ydot[1] = -omega * omega * y[0];
        } else if (problem_type == "robertson") {
            ydot[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
            ydot[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
            ydot[2] = 3e7 * y[1] * y[1];
        } else if (problem_type == "vanderpol") {
            ydot[0] = y[1];
            ydot[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        } else {
            ydot[0] = 0.0;
        }
    }
};

static std::string jsonEscape(const std::string& s) {
    std::string res;
    for (char c : s) {
        if (c == '"') res += "\\\"";
        else if (c == '\\') res += "\\\\";
        else if (c == '\n') res += "\\n";
        else if (c == '\r') res += "\\r";
        else if (c == '\t') res += "\\t";
        else res += c;
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: run_native_cvodes_oracle <corpus_jsonl> <output_jsonl> <run_id>" << std::endl;
        return 1;
    }
    std::string corpus_file = argv[1];
    std::string output_file = argv[2];
    std::string run_id = argv[3];

    std::ifstream infile(corpus_file);
    if (!infile.is_open()) {
        std::cerr << "Cannot open corpus file: " << corpus_file << std::endl;
        return 1;
    }

    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Cannot open output file: " << output_file << std::endl;
        return 1;
    }

    std::string line;
    int count = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        AnyMap doc;
        try {
            doc = AnyMap::fromYamlString(line);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse line: " << e.what() << std::endl;
            continue;
        }

        std::string case_id = doc["case_id"].as<std::string>();
        std::string prob_type = doc["problem_type"].as<std::string>();
        std::string mode = doc.getString("mode", "integrate");
        double t0 = doc.getDouble("t0", 0.0);
        double tout = doc.getDouble("tout", 1.0);
        double reltol = doc.getDouble("reltol", 1e-9);
        double abstol = doc.getDouble("abstol", 1e-15);
        int max_steps = doc.getInt("max_steps", 20000);
        int max_order = doc.getInt("max_order", 5);
        double hmax = doc.getDouble("hmax", 0.0);
        double hmin = doc.getDouble("hmin", 0.0);
        int max_err_test_fails = doc.getInt("max_err_test_fails", 0);
        std::string linear_solver = doc.getString("linear_solver", "DENSE");
        int bandwidth_upper = doc.getInt("bandwidth_upper", 0);
        int bandwidth_lower = doc.getInt("bandwidth_lower", 0);

        double lambda_val = doc.getDouble("lambda", 1.0);
        double omega_val = doc.getDouble("omega", 1.0);
        double mu_val = doc.getDouble("mu", 10.0);
        double fail_t = doc.getDouble("fail_time", 1e9);

        std::vector<double> y_init;
        if (doc.hasKey("y_init")) {
            y_init = doc["y_init"].as<std::vector<double>>();
        }
        if (y_init.empty()) {
            y_init = {1.0};
        }

        std::vector<double> abstol_vec;
        if (doc.hasKey("abstol_vec")) {
            abstol_vec = doc["abstol_vec"].as<std::vector<double>>();
        }

        CorpusFunc func(prob_type, y_init);
        func.lambda = lambda_val;
        func.omega = omega_val;
        func.mu = mu_val;
        func.fail_time = fail_t;
        if (doc.hasKey("p_val")) {
            func.m_sens_params = {doc.getDouble("p_val", 2.0)};
        }

        try {
            CVodesIntegrator ode;
            if (!abstol_vec.empty()) {
                ode.setTolerances(reltol, abstol_vec);
            } else {
                ode.setTolerances(reltol, abstol);
            }
            if (max_order > 0) ode.setMaxOrder(max_order);
            if (max_steps > 0) ode.setMaxSteps(max_steps);
            if (hmax > 0) ode.setMaxStepSize(hmax);
            if (hmin > 0) ode.setMinStepSize(hmin);
            if (max_err_test_fails > 0) ode.setMaxErrTestFails(max_err_test_fails);
            if (linear_solver != "DENSE") {
                ode.setLinearSolverType(linear_solver);
                if (linear_solver == "BAND") {
                    ode.setBandwidth(bandwidth_upper, bandwidth_lower);
                }
            }

            ode.initialize(t0, func);

            if (mode == "step") {
                double t_reached = ode.step(tout);
                outfile << std::setprecision(16);
                outfile << "{\"case_id\": \"" << case_id << "\", \"run_id\": \"" << run_id
                        << "\", \"status\": \"PASS\", \"time\": " << t_reached << ", \"y\": [";
                span<double> sol = ode.solution();
                for (size_t i = 0; i < sol.size(); ++i) {
                    outfile << sol[i] << (i + 1 < sol.size() ? ", " : "");
                }
                outfile << "], \"last_order\": " << ode.lastOrder()
                        << ", \"n_evals\": " << ode.nEvals() << "}\n";
            } else if (mode == "reinit") {
                ode.integrate(tout * 0.5);
                std::vector<double> y_reinit;
                if (doc.hasKey("y_reinit")) {
                    y_reinit = doc["y_reinit"].as<std::vector<double>>();
                }
                if (y_reinit.empty()) y_reinit = {2.0};
                func.y_init = y_reinit;
                double t_reinit = doc.getDouble("t_reinit", tout * 0.5);
                ode.reinitialize(t_reinit, func);
                ode.integrate(tout);
                outfile << std::setprecision(16);
                outfile << "{\"case_id\": \"" << case_id << "\", \"run_id\": \"" << run_id
                        << "\", \"status\": \"PASS\", \"time\": " << ode.currentTime() << ", \"y\": [";
                span<double> sol = ode.solution();
                for (size_t i = 0; i < sol.size(); ++i) {
                    outfile << sol[i] << (i + 1 < sol.size() ? ", " : "");
                }
                outfile << "], \"last_order\": " << ode.lastOrder()
                        << ", \"n_evals\": " << ode.nEvals() << "}\n";
            } else if (mode == "sens") {
                ode.integrate(tout);
                outfile << std::setprecision(16);
                outfile << "{\"case_id\": \"" << case_id << "\", \"run_id\": \"" << run_id
                        << "\", \"status\": \"PASS\", \"time\": " << ode.currentTime() << ", \"y\": [";
                span<double> sol = ode.solution();
                for (size_t i = 0; i < sol.size(); ++i) {
                    outfile << sol[i] << (i + 1 < sol.size() ? ", " : "");
                }
                outfile << "], \"sens\": [";
                for (size_t p = 0; p < (size_t)ode.nSensParams(); ++p) {
                    outfile << "[";
                    for (size_t k = 0; k < (size_t)ode.nEquations(); ++k) {
                        outfile << ode.sensitivity(k, p) << (k + 1 < (size_t)ode.nEquations() ? ", " : "");
                    }
                    outfile << "]" << (p + 1 < (size_t)ode.nSensParams() ? ", " : "");
                }
                outfile << "], \"last_order\": " << ode.lastOrder()
                        << ", \"n_evals\": " << ode.nEvals() << "}\n";
            } else {
                // standard integrate
                ode.integrate(tout);
                span<double> dky = ode.derivative(tout, 1);
                outfile << std::setprecision(16);
                outfile << "{\"case_id\": \"" << case_id << "\", \"run_id\": \"" << run_id
                        << "\", \"status\": \"PASS\", \"time\": " << ode.currentTime() << ", \"y\": [";
                span<double> sol = ode.solution();
                for (size_t i = 0; i < sol.size(); ++i) {
                    outfile << sol[i] << (i + 1 < sol.size() ? ", " : "");
                }
                outfile << "], \"ydot\": [";
                for (size_t i = 0; i < dky.size(); ++i) {
                    outfile << dky[i] << (i + 1 < dky.size() ? ", " : "");
                }
                outfile << "], \"last_order\": " << ode.lastOrder()
                        << ", \"n_evals\": " << ode.nEvals()
                        << ", \"linear_solver\": \"" << ode.linearSolverType() << "\""
                        << "}\n";
            }
        } catch (const std::exception& e) {
            outfile << "{\"case_id\": \"" << case_id << "\", \"run_id\": \"" << run_id
                    << "\", \"status\": \"ERROR\", \"error_message\": \""
                    << jsonEscape(e.what()) << "\"}\n";
        }
        count++;
    }

    std::cout << "Native CVodes oracle processed " << count << " cases." << std::endl;
    return 0;
}
