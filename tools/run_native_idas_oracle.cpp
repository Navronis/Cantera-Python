#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <memory>

#include "cantera/base/AnyMap.h"
#include "cantera/numerics/IdasIntegrator.h"
#include "cantera/numerics/FuncEval.h"

using namespace Cantera;

class PolynomialDAE : public FuncEval {
public:
    double m_y0 = 1.0, m_y1 = 1.0;
    double m_yd0 = 0.0, m_yd1 = 0.0;

    size_t neq() const override { return 2; }
    void getStateDae(span<double> y, span<double> ydot) override {
        y[0] = m_y0; y[1] = m_y1;
        ydot[0] = m_yd0; ydot[1] = m_yd1;
    }
    void getConstraints(span<double> constraints) override {
        constraints[0] = 1.0;
        constraints[1] = 0.0;
    }
    void evalDae(double t, span<const double> y, span<const double> ydot,
                 span<const double> p, span<double> r) override {
        r[0] = ydot[0] - (-y[0] + y[1]);
        r[1] = y[0] * y[0] - y[1];
    }
    void setState(double y0, double y1, double yd0, double yd1) {
        m_y0 = y0; m_y1 = y1;
        m_yd0 = yd0; m_yd1 = yd1;
    }
};

class TimeDepDAE : public FuncEval {
public:
    size_t neq() const override { return 2; }
    void getStateDae(span<double> y, span<double> ydot) override {
        y[0] = 1.0; y[1] = 1.0;
        ydot[0] = -1.0; ydot[1] = 0.0;
    }
    void getConstraints(span<double> constraints) override {
        constraints[0] = 1.0;
        constraints[1] = 0.0;
    }
    void evalDae(double t, span<const double> y, span<const double> ydot,
                 span<const double> p, span<double> r) override {
        r[0] = ydot[0] - (-y[0] + std::sin(t));
        r[1] = y[1] - std::cos(t);
    }
};

class RobertsonDAE : public FuncEval {
public:
    size_t neq() const override { return 3; }
    void getStateDae(span<double> y, span<double> ydot) override {
        y[0] = 1.0; y[1] = 0.0; y[2] = 0.0;
        ydot[0] = -0.04; ydot[1] = 0.04; ydot[2] = 0.0;
    }
    void getConstraints(span<double> constraints) override {
        constraints[0] = 1.0;
        constraints[1] = 1.0;
        constraints[2] = 0.0;
    }
    void evalDae(double t, span<const double> y, span<const double> ydot,
                 span<const double> p, span<double> r) override {
        r[0] = ydot[0] - (-0.04 * y[0] + 1e4 * y[1] * y[2]);
        r[1] = ydot[1] - (0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]);
        r[2] = y[0] + y[1] + y[2] - 1.0;
    }
};

class Coupled5DAE : public FuncEval {
public:
    size_t neq() const override { return 5; }
    void getStateDae(span<double> y, span<double> ydot) override {
        y[0] = 1.0; y[1] = 0.5; y[2] = 0.2;
        y[3] = 2.25; y[4] = std::exp(-0.2);
        ydot[0] = 1.75; ydot[1] = std::exp(-0.2) - 0.5; ydot[2] = 1.1;
        ydot[3] = 0.0; ydot[4] = 0.0;
    }
    void getConstraints(span<double> constraints) override {
        constraints[0] = 1.0;
        constraints[1] = 1.0;
        constraints[2] = 1.0;
        constraints[3] = 0.0;
        constraints[4] = 0.0;
    }
    void evalDae(double t, span<const double> y, span<const double> ydot,
                 span<const double> p, span<double> r) override {
        r[0] = ydot[0] - (-0.5 * y[0] + y[3]);
        r[1] = ydot[1] - (-y[1] + y[4]);
        r[2] = ydot[2] - (y[0] + y[1] - 2.0 * y[2]);
        r[3] = y[3] - (y[0] + y[1]) * (y[0] + y[1]);
        r[4] = y[4] - std::exp(-y[2]);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <corpus_path> <output_path> <run_id>" << std::endl;
        return 1;
    }

    std::string corpus_path = argv[1];
    std::string output_path = argv[2];
    std::string run_id = argv[3];

    std::ifstream infile(corpus_path);
    if (!infile.is_open()) {
        std::cerr << "Could not open corpus file: " << corpus_path << std::endl;
        return 1;
    }

    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Could not open output file: " << output_path << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        AnyMap case_def = AnyMap::fromYamlString(line);
        std::string case_id = case_def["case_id"].as<std::string>();
        std::string sys_type = case_def["system_type"].as<std::string>();
        std::string mode = case_def["mode"].as<std::string>();
        double rtol = case_def["rtol"].as<double>();
        double atol = case_def["atol"].as<double>();
        int max_order = case_def.getInt("max_order", 5);
        bool incl_alg = case_def.getBool("include_algebraic_in_error_test", true);
        auto sample_times = case_def["sample_times"].as<std::vector<double>>();

        std::unique_ptr<FuncEval> sys;
        PolynomialDAE* poly_ptr = nullptr;
        if (sys_type == "polynomial" || sys_type == "reinit_poly") {
            auto p = std::make_unique<PolynomialDAE>();
            poly_ptr = p.get();
            sys = std::move(p);
        } else if (sys_type == "timedep") {
            sys = std::make_unique<TimeDepDAE>();
        } else if (sys_type == "robertson") {
            sys = std::make_unique<RobertsonDAE>();
        } else if (sys_type == "coupled5") {
            sys = std::make_unique<Coupled5DAE>();
        } else {
            std::cerr << "Unknown system type: " << sys_type << std::endl;
            continue;
        }

        IdasIntegrator idas;
        idas.setTolerances(rtol, atol);
        idas.setMaxOrder(max_order);
        idas.includeAlgebraicInErrorTest(incl_alg);
        idas.initialize(0.0, *sys);

        std::stringstream json_out;
        json_out << "{\"case_id\": \"" << case_id << "\", \"oracle\": \"native_cantera_726522be\", ";
        json_out << "\"run_id\": \"" << run_id << "\", \"status\": \"PASS\", \"checkpoints\": [";

        bool first_cp = true;
        for (double t_target : sample_times) {
            if (sys_type == "reinit_poly" && std::abs(t_target - 0.15) < 1e-9) {
                // Perform reinitialization at t=0.1
                poly_ptr->setState(0.5, 0.25, 0.0, 0.0);
                idas.reinitialize(0.1, *sys);
            }

            double actual_time = t_target;
            if (mode == "step") {
                actual_time = idas.step(t_target);
            } else {
                idas.integrate(t_target);
                actual_time = idas.currentTime();
            }

            span<double> sol = idas.solution();
            std::vector<double> r_vec(sol.size());
            std::vector<double> ydot_dummy(sol.size(), 0.0);
            std::vector<double> p_dummy;
            sys->evalDae(actual_time, sol, ydot_dummy, p_dummy, r_vec);

            if (!first_cp) json_out << ", ";
            first_cp = false;

            json_out << "{\"time\": " << std::setprecision(12) << actual_time << ", ";
            json_out << "\"y\": [";
            for (size_t i = 0; i < sol.size(); ++i) {
                if (i > 0) json_out << ", ";
                json_out << std::setprecision(12) << sol[i];
            }
            json_out << "], \"algebraic_residual\": " << std::setprecision(12) << r_vec.back() << "}";
        }

        auto stats = idas.solverStats();
        json_out << "], \"metrics\": {";
        json_out << "\"steps\": " << stats["steps"].as<long int>() << ", ";
        json_out << "\"res_evals\": " << stats["res_evals"].as<long int>() << ", ";
        json_out << "\"nonlinear_iters\": " << stats["nonlinear_iters"].as<long int>() << ", ";
        json_out << "\"last_order\": " << stats["last_order"].as<long int>();
        json_out << "}}\n";

        outfile << json_out.str();
    }

    std::cout << "Native IDAS oracle execution completed successfully." << std::endl;
    return 0;
}
