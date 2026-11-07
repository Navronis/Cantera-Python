#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "cantera/oneD/MultiNewton.h"
#include "cantera/oneD/MultiJac.h"
#include "cantera/numerics/SteadyStateSystem.h"

class Benchmark1DSystem : public Cantera::SteadyStateSystem
{
public:
    double diff = 0.1;
    double k = 2.0;
    std::vector<double> s;
    std::vector<double> lower_b;
    std::vector<double> upper_b;
    std::vector<double> atol_vec;
    std::vector<double> rtol_vec;
    std::vector<double> m_current_x;

    Benchmark1DSystem(size_t sz) {
        m_size = sz;
        m_bw = 1;
        m_rdt = 0.0;
        m_mask.assign(m_size, 1);
        m_current_x.assign(m_size, 0.5);

        s.resize(m_size);
        for (size_t i = 0; i < m_size; ++i) {
            s[i] = 1.0 + 5.0 * std::sin(M_PI * i / (m_size - 1));
        }
        lower_b.assign(m_size, 0.0);
        upper_b.assign(m_size, 50.0);
        atol_vec.assign(m_size, 1e-4);
        rtol_vec.assign(m_size, 1e-3);

        auto mjac = std::make_shared<Cantera::MultiJac>();
        mjac->initialize(m_size);
        mjac->setBandwidth(1);
        m_jac = mjac;
    }

    void eval(Cantera::span<const double> x, Cantera::span<double> r,
              double rdt = -1.0, int count = 1) override {
        size_t n = m_size;
        for (size_t i = 0; i < n; ++i) {
            double xl = (i > 0) ? x[i - 1] : 0.0;
            double xr = (i + 1 < n) ? x[i + 1] : 0.0;
            double d2x = xr - 2.0 * x[i] + xl;
            r[i] = -diff * d2x + k * x[i] * x[i] - s[i];
        }
    }

    void evalJacobian(Cantera::span<const double> x0) override {
        for (size_t i = 0; i < m_size; ++i) {
            m_current_x[i] = x0[i];
        }
        m_jac->reset();
        size_t n = m_size;
        for (size_t i = 0; i < n; ++i) {
            m_jac->setValue(i, i, 2.0 * diff + 2.0 * k * x0[i]);
            if (i > 0) m_jac->setValue(i, i - 1, -diff);
            if (i + 1 < n) m_jac->setValue(i, i + 1, -diff);
        }
    }

    double weightedNorm(Cantera::span<const double> step) const override {
        double sum = 0.0;
        for (size_t i = 0; i < m_size; ++i) {
            double ewt = rtol_vec[i] * std::abs(m_current_x[i]) + atol_vec[i];
            double val = step[i] / ewt;
            sum += val * val;
        }
        return std::sqrt(sum / m_size);
    }

    double upperBound(size_t i) const override { return upper_b[i]; }
    double lowerBound(size_t i) const override { return lower_b[i]; }
};

int main(int argc, char* argv[]) {
    const size_t N = 50;
    const int WARMUP = 10;
    const int TRIALS = 50;

    std::vector<double> solve_times;
    solve_times.reserve(TRIALS);

    // Warmup
    for (int w = 0; w < WARMUP; ++w) {
        Benchmark1DSystem sys(N);
        Cantera::MultiNewton newt(N);
        newt.resize(N);
        std::vector<double> x0(N, 0.5);
        std::vector<double> x1(N, 0.0);
        sys.evalJacobian(x0);
        sys.factorizeJacobian();
        newt.solve(x0, x1, sys, 0);
    }

    // Benchmark trials
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();

        Benchmark1DSystem sys(N);
        Cantera::MultiNewton newt(N);
        newt.resize(N);
        std::vector<double> x0(N, 0.5);
        std::vector<double> x1(N, 0.0);
        sys.evalJacobian(x0);
        sys.factorizeJacobian();
        newt.solve(x0, x1, sys, 0);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        solve_times.push_back(ms);
    }

    std::sort(solve_times.begin(), solve_times.end());
    double median_ms = solve_times[TRIALS / 2];
    double p95_ms = solve_times[static_cast<size_t>(TRIALS * 0.95)];
    double min_ms = solve_times.front();
    double max_ms = solve_times.back();

    std::string out_file = (argc > 1) ? argv[1] : "audit/native_multi_newton_benchmark.json";
    std::ofstream out(out_file);
    out << "{\n";
    out << "  \"system_size\": " << N << ",\n";
    out << "  \"trials\": " << TRIALS << ",\n";
    out << "  \"median_ms\": " << median_ms << ",\n";
    out << "  \"p95_ms\": " << p95_ms << ",\n";
    out << "  \"min_ms\": " << min_ms << ",\n";
    out << "  \"max_ms\": " << max_ms << "\n";
    out << "}\n";

    std::cout << "Native C++ MultiNewton Benchmark (N=" << N << ", " << TRIALS << " trials):" << std::endl;
    std::cout << "  Median: " << median_ms << " ms" << std::endl;
    std::cout << "  P95   : " << p95_ms << " ms" << std::endl;
    std::cout << "  Min   : " << min_ms << " ms" << std::endl;
    return 0;
}
