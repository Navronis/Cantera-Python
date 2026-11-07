#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>

#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/OneDim.h"

class BenchOneDimDomain : public Cantera::Domain1D
{
public:
    std::string m_domain_type;

    BenchOneDimDomain(const std::string& name, const std::string& type, size_t nv, size_t np, size_t bw)
        : Cantera::Domain1D(nv, np), m_domain_type(type)
    {
        m_id = name;
        setBandwidth(static_cast<int>(bw));
        for (size_t i = 0; i < nv; ++i) {
            setComponentName(i, name + "_var_" + std::to_string(i));
            setBounds(i, -100.0, 10000.0);
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

int main(int argc, char** argv) {
    std::vector<size_t> sizes = {25, 100, 500, 2000};
    size_t nv = 4;
    int repetitions = 100;

    std::ofstream outFile("audit/onedim_native_benchmark_results.json");
    outFile << "{\n  \"benchmarks\": [\n";

    for (size_t s_idx = 0; s_idx < sizes.size(); ++s_idx) {
        size_t total_points = sizes[s_idx];
        size_t bulk_pts = (total_points > 2) ? (total_points - 2) : 1;

        auto inlet = std::make_shared<BenchOneDimDomain>("inlet", "inlet", nv, 1, 1);
        auto flow = std::make_shared<BenchOneDimDomain>("flow", "flow", nv, bulk_pts, 3);
        auto outlet = std::make_shared<BenchOneDimDomain>("outlet", "outlet", nv, 1, 1);

        std::vector<double> rtol = {1e-4, 1e-4, 1e-4, 1e-4};
        std::vector<double> atol = {1e-9, 1e-9, 1e-9, 1e-9};
        inlet->setTolerances(rtol, atol);
        flow->setTolerances(rtol, atol);
        outlet->setTolerances(rtol, atol);

        Cantera::OneDim oned;
        oned.addDomain(inlet);
        oned.addDomain(flow);
        oned.addDomain(outlet);
        oned.init();
        oned.resize();

        size_t sz = oned.size();
        std::vector<double> x(sz, 1.0);
        std::vector<double> step(sz, 0.01);
        std::vector<double> r(sz, 0.0);

        for (size_t i = 0; i < sz; ++i) {
            x[i] = 1.0 + 0.5 * std::sin(static_cast<double>(i) * 0.1);
            step[i] = 0.01 * std::cos(static_cast<double>(i) * 0.1);
        }

        oned.setInitialGuess(x);

        // Warmup
        for (int rep = 0; rep < 10; ++rep) {
            oned.eval(Cantera::npos, x, r, 0.0, 1);
            volatile double wn = oned.weightedNorm(step);
            (void)wn;
        }

        std::vector<double> timings_ms;
        timings_ms.reserve(repetitions);

        for (int rep = 0; rep < repetitions; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            oned.eval(Cantera::npos, x, r, 0.0, 1);
            volatile double wn = oned.weightedNorm(step);
            (void)wn;
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings_ms.push_back(elapsed_ms);
        }

        std::sort(timings_ms.begin(), timings_ms.end());
        double median_ms = timings_ms[timings_ms.size() / 2];
        double min_ms = timings_ms.front();
        double max_ms = timings_ms.back();

        outFile << "    {\n"
                << "      \"points\": " << total_points << ",\n"
                << "      \"size\": " << sz << ",\n"
                << "      \"median_ms\": " << std::setprecision(6) << median_ms << ",\n"
                << "      \"min_ms\": " << min_ms << ",\n"
                << "      \"max_ms\": " << max_ms << "\n"
                << "    }" << (s_idx + 1 < sizes.size() ? ",\n" : "\n");

        std::cout << "Native OneDim N=" << total_points << " (size=" << sz
                  << "): median=" << median_ms << " ms\n";
    }

    outFile << "  ]\n}\n";
    outFile.close();

    return 0;
}
