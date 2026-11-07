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
#include "cantera/oneD/Sim1D.h"
#include "cantera/oneD/MultiJac.h"

class BenchSim1DDomain : public Cantera::Domain1D
{
public:
    std::string m_domain_type;

    BenchSim1DDomain(const std::string& name, const std::string& type, size_t nv, size_t np, size_t bw,
                     const std::vector<double>& grid)
        : Cantera::Domain1D(nv, np), m_domain_type(type)
    {
        m_id = name;
        setBandwidth(static_cast<int>(bw));
        for (size_t i = 0; i < nv; ++i) {
            setComponentName(i, name + "_var_" + std::to_string(i));
            setBounds(i, -100.0, 10000.0);
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

class BenchSim1D : public Cantera::Sim1D
{
public:
    using Cantera::Sim1D::Sim1D;
    using Cantera::Sim1D::_setValue;
    using Cantera::Sim1D::_value;
    using Cantera::Sim1D::_workValue;
};

int main(int argc, char** argv) {
    std::vector<size_t> sizes = {25, 100, 500, 2000};
    size_t nv = 4;
    int repetitions = 50;

    std::ofstream outFile("audit/sim1d_native_benchmark_results.json");
    outFile << "{\n  \"benchmarks\": [\n";

    for (size_t s_idx = 0; s_idx < sizes.size(); ++s_idx) {
        size_t total_points = sizes[s_idx];
        size_t bulk_pts = (total_points > 2) ? (total_points - 2) : 1;

        std::vector<double> z_inlet = {0.0};
        std::vector<double> z_bulk(bulk_pts);
        for (size_t j = 0; j < bulk_pts; ++j) {
            z_bulk[j] = static_cast<double>(j) / static_cast<double>(bulk_pts - 1);
        }
        std::vector<double> z_outlet = {1.0};

        std::vector<double> timings_ms;
        timings_ms.reserve(repetitions);

        for (int rep = 0; rep < repetitions; ++rep) {
            std::vector<std::shared_ptr<Cantera::Domain1D>> doms;
            auto inlet = std::make_shared<BenchSim1DDomain>("inlet", "inlet", nv, 1, 1, z_inlet);
            auto flow = std::make_shared<BenchSim1DDomain>("flow", "flow", nv, bulk_pts, 3, z_bulk);
            auto outlet = std::make_shared<BenchSim1DDomain>("outlet", "outlet", nv, 1, 1, z_outlet);
            doms.push_back(inlet);
            doms.push_back(flow);
            doms.push_back(outlet);

            BenchSim1D sim(doms);
            sim.setMaxGridPoints(-1, 100000);
            sim.setRefineCriteria(1, 10.0, 0.4, 0.4, 0.0);

            // Populate steep flame gradient
            for (size_t j = 0; j < bulk_pts; ++j) {
                double zj = z_bulk[j];
                double val = 300.0 + 1700.0 / (1.0 + std::exp(-25.0 * (zj - 0.4)));
                for (size_t c = 0; c < nv; ++c) {
                    sim._setValue(1, c, j, val * (1.0 + 0.1 * c));
                }
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            volatile int ref_res = sim.refine(0);
            (void)ref_res;
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
                << "      \"size\": " << (total_points * nv) << ",\n"
                << "      \"median_ms\": " << std::setprecision(6) << median_ms << ",\n"
                << "      \"min_ms\": " << min_ms << ",\n"
                << "      \"max_ms\": " << max_ms << "\n"
                << "    }" << (s_idx + 1 < sizes.size() ? ",\n" : "\n");

        std::cout << "Native Sim1D N=" << total_points << " (size=" << (total_points * nv)
                  << "): median=" << median_ms << " ms\n";
    }

    outFile << "  ]\n}\n";
    outFile.close();

    return 0;
}
