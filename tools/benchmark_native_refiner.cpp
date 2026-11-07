#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/refine.h"

class TestDomain : public Cantera::Domain1D
{
public:
    TestDomain(size_t nv, size_t points) : Cantera::Domain1D(nv, points) {}
};

int main(int argc, char** argv) {
    std::vector<size_t> sizes = {25, 100, 500, 2000};
    size_t nv = 4;
    int repetitions = 200;

    std::ofstream outFile("audit/refiner_native_benchmark_results.json");
    outFile << "{\n  \"benchmarks\": [\n";

    for (size_t s_idx = 0; s_idx < sizes.size(); ++s_idx) {
        size_t n = sizes[s_idx];
        TestDomain dom(nv, n);
        Cantera::Refiner& ref = dom.refiner();
        ref.setCriteria(10.0, 0.5, 0.5, 0.02);
        ref.setMaxPoints(100000);

        std::vector<double> z(n);
        for (size_t j = 0; j < n; ++j) {
            z[j] = static_cast<double>(j) / static_cast<double>(n - 1);
        }

        std::vector<double> x(n * nv);
        for (size_t j = 0; j < n; ++j) {
            double zj = z[j];
            double T = 300.0 + 1700.0 * 0.5 * (1.0 + std::tanh(20.0 * (zj - 0.4)));
            double u = 0.5 + 2.0 * 0.5 * (1.0 + std::tanh(20.0 * (zj - 0.4)));
            double ych4 = 0.05 * 0.5 * (1.0 - std::tanh(20.0 * (zj - 0.4)));
            double yo2 = 0.20 * 0.5 * (1.0 - std::tanh(20.0 * (zj - 0.4)));
            x[dom.index(0, j)] = T;
            x[dom.index(1, j)] = u;
            x[dom.index(2, j)] = ych4;
            x[dom.index(3, j)] = yo2;
        }

        // Warmup
        for (int r = 0; r < 10; ++r) {
            ref.analyze(n, z, x);
        }

        std::vector<double> timings_ms;
        timings_ms.reserve(repetitions);

        for (int r = 0; r < repetitions; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ref.analyze(n, z, x);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings_ms.push_back(elapsed_ms);
        }

        std::sort(timings_ms.begin(), timings_ms.end());
        double median_ms = timings_ms[timings_ms.size() / 2];
        double min_ms = timings_ms.front();
        double max_ms = timings_ms.back();
        double mean_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) / timings_ms.size();

        if (s_idx > 0) outFile << ",\n";
        outFile << "    {\n";
        outFile << "      \"grid_size\": " << n << ",\n";
        outFile << "      \"n_components\": " << nv << ",\n";
        outFile << "      \"repetitions\": " << repetitions << ",\n";
        outFile << "      \"median_ms\": " << median_ms << ",\n";
        outFile << "      \"mean_ms\": " << mean_ms << ",\n";
        outFile << "      \"min_ms\": " << min_ms << ",\n";
        outFile << "      \"max_ms\": " << max_ms << "\n";
        outFile << "    }";
    }

    outFile << "\n  ]\n}\n";
    outFile.close();
    std::cout << "Native benchmark complete." << std::endl;
    return 0;
}
