/**
 * Native C++ BandMatrix Performance Benchmark.
 * Measures execution times for construction, factor, solve, repeated solve, mult, and rcond.
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>

#include <cantera/numerics/BandMatrix.h>

using namespace Cantera;

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p * v.size())) - 1;
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

double median(std::vector<double>& v) {
    return percentile(v, 0.50);
}

int main(int argc, char** argv) {
    const size_t n = 200;
    const size_t kl = 6;
    const size_t ku = 6;
    const size_t n_solves = 100;
    const size_t n_mults = 100;
    const size_t repetitions = 15;

    // Warm-up
    {
        BandMatrix warm(n, kl, ku);
        for (size_t i = 0; i < n; ++i) {
            size_t j_start = i > kl ? i - kl : 0;
            size_t j_stop = std::min(n, i + ku + 1);
            for (size_t j = j_start; j < j_stop; ++j) {
                warm(i, j) = 1.0 / (abs(static_cast<int>(i) - static_cast<int>(j)) + 1.0);
                if (i == j) warm(i, j) += 10.0;
            }
        }
        warm.factor();
        std::vector<double> rhs(n, 1.0);
        warm.solve(rhs);
    }

    std::vector<double> construction_times;
    std::vector<double> factor_times;
    std::vector<double> one_solve_times;
    std::vector<double> repeated_solve_times;
    std::vector<double> mult_times;
    std::vector<double> rcond_times;
    std::vector<double> total_lifecycle_times;

    for (size_t r = 0; r < repetitions; ++r) {
        auto t_total_start = std::chrono::high_resolution_clock::now();

        // 1. Construction & fill
        auto t0 = std::chrono::high_resolution_clock::now();
        BandMatrix mat(n, kl, ku);
        for (size_t i = 0; i < n; ++i) {
            size_t j_start = i > kl ? i - kl : 0;
            size_t j_stop = std::min(n, i + ku + 1);
            for (size_t j = j_start; j < j_stop; ++j) {
                mat(i, j) = 1.0 / (abs(static_cast<int>(i) - static_cast<int>(j)) + 1.0);
                if (i == j) mat(i, j) += 10.0;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        construction_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        // 2. Mult (100 times)
        std::vector<double> b(n, 1.0);
        std::vector<double> prod(n, 0.0);
        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t m = 0; m < n_mults; ++m) {
            mat.mult(b, prod);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        mult_times.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());

        // 3. Factorization
        auto t4 = std::chrono::high_resolution_clock::now();
        mat.factor();
        auto t5 = std::chrono::high_resolution_clock::now();
        factor_times.push_back(std::chrono::duration<double, std::milli>(t5 - t4).count());

        // 4. One linear solve
        std::vector<double> x = b;
        auto t6 = std::chrono::high_resolution_clock::now();
        mat.solve(x);
        auto t7 = std::chrono::high_resolution_clock::now();
        one_solve_times.push_back(std::chrono::duration<double, std::milli>(t7 - t6).count());

        // 5. Repeated solves with reused LU (100 times)
        auto t8 = std::chrono::high_resolution_clock::now();
        for (size_t s = 0; s < n_solves; ++s) {
            std::vector<double> xs = b;
            mat.solve(xs);
        }
        auto t9 = std::chrono::high_resolution_clock::now();
        repeated_solve_times.push_back(std::chrono::duration<double, std::milli>(t9 - t8).count());

        // 6. rcond
        double anorm = mat.oneNorm();
        auto t10 = std::chrono::high_resolution_clock::now();
        double rc = mat.rcond(anorm);
        auto t11 = std::chrono::high_resolution_clock::now();
        rcond_times.push_back(std::chrono::duration<double, std::milli>(t11 - t10).count());

        auto t_total_end = std::chrono::high_resolution_clock::now();
        total_lifecycle_times.push_back(std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count());
    }

    std::string out_path = (argc > 1) ? argv[1] : "audit/benchmark_native_band_matrix.json";
    std::ofstream out(out_path);
    out << std::setprecision(6) << std::fixed;
    out << "{\n";
    out << "  \"workload\": \"N=" << n << "_kl=" << kl << "_ku=" << ku << "_100_solves_100_mults\",\n";
    out << "  \"construction_median_ms\": " << median(construction_times) << ",\n";
    out << "  \"factor_median_ms\": " << median(factor_times) << ",\n";
    out << "  \"one_solve_median_ms\": " << median(one_solve_times) << ",\n";
    out << "  \"repeated_solve_median_ms\": " << median(repeated_solve_times) << ",\n";
    out << "  \"mult_median_ms\": " << median(mult_times) << ",\n";
    out << "  \"rcond_median_ms\": " << median(rcond_times) << ",\n";
    out << "  \"total_lifecycle_median_ms\": " << median(total_lifecycle_times) << ",\n";
    out << "  \"total_lifecycle_p95_ms\": " << percentile(total_lifecycle_times, 0.95) << "\n";
    out << "}\n";

    std::cout << "Native C++ benchmark finished -> " << out_path << std::endl;
    return 0;
}
