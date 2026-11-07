#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cantera/numerics/BandMatrix.h>

using namespace Cantera;

struct CaseSpec {
    size_t n;
    size_t kl;
    size_t ku;
};

static double coefficient(size_t case_id, size_t i, size_t j,
                          size_t kl, size_t ku) {
    const auto raw = static_cast<int>(((i + 2) * (j + 5) + case_id * 7) % 13);
    double value = static_cast<double>(raw - 6) / 7.0;
    if (i == j) {
        value += 3.0 + static_cast<double>(kl + ku);
    }
    if (case_id == 0 && i == 0 && j == 0 && kl > 0) {
        value = 0.0; // Force the first DGBTRF step to pivot from below.
    }
    if (case_id == 0 && i == 1 && j == 0) {
        value = 1.0;
    }
    return value;
}

static double rhs_value(size_t case_id, size_t i, size_t column) {
    const auto raw = static_cast<int>((i * 7 + column * 3 + case_id) % 11);
    return static_cast<double>(raw - 5) / 3.0;
}

int main(int argc, char** argv) {
    const std::vector<CaseSpec> cases = {
        {6, 1, 1}, {8, 2, 1}, {10, 1, 3}, {16, 3, 2},
        {4, 0, 2}, {5, 2, 0}, {30, 4, 3},
    };
    std::ofstream file;
    std::ostream* output = &std::cout;
    if (argc > 1) {
        file.open(argv[1]);
        if (!file) {
            std::cerr << "Could not open output file: " << argv[1] << '\n';
            return 2;
        }
        output = &file;
    }
    *output << std::setprecision(17);
    for (size_t case_id = 0; case_id < cases.size(); ++case_id) {
        const auto [n, kl, ku] = cases[case_id];
        BandMatrix matrix(n, kl, ku);
        for (size_t i = 0; i < n; ++i) {
            const size_t j_start = i > kl ? i - kl : 0;
            const size_t j_stop = std::min(n, i + ku + 1);
            for (size_t j = j_start; j < j_stop; ++j) {
                matrix(i, j) = coefficient(case_id, i, j, kl, ku);
            }
        }
        std::vector<double> rhs(2 * n);
        for (size_t column = 0; column < 2; ++column) {
            for (size_t i = 0; i < n; ++i) {
                rhs[column * n + i] = rhs_value(case_id, i, column);
            }
        }
        matrix.solve(rhs, 2, n);
        *output << case_id << ' ' << n << ' ' << kl << ' ' << ku;
        for (size_t i = 0; i < n; ++i) {
            for (size_t column = 0; column < 2; ++column) {
                *output << ' ' << rhs[column * n + i];
            }
        }
        *output << '\n';
    }
    return 0;
}
