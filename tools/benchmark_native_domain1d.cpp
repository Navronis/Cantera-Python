/**
 * Pinned native Cantera Domain1D performance benchmark.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/refine.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

class BenchDomain1D : public Domain1D
{
public:
    BenchDomain1D(size_t nv, size_t points) : Domain1D(nv, points) {}
    void eval(size_t j, span<const double> x, span<double> r, span<int> mask, double rdt=0.0) override {}
    double initialValue(size_t n, size_t j) override { return 1.0; }
};

int main()
{
    vector<size_t> grid_sizes = {25, 100, 500, 2000};
    size_t nv = 5;
    int reps = 1000;

    cout << "GRID_SIZE,REPS,MEDIAN_US,MEAN_US" << endl;

    for (size_t np : grid_sizes) {
        vector<double> times;
        times.reserve(reps);

        // Warmup
        for (int r = 0; r < 50; r++) {
            BenchDomain1D d(nv, np);
            d.setupUniformGrid(np, 0.1, 0.0);
            d.setSteadyTolerances(1e-4, 1e-9, -1);
            d.setTransientTolerances(1e-4, 1e-11, -1);
            d.setBounds(0, 0.0, 100.0);
            d.locate();
        }

        for (int r = 0; r < reps; r++) {
            auto t0 = high_resolution_clock::now();
            BenchDomain1D d(nv, np);
            d.setupUniformGrid(np, 0.1, 0.0);
            d.setSteadyTolerances(1e-4, 1e-9, -1);
            d.setTransientTolerances(1e-4, 1e-11, -1);
            d.setBounds(0, 0.0, 100.0);
            d.locate();
            auto t1 = high_resolution_clock::now();
            double dur = duration<double, micro>(t1 - t0).count();
            times.push_back(dur);
        }

        sort(times.begin(), times.end());
        double median = times[times.size() / 2];
        double mean = accumulate(times.begin(), times.end(), 0.0) / times.size();

        cout << np << "," << reps << "," << fixed << setprecision(3) << median << "," << mean << endl;
    }

    return 0;
}
