#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/OneDim.h"
#include "cantera/oneD/Sim1D.h"

class TestDomain : public Cantera::Domain1D
{
public:
    TestDomain(const std::string& name, size_t nv, size_t np)
        : Cantera::Domain1D(nv, np)
    {
        m_id = name;
        for (size_t i = 0; i < nv; ++i) {
            setComponentName(i, name + "_" + std::to_string(i));
        }
        std::vector<double> z(np);
        for (size_t j = 0; j < np; ++j) {
            z[j] = (np > 1) ? static_cast<double>(j) / (np - 1) : 0.0;
        }
        setupGrid(z);
    }

    double initialValue(size_t n, size_t j) override {
        return 1.0;
    }
};

class TestSim1D : public Cantera::Sim1D
{
public:
    using Cantera::Sim1D::Sim1D;
    using Cantera::Sim1D::_setValue;
    using Cantera::Sim1D::_value;
    using Cantera::Sim1D::_workValue;
};

int main() {
    std::vector<std::shared_ptr<Cantera::Domain1D>> doms;
    auto d1 = std::make_shared<TestDomain>("inlet", 1, 1);
    auto d2 = std::make_shared<TestDomain>("bulk", 1, 7);
    auto d3 = std::make_shared<TestDomain>("outlet", 1, 1);
    doms.push_back(d1);
    doms.push_back(d2);
    doms.push_back(d3);

    TestSim1D sim(doms);
    sim.setRefineCriteria(1, 10.0, 0.3, 0.3, 0.0);

    // Set a steep gradient in bulk domain
    for (size_t j = 0; j < 7; ++j) {
        double val = 1.0 + 10.0 / (1.0 + std::exp(-20.0 * (d2->z(j) - 0.5)));
        sim._setValue(1, 0, j, val);
    }

    std::cout << "Before refine: points=" << sim.points() << ", size=" << sim.size() << std::endl;
    int ref_res = sim.refine(0);
    std::cout << "After refine: ref_res=" << ref_res << ", points=" << sim.points()
              << ", size=" << sim.size() << std::endl;
    std::cout << "Bulk points: " << d2->nPoints() << std::endl;
    for (size_t j = 0; j < d2->nPoints(); ++j) {
        std::cout << "  pt " << j << ": z=" << d2->z(j) << " val=" << sim._value(1, 0, j) << std::endl;
    }

    return 0;
}
