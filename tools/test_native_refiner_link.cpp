#include <iostream>
#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/refine.h"

class TestDomain : public Cantera::Domain1D
{
public:
    TestDomain(size_t nv, size_t points) : Cantera::Domain1D(nv, points) {}
};

int main() {
    TestDomain dom(2, 5);
    Cantera::Refiner& ref = dom.refiner();
    std::cout << "Refiner ratio: " << ref.maxRatio() << std::endl;
    std::cout << "Refiner maxDelta: " << ref.maxDelta() << std::endl;
    std::cout << "Refiner maxSlope: " << ref.maxSlope() << std::endl;
    std::cout << "Refiner prune: " << ref.prune() << std::endl;
    return 0;
}
