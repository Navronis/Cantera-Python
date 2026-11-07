#include <iostream>
#include "cantera/numerics/CVodesIntegrator.h"
#include "cantera/numerics/FuncEval.h"

using namespace Cantera;

class SimpleDecay : public FuncEval {
public:
    size_t neq() const override { return 1; }
    void eval(double t, span<const double> y, span<double> ydot, span<const double> p) override {
        ydot[0] = -100.0 * y[0];
    }
    void getState(span<double> y) override {
        y[0] = 1.0;
    }
};

int main() {
    CVodesIntegrator ode;
    std::cout << "CVodesIntegrator default max steps: " << ode.maxSteps() << std::endl;
    std::cout << "Linear solver: " << ode.linearSolverType() << std::endl;
    SimpleDecay decay;
    ode.setTolerances(1e-6, 1e-10);
    ode.initialize(0.0, decay);
    ode.integrate(0.05);
    std::cout << "Solution at 0.05: " << ode.solution(0) << std::endl;
    return 0;
}
