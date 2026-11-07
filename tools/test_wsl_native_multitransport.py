#!/usr/bin/env python3
"""Test native Cantera C++ MultiTransport in WSL2."""
import subprocess
import os

cpp_code = r"""#include <iostream>
#include <vector>
#include <iomanip>
#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

int main() {
    try {
        auto sol = Cantera::newSolution("gri30.yaml", "", "multicomponent");
        auto thermo = sol->thermo();
        thermo->setState_TPX(1200.0, Cantera::OneAtm, "CH4:0.1, O2:0.2, N2:0.7");
        auto multi = std::dynamic_pointer_cast<Cantera::MultiTransport>(sol->transport());

        size_t nsp = thermo->nSpecies();
        std::cout << "Species: " << nsp << std::endl;
        std::cout << "Viscosity: " << std::setprecision(16) << multi->viscosity() << std::endl;
        std::cout << "ThermalCond: " << std::setprecision(16) << multi->thermalConductivity() << std::endl;

        std::vector<double> dt(nsp);
        multi->getThermalDiffCoeffs(dt);
        std::cout << "ThermalDiff[0] (H2): " << std::setprecision(16) << dt[0] << std::endl;

        std::vector<double> d(nsp * nsp);
        multi->getMultiDiffCoeffs(nsp, d);
        std::cout << "MultiDiff[0,1]: " << std::setprecision(16) << d[1] << std::endl;

        // Test species fluxes
        std::vector<double> grad_T = {100.0};
        std::vector<double> grad_X(nsp, 0.0);
        grad_X[0] = 0.1;
        grad_X[1] = -0.1;
        std::vector<double> fluxes(nsp, 0.0);
        multi->getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes);
        std::cout << "Flux[0]: " << std::setprecision(16) << fluxes[0] << std::endl;
        double sum_flux = 0.0;
        for (double f : fluxes) sum_flux += f;
        std::cout << "Sum fluxes: " << sum_flux << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
"""

with open("/tmp/test_native_multi.cpp", "w") as f:
    f.write(cpp_code)

res = subprocess.run([
    "g++", "-std=c++20", "-O2", "/tmp/test_native_multi.cpp",
    "-I/home/nebula/cantera-pinned-oracle/include",
    "-L/home/nebula/cantera-pinned-oracle/build/lib",
    "-lcantera_shared",
    "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib",
    "-o", "/tmp/test_native_multi"
], capture_output=True, text=True)

if res.returncode != 0:
    print("Compilation failed:")
    print(res.stderr)
else:
    print("Compilation succeeded. Running:")
    run_res = subprocess.run(["/tmp/test_native_multi"], capture_output=True, text=True)
    print(run_res.stdout)
    if run_res.stderr:
        print("Stderr:", run_res.stderr)
