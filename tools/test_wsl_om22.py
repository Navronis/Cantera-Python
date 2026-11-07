import subprocess

cpp = r"""#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "MMCollisionInt.h"

int main() {
    Cantera::MMCollisionInt integrals;
    integrals.init(0.1, 100.0);
    double dt = (3000.0 - 300.0) / 49.0;
    double eps = 38.0; // Kelvin
    std::cout << "C++ om22 for H2 at 50 points:" << std::endl;
    for (int n = 0; n < 50; ++n) {
        double T = 300.0 + dt * n;
        double tstar = T / eps;
        double om22 = integrals.omega22(tstar, 0.0);
        std::cout << "n=" << n << " T=" << T << " ts=" << tstar << " om22=" << std::setprecision(16) << om22 << std::endl;
    }
    return 0;
}"""

with open("/tmp/test_om22.cpp", "w") as f:
    f.write(cpp)

subprocess.run(["g++", "-std=c++20", "-O2", "/tmp/test_om22.cpp", "/tmp/mm.o",
    "-I/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/src/transport",
    "-I/home/nebula/cantera-pinned-oracle/include",
    "-L/home/nebula/cantera-pinned-oracle/build/lib",
    "-lcantera_shared", "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib",
    "-o", "/tmp/test_om22"], check=True)

res = subprocess.run(["/tmp/test_om22"], capture_output=True, text=True)
print(res.stdout)
