import sys; sys.path.insert(0, '.')
import subprocess
import numpy as np
import cantera_python as ct

# Run native C++ to dump intermediate arrays at T=1200K, P=1e5
cpp_source = r"""
#include <iostream>
#include <iomanip>
#include <vector>
#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

using namespace Cantera;

class ExposedMultiTransport : public MultiTransport {
public:
    void dumpInternals() {
        solveLMatrixEquation();
        std::cout << "--- C++ DUMP ---" << std::endl;
        std::cout << std::scientific << std::setprecision(16);
        std::cout << "thermalConductivity: " << thermalConductivity() << std::endl;
        std::cout << "cinternal for first 5:" << std::endl;
        for (size_t k = 0; k < 5; ++k) {
            std::cout << m_thermo->speciesName(k) << ": " << m_cinternal[k] << " hasInternal=" << hasInternalModes(k) << std::endl;
        }
        std::cout << "m_b (first 10 of block 1, 2, 3):" << std::endl;
        for (size_t k = 0; k < 5; ++k) {
            std::cout << "b[0][" << k << "]=" << m_b[k]
                      << " b[1][" << k << "]=" << m_b[k + m_nsp]
                      << " b[2][" << k << "]=" << m_b[k + 2*m_nsp] << std::endl;
        }
        std::cout << "m_a (first 5 of block 1, 2, 3):" << std::endl;
        for (size_t k = 0; k < 5; ++k) {
            std::cout << "a[0][" << k << "]=" << m_a[k]
                      << " a[1][" << k << "]=" << m_a[k + m_nsp]
                      << " a[2][" << k << "]=" << m_a[k + 2*m_nsp] << std::endl;
        }
    }
};

int main() {
    auto soln = newSolution("gri30.yaml", "", "none");
    auto thermo = soln->thermo();
    
    std::vector<double> X0 = {
        0.269205, 0.000107082, 1.36377e-09, 4.35475e-10, 4.34036e-06, 0.192249,
        3.59356e-13, 2.78061e-12, 4.7406e-18, 4.12955e-17, 2.58549e-14, 8.96502e-16,
        6.09056e-11, 7.56752e-09, 0.192253, 0.0385036, 1.49596e-08, 2.22378e-08,
        1.43096e-13, 1.45312e-15, 1.96948e-12, 8.41937e-19, 3.18852e-13, 7.93625e-18,
        3.20653e-15, 1.15149e-19, 1.61189e-18, 1.4719e-15, 5.24728e-13, 6.90582e-17,
        6.37248e-12, 5.93728e-11, 2.71219e-09, 2.66645e-06, 6.57142e-11, 9.52453e-08,
        1.26006e-14, 3.49802e-12, 1.19232e-11, 7.17782e-13, 1.85347e-07, 8.25325e-14,
        5.00914e-20, 1.54407e-16, 3.07176e-11, 4.93198e-08, 4.84792e-12, 0.307675,
        0.0, 6.21649e-29, 8.42393e-28, 6.77865e-18, 2.19225e-16
    };
    thermo->setState_TPX(1500.0, 1e5, X0);
    thermo->setState_TP(1200.0, 1e5);
    
    auto multi = std::make_shared<ExposedMultiTransport>();
    multi->init(thermo);
    multi->dumpInternals();
    return 0;
}
"""

with open("tools/dump_internals.cpp", "w") as f:
    f.write(cpp_source)

cmd = (
    "wsl g++ -std=c++20 tools/dump_internals.cpp "
    "-I/home/nebula/cantera-pinned-oracle/include "
    "-L/home/nebula/cantera-pinned-oracle/build/lib "
    "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib "
    "-lcantera_shared -o tools/dump_internals && wsl ./tools/dump_internals"
)
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

from tests.test_multicomponent_transport import GRI30_PATH
py_soln = ct.Solution(GRI30_PATH)
X0 = [
    0.269205, 0.000107082, 1.36377e-09, 4.35475e-10, 4.34036e-06, 0.192249,
    3.59356e-13, 2.78061e-12, 4.7406e-18, 4.12955e-17, 2.58549e-14, 8.96502e-16,
    6.09056e-11, 7.56752e-09, 0.192253, 0.0385036, 1.49596e-08, 2.22378e-08,
    1.43096e-13, 1.45312e-15, 1.96948e-12, 8.41937e-19, 3.18852e-13, 7.93625e-18,
    3.20653e-15, 1.15149e-19, 1.61189e-18, 1.4719e-15, 5.24728e-13, 6.90582e-17,
    6.37248e-12, 5.93728e-11, 2.71219e-09, 2.66645e-06, 6.57142e-11, 9.52453e-08,
    1.26006e-14, 3.49802e-12, 1.19232e-11, 7.17782e-13, 1.85347e-07, 8.25325e-14,
    5.00914e-20, 1.54407e-16, 3.07176e-11, 4.93198e-08, 4.84792e-12, 0.307675,
    0.0, 6.21649e-29, 8.42393e-28, 6.77865e-18, 2.19225e-16
]
py_soln.TPX = 1500.0, 1e5, X0
py_soln.transport_model = 'multicomponent'
py_soln.TP = 1200.0, 1e5
tr = py_soln.transport
tr.solveLMatrixEquation()
print("--- PYTHON DUMP ---")
print("thermalConductivity:", f"{tr.thermalConductivity():.16e}")
print("cinternal for first 5:")
for k in range(5):
    print(f"{py_soln.species_names[k]}: {tr._cinternal[k]:.16e} hasInternal={tr.hasInternalModes(k)}")
print("m_b (first 5 of block 1, 2, 3):")
nsp = tr._nsp
for k in range(5):
    print(f"b[0][{k}]={tr._b[k]:.16e} b[1][{k}]={tr._b[k + nsp]:.16e} b[2][{k}]={tr._b[k + 2*nsp]:.16e}")
print("m_a (first 5 of block 1, 2, 3):")
for k in range(5):
    print(f"a[0][{k}]={tr._a[k]:.16e} a[1][{k}]={tr._a[k + nsp]:.16e} a[2][{k}]={tr._a[k + 2*nsp]:.16e}")

