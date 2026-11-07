#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <cantera/core.h>
#include <cantera/transport/MultiTransport.h>

// This internal header lives under src/transport rather than the public
// installed include tree. The probe is built against the pinned source tree.
#include "MMCollisionInt.h"

using namespace Cantera;

class MatrixProbe : public MultiTransport {
public:
    void write(const std::string& path) {
        updateThermal_T();
        update_C();
        m_Lmatrix.resize(3 * m_nsp, 3 * m_nsp, 0.0);
        m_b.assign(3 * m_nsp, 0.0);
        for (size_t k = 0; k < m_nsp; ++k) {
            m_b[k + m_nsp] = m_molefracs[k];
            m_b[k + 2 * m_nsp] = hasInternalModes(k) ? m_molefracs[k] : 0.0;
        }
        eval_L0000(m_molefracs);
        eval_L0010(m_molefracs);
        eval_L0001();
        eval_L1000();
        eval_L1010(m_molefracs);
        eval_L1001(m_molefracs);
        eval_L0100();
        eval_L0110();
        eval_L0101(m_molefracs);

        std::ofstream out(path);
        out << std::setprecision(17) << m_nsp << '\n';
        for (size_t i = 0; i < 3 * m_nsp; ++i) {
            for (size_t j = 0; j < 3 * m_nsp; ++j) {
                out << m_Lmatrix(i, j) << ' ';
            }
            out << '\n';
        }
        for (double value : m_b) {
            out << value << ' ';
        }
        out << '\n';
        m_a = m_b;
        solve(m_Lmatrix, m_a);
        for (double value : m_a) {
            out << value << ' ';
        }
        out << '\n';

        std::ofstream diag(path + ".diag");
        diag << std::setprecision(17);
        auto vector = [&diag](const char* name, const auto& values) {
            diag << name << ' ';
            for (const auto& value : values) {
                diag << value << ' ';
            }
            diag << '\n';
        };
        auto matrix = [&diag, this](const char* name, const auto& values) {
            diag << name << ' ';
            for (size_t i = 0; i < m_nsp; ++i) {
                for (size_t j = 0; j < m_nsp; ++j) {
                    diag << values(i, j) << ' ';
                }
            }
            diag << '\n';
        };
        vector("molefracs", m_molefracs);
        matrix("m_bdiff", m_bdiff);
        matrix("astar", m_astar);
        matrix("bstar", m_bstar);
        matrix("cstar", m_cstar);
        vector("cinternal", m_cinternal);
        vector("rotrelax", m_rotrelax);
        vector("viscosity", m_visc);
        vector("sigma", m_sigma);
        vector("epsilon", m_eps);
        matrix("delta", m_delta);
        MMCollisionInt collision;
        collision.init(0.01, 500.0);
        std::vector<double> tstar(m_nsp);
        std::vector<double> omega22(m_nsp);
        for (size_t k = 0; k < m_nsp; ++k) {
            tstar[k] = Boltzmann * m_thermo->temperature() / m_eps[k];
            omega22[k] = collision.omega22(tstar[k], m_delta(k, k));
        }
        vector("tstar", tstar);
        vector("omega22", omega22);
    }
};

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        return 2;
    }
    const std::string profile = argc > 2 ? argv[2] : "trace-heavy";
    const double temperature = argc > 3 ? std::stod(argv[3]) : 1200.0;
    const double pressure = argc > 4 ? std::stod(argv[4]) : 1e5;
    auto soln = newSolution("gri30.yaml", "", "none");
    auto thermo = soln->thermo();
    std::vector<double> x = {
        0.269205, 0.000107082, 1.36377e-09, 4.35475e-10, 4.34036e-06,
        0.192249, 3.59356e-13, 2.78061e-12, 4.7406e-18, 4.12955e-17,
        2.58549e-14, 8.96502e-16, 6.09056e-11, 7.56752e-09, 0.192253,
        0.0385036, 1.49596e-08, 2.22378e-08, 1.43096e-13, 1.45312e-15,
        1.96948e-12, 8.41937e-19, 3.18852e-13, 7.93625e-18, 3.20653e-15,
        1.15149e-19, 1.61189e-18, 1.4719e-15, 5.24728e-13, 6.90582e-17,
        6.37248e-12, 5.93728e-11, 2.71219e-09, 2.66645e-06, 6.57142e-11,
        9.52453e-08, 1.26006e-14, 3.49802e-12, 1.19232e-11, 7.17782e-13,
        1.85347e-07, 8.25325e-14, 5.00914e-20, 1.54407e-16, 3.07176e-11,
        4.93198e-08, 4.84792e-12, 0.307675, 0.0, 6.21649e-29,
        8.42393e-28, 6.77865e-18, 2.19225e-16
    };
    if (profile == "uniform") {
        x.assign(thermo->nSpecies(), 1.0 / thermo->nSpecies());
    } else if (profile == "harmonic") {
        double sum = 0.0;
        for (size_t k = 0; k < thermo->nSpecies(); ++k) {
            x[k] = 1.0 / (k + 1.0);
            sum += x[k];
        }
        for (double& value : x) {
            value /= sum;
        }
    } else if (profile != "trace-heavy") {
        return 2;
    }
    thermo->setState_TPX(temperature, pressure, x);
    auto probe = std::make_shared<MatrixProbe>();
    probe->init(thermo);
    probe->write(argv[1]);
}
