#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/thermo/EdgePhase.h"
#include "cantera/thermo/CoverageDependentSurfPhase.h"
#include "cantera/thermo/IdealSolidSolnPhase.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/base/global.h"

using namespace Cantera;

void dump_vector(std::ostream& os, const std::string& name, const std::vector<double>& v, bool last = false) {
    os << "      \"" << name << "\": [";
    for (size_t i = 0; i < v.size(); ++i) {
        os << std::setprecision(16) << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]" << (last ? "\n" : ",\n");
}

struct PhaseTestCase {
    std::string mech;
    std::string phase_name;
    std::string model;
    std::vector<double> temps;
    std::vector<double> pressures;
    std::vector<std::string> compositions;
};

int main(int argc, char** argv) {
    addDataDirectory("/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/test/data");
    addDataDirectory("/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/data");

    std::ofstream out("audit/native_thermo_family4_oracle.json");
    if (!out.is_open()) {
        std::cerr << "Failed to open audit/native_thermo_family4_oracle.json" << std::endl;
        return 1;
    }

    out << "{\n  \"cases\": [\n";

    std::vector<PhaseTestCase> test_cases = {
        // 1. IdealGasPhase
        {
            "gri30.yaml", "", "ideal-gas",
            {300.0, 500.0, 1000.0, 1500.0, 2200.0},
            {101325.0, 506625.0, 2026500.0},
            {"CH4:1, O2:2, N2:7.52", "CH4:1.5, O2:2, N2:7.52"}
        },
        // 2. StoichSubstance / SingleSpeciesTP
        {
            "thermo-models.yaml", "NaCl(s)", "fixed-stoichiometry",
            {300.0, 500.0, 800.0, 1000.0},
            {101325.0, 1000000.0, 5000000.0},
            {""}
        },
        // 3. IdealSolidSolnPhase
        {
            "thermo-models.yaml", "IdealSolidSolnPhase", "ideal-condensed",
            {300.0, 500.0, 800.0, 1200.0},
            {101325.0, 1000000.0, 10000000.0},
            {"sp1:1.0", "sp1:0.5, sp2:0.3, sp3:0.2", "sp2:1.0"}
        },
        // 4. SurfPhase
        {
            "surface-phases.yaml", "Pt-surf", "ideal-surface",
            {300.0, 600.0, 900.0, 1200.0},
            {101325.0, 500000.0},
            {"Pt(s):1.0", "Pt(s):0.8, H(s):0.2", "Pt(s):0.5, H(s):0.3, O(s):0.2"}
        },
        // 5. EdgePhase
        {
            "surface-phases.yaml", "TPB", "edge",
            {300.0, 600.0, 1000.0},
            {101325.0, 500000.0},
            {"(tpb):1.0"}
        },
        // 6. MetalPhase
        {
            "thermo-models.yaml", "Metal", "electron-cloud",
            {300.0, 600.0, 1000.0},
            {101325.0, 1000000.0, 5000000.0},
            {""}
        },
        // 7. CoverageDependentSurfPhase
        {
            "copt_covdepsurf_example.yaml", "covdep", "coverage-dependent-surface",
            {300.0, 500.0, 700.0, 1000.0},
            {101325.0, 500000.0},
            {"Pt:1.0", "Pt:0.6, OC_Pt:0.3, O_Pt:0.1", "Pt:0.3, OC_Pt:0.5, C_Pt:0.2"}
        }
    };

    bool first_case = true;

    for (const auto& tc : test_cases) {
        std::cout << "Running native oracle for: " << tc.mech << " (" << tc.phase_name << ")" << std::endl;
        std::shared_ptr<ThermoPhase> thermo;
        try {
            thermo = newThermo(tc.mech, tc.phase_name);
        } catch (std::exception& e) {
            std::cerr << "Failed to load phase " << tc.mech << ":" << tc.phase_name << ": " << e.what() << std::endl;
            return 1;
        }

        size_t nsp = thermo->nSpecies();
        auto surf = std::dynamic_pointer_cast<SurfPhase>(thermo);

        for (double T : tc.temps) {
            for (double P : tc.pressures) {
                for (const auto& comp : tc.compositions) {
                    if (!comp.empty()) {
                        thermo->setState_TPX(T, P, comp);
                    } else {
                        thermo->setState_TP(T, P);
                    }

                    std::vector<double> mu(nsp), hbar(nsp), sbar(nsp), ubar(nsp), cpbar(nsp), vbar(nsp);
                    std::vector<double> mu0(nsp), h_rt(nsp), s_r(nsp), g_rt(nsp), u_rt(nsp), cp_r(nsp), v_std(nsp);
                    std::vector<double> act(nsp), act_coeff(nsp);

                    try { thermo->getChemPotentials(mu); } catch (...) { mu.clear(); }
                    try { thermo->getPartialMolarEnthalpies(hbar); } catch (...) { hbar.clear(); }
                    try { thermo->getPartialMolarEntropies(sbar); } catch (...) { sbar.clear(); }
                    try { thermo->getPartialMolarIntEnergies(ubar); } catch (...) { ubar.clear(); }
                    try { thermo->getPartialMolarCp(cpbar); } catch (...) { cpbar.clear(); }
                    try { thermo->getPartialMolarVolumes(vbar); } catch (...) { vbar.clear(); }

                    try { thermo->getStandardChemPotentials(mu0); } catch (...) { mu0.clear(); }
                    try { thermo->getEnthalpy_RT(h_rt); } catch (...) { h_rt.clear(); }
                    try { thermo->getEntropy_R(s_r); } catch (...) { s_r.clear(); }
                    try { thermo->getGibbs_RT(g_rt); } catch (...) { g_rt.clear(); }
                    try { thermo->getIntEnergy_RT(u_rt); } catch (...) { u_rt.clear(); }
                    try { thermo->getCp_R(cp_r); } catch (...) { cp_r.clear(); }
                    try { thermo->getStandardVolumes(v_std); } catch (...) { v_std.clear(); }

                    try { thermo->getActivities(act); } catch (...) { act.clear(); }
                    try { thermo->getActivityCoefficients(act_coeff); } catch (...) { act_coeff.clear(); }

                    std::vector<double> cov;
                    double site_density = 0.0;
                    if (surf) {
                        cov.resize(nsp);
                        try { surf->getCoverages(cov); } catch (...) { cov.clear(); }
                        try { site_density = surf->siteDensity(); } catch (...) { site_density = 0.0; }
                    }

                    std::vector<double> X(nsp);
                    try { thermo->getMoleFractions(X); } catch (...) { X.clear(); }

                    double beta = 0.0, alpha = 0.0;
                    try { beta = thermo->isothermalCompressibility(); } catch (...) { beta = -1e99; }
                    try { alpha = thermo->thermalExpansionCoeff(); } catch (...) { alpha = -1e99; }

                    if (!first_case) out << ",\n";
                    first_case = false;

                    out << "    {\n";
                    out << "      \"mechanism\": \"" << tc.mech << "\",\n";
                    out << "      \"phase_name\": \"" << tc.phase_name << "\",\n";
                    out << "      \"model\": \"" << tc.model << "\",\n";
                    out << "      \"composition\": \"" << comp << "\",\n";
                    out << std::setprecision(16);
                    out << "      \"T\": " << T << ",\n";
                    out << "      \"P\": " << P << ",\n";
                    out << "      \"density\": " << thermo->density() << ",\n";
                    out << "      \"molar_density\": " << thermo->molarDensity() << ",\n";
                    out << "      \"molar_volume\": " << thermo->molarVolume() << ",\n";
                    out << "      \"mean_molecular_weight\": " << thermo->meanMolecularWeight() << ",\n";
                    out << "      \"enthalpy_mole\": " << thermo->enthalpy_mole() << ",\n";
                    out << "      \"int_energy_mole\": " << thermo->intEnergy_mole() << ",\n";
                    out << "      \"entropy_mole\": " << thermo->entropy_mole() << ",\n";
                    out << "      \"gibbs_mole\": " << thermo->gibbs_mole() << ",\n";
                    out << "      \"cp_mole\": " << thermo->cp_mole() << ",\n";
                    out << "      \"cv_mole\": " << thermo->cv_mole() << ",\n";
                    out << "      \"enthalpy_mass\": " << thermo->enthalpy_mass() << ",\n";
                    out << "      \"int_energy_mass\": " << thermo->intEnergy_mass() << ",\n";
                    out << "      \"entropy_mass\": " << thermo->entropy_mass() << ",\n";
                    out << "      \"gibbs_mass\": " << thermo->gibbs_mass() << ",\n";
                    out << "      \"cp_mass\": " << thermo->cp_mass() << ",\n";
                    out << "      \"cv_mass\": " << thermo->cv_mass() << ",\n";
                    out << "      \"isothermal_compressibility\": " << beta << ",\n";
                    out << "      \"thermal_expansion_coeff\": " << alpha << ",\n";
                    out << "      \"site_density\": " << site_density << ",\n";
                    dump_vector(out, "X", X);
                    dump_vector(out, "coverages", cov);
                    dump_vector(out, "chem_potentials", mu);
                    dump_vector(out, "partial_molar_enthalpies", hbar);
                    dump_vector(out, "partial_molar_entropies", sbar);
                    dump_vector(out, "partial_molar_int_energies", ubar);
                    dump_vector(out, "partial_molar_cp", cpbar);
                    dump_vector(out, "partial_molar_volumes", vbar);
                    dump_vector(out, "standard_chem_potentials", mu0);
                    dump_vector(out, "enthalpy_RT", h_rt);
                    dump_vector(out, "entropy_R", s_r);
                    dump_vector(out, "gibbs_RT", g_rt);
                    dump_vector(out, "int_energy_RT", u_rt);
                    dump_vector(out, "cp_R", cp_r);
                    dump_vector(out, "standard_volumes", v_std);
                    dump_vector(out, "activities", act);
                    dump_vector(out, "activity_coefficients", act_coeff, true);
                    out << "    }";
                }
            }
        }
    }

    out << "\n  ]\n}\n";
    out.close();
    std::cout << "Native oracle generation complete -> audit/native_thermo_family4_oracle.json" << std::endl;
    return 0;
}
