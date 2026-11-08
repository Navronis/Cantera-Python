#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/PengRobinson.h"
#include "cantera/thermo/RedlichKwongMFTP.h"
#include "cantera/thermo/PureFluidPhase.h"
#include "cantera/thermo/WaterProps.h"
#include "cantera/thermo/WaterPropsIAPWS.h"
#include "cantera/thermo/WaterPropsIAPWSphi.h"
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

    std::ofstream out("audit/native_thermo_family5_oracle.json");
    if (!out.is_open()) {
        std::cerr << "Failed to open audit/native_thermo_family5_oracle.json" << std::endl;
        return 1;
    }

    out << "{\n  \"cases\": [\n";

    std::vector<PhaseTestCase> test_cases = {
        // 1. Peng-Robinson
        {
            "co2_PR_example.yaml", "CO2-PR", "Peng-Robinson",
            {250.0, 300.0, 400.0, 600.0, 1000.0},
            {101325.0, 500000.0, 2000000.0, 5000000.0, 10000000.0},
            {"CO2:1.0", "CO2:0.9, H2:0.1", "CO2:0.5, H2:0.5", "CO2:0.7, H2O:0.1, CH4:0.1, O2:0.1"}
        },
        // 2. Redlich-Kwong
        {
            "co2_RK_example.yaml", "carbondioxide", "Redlich-Kwong",
            {250.0, 300.0, 400.0, 600.0, 1000.0},
            {101325.0, 500000.0, 2000000.0, 5000000.0, 10000000.0},
            {"CO2:1.0", "CO2:0.9, H2:0.1", "CO2:0.5, H2:0.5", "CO2:0.7, H2O:0.1, CH4:0.1, O2:0.1"}
        },
        // 3. PureFluidPhase - water
        {
            "liquidvapor.yaml", "water", "pure-fluid",
            {300.0, 350.0, 400.0, 500.0, 600.0, 700.0},
            {101325.0, 500000.0, 2000000.0, 10000000.0, 30000000.0},
            {""}
        },
        // 4. PureFluidPhase - nitrogen
        {
            "liquidvapor.yaml", "nitrogen", "pure-fluid",
            {100.0, 200.0, 300.0, 500.0},
            {101325.0, 1000000.0, 5000000.0},
            {""}
        },
        // 5. PureFluidPhase - methane
        {
            "liquidvapor.yaml", "methane", "pure-fluid",
            {150.0, 250.0, 300.0, 400.0},
            {101325.0, 1000000.0, 5000000.0},
            {""}
        },
        // 6. PureFluidPhase - carbon-dioxide
        {
            "liquidvapor.yaml", "carbon-dioxide", "pure-fluid",
            {250.0, 300.0, 400.0, 500.0},
            {101325.0, 2000000.0, 8000000.0},
            {""}
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
        auto pure = std::dynamic_pointer_cast<PureFluidPhase>(thermo);

        for (double T : tc.temps) {
            for (double P : tc.pressures) {
                for (const auto& comp : tc.compositions) {
                    try {
                        if (!comp.empty()) {
                            thermo->setState_TPX(T, P, comp);
                        } else {
                            thermo->setState_TP(T, P);
                        }
                    } catch (...) {
                        // Skip unphysical/unsupported single-phase conditions (e.g. within two-phase boundary for pure fluids)
                        continue;
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

                    double kappa = 0.0, alpha = 0.0, sound = 0.0;
                    try { kappa = thermo->isothermalCompressibility(); } catch (...) { kappa = 0.0; }
                    try { alpha = thermo->thermalExpansionCoeff(); } catch (...) { alpha = 0.0; }
                    try { sound = thermo->soundSpeed(); } catch (...) { sound = 0.0; }

                    if (!first_case) {
                        out << ",\n";
                    }
                    first_case = false;

                    out << "    {\n";
                    out << "      \"mech\": \"" << tc.mech << "\",\n";
                    out << "      \"phase_name\": \"" << tc.phase_name << "\",\n";
                    out << "      \"model\": \"" << tc.model << "\",\n";
                    out << "      \"T\": " << std::setprecision(16) << T << ",\n";
                    out << "      \"P\": " << std::setprecision(16) << P << ",\n";
                    out << "      \"composition\": \"" << comp << "\",\n";
                    out << "      \"temperature\": " << std::setprecision(16) << thermo->temperature() << ",\n";
                    out << "      \"pressure\": " << std::setprecision(16) << thermo->pressure() << ",\n";
                    out << "      \"density\": " << std::setprecision(16) << thermo->density() << ",\n";
                    out << "      \"molar_density\": " << std::setprecision(16) << thermo->molarDensity() << ",\n";
                    out << "      \"mean_molecular_weight\": " << std::setprecision(16) << thermo->meanMolecularWeight() << ",\n";
                    out << "      \"enthalpy_mole\": " << std::setprecision(16) << thermo->enthalpy_mole() << ",\n";
                    out << "      \"int_energy_mole\": " << std::setprecision(16) << thermo->intEnergy_mole() << ",\n";
                    out << "      \"entropy_mole\": " << std::setprecision(16) << thermo->entropy_mole() << ",\n";
                    out << "      \"gibbs_mole\": " << std::setprecision(16) << thermo->gibbs_mole() << ",\n";
                    out << "      \"cp_mole\": " << std::setprecision(16) << thermo->cp_mole() << ",\n";
                    out << "      \"cv_mole\": " << std::setprecision(16) << thermo->cv_mole() << ",\n";
                    out << "      \"enthalpy_mass\": " << std::setprecision(16) << thermo->enthalpy_mass() << ",\n";
                    out << "      \"int_energy_mass\": " << std::setprecision(16) << thermo->intEnergy_mass() << ",\n";
                    out << "      \"entropy_mass\": " << std::setprecision(16) << thermo->entropy_mass() << ",\n";
                    out << "      \"gibbs_mass\": " << std::setprecision(16) << thermo->gibbs_mass() << ",\n";
                    out << "      \"cp_mass\": " << std::setprecision(16) << thermo->cp_mass() << ",\n";
                    out << "      \"cv_mass\": " << std::setprecision(16) << thermo->cv_mass() << ",\n";
                    out << "      \"isothermal_compressibility\": " << std::setprecision(16) << kappa << ",\n";
                    out << "      \"thermal_expansion_coeff\": " << std::setprecision(16) << alpha << ",\n";
                    out << "      \"sound_speed\": " << std::setprecision(16) << sound << ",\n";

                    if (pure) {
                        out << "      \"vapor_fraction\": " << std::setprecision(16) << pure->vaporFraction() << ",\n";
                        out << "      \"phase_of_matter\": \"" << pure->phaseOfMatter() << "\",\n";
                    }

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

    // 7. WaterProps and WaterPropsIAPWS standalone evaluations
    std::cout << "Running native oracle for WaterProps & WaterPropsIAPWS" << std::endl;
    WaterProps wp;
    WaterPropsIAPWS iapws;

    std::vector<double> water_temps = {280.0, 300.0, 350.0, 400.0, 500.0, 600.0};
    std::vector<double> water_press = {1e5, 1e6, 5e6, 2e7};

    for (double T : water_temps) {
        for (double P : water_press) {
            double rho = 0.0;
            try {
                rho = wp.density_IAPWS(T, P);
            } catch (...) {
                continue;
            }
            if (rho <= 0.0) continue;

            iapws.setState_TD(T, rho);

            double dens_T = wp.density_T(T, P, 0);
            double rel_eps = wp.relEpsilon(T, P, 0);
            double a_debye = wp.ADebye(T, P, 0);
            double psat = wp.satPressure(T);
            double cte = 0.0, iso_k = 0.0;
            try { cte = wp.coeffThermalExp_IAPWS(T, P); } catch (...) {}
            try { iso_k = wp.isothermalCompressibility_IAPWS(T, P); } catch (...) {}

            double u_mass = iapws.intEnergy_mass();
            double h_mass = iapws.enthalpy_mass();
            double s_mass = iapws.entropy_mass();
            double g_mass = iapws.gibbs_mass();
            double cv_mass = iapws.cv_mass();
            double cp_mass = iapws.cp_mass();
            double p_calc = iapws.pressure();

            if (!first_case) {
                out << ",\n";
            }
            first_case = false;

            out << "    {\n";
            out << "      \"mech\": \"water-props-iapws\",\n";
            out << "      \"phase_name\": \"water\",\n";
            out << "      \"model\": \"water-props\",\n";
            out << "      \"T\": " << std::setprecision(16) << T << ",\n";
            out << "      \"P\": " << std::setprecision(16) << P << ",\n";
            out << "      \"composition\": \"H2O:1.0\",\n";
            out << "      \"density_IAPWS\": " << std::setprecision(16) << rho << ",\n";
            out << "      \"density_T\": " << std::setprecision(16) << dens_T << ",\n";
            out << "      \"rel_epsilon\": " << std::setprecision(16) << rel_eps << ",\n";
            out << "      \"a_debye\": " << std::setprecision(16) << a_debye << ",\n";
            out << "      \"sat_pressure\": " << std::setprecision(16) << psat << ",\n";
            out << "      \"coeff_thermal_exp\": " << std::setprecision(16) << cte << ",\n";
            out << "      \"isothermal_compressibility\": " << std::setprecision(16) << iso_k << ",\n";
            out << "      \"pressure\": " << std::setprecision(16) << p_calc << ",\n";
            out << "      \"int_energy_mass\": " << std::setprecision(16) << u_mass << ",\n";
            out << "      \"enthalpy_mass\": " << std::setprecision(16) << h_mass << ",\n";
            out << "      \"entropy_mass\": " << std::setprecision(16) << s_mass << ",\n";
            out << "      \"gibbs_mass\": " << std::setprecision(16) << g_mass << ",\n";
            out << "      \"cv_mass\": " << std::setprecision(16) << cv_mass << ",\n";
            out << "      \"cp_mass\": " << std::setprecision(16) << cp_mass << "\n";
            out << "    }";
        }
    }

    out << "\n  ]\n}\n";
    out.close();

    std::cout << "Native oracle generation complete. Written to audit/native_thermo_family5_oracle.json" << std::endl;
    return 0;
}
