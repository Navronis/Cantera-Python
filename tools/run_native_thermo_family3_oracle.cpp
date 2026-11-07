#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <span>
#include "cantera/base/Solution.h"
#include "cantera/base/global.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/ThermoFactory.h"

using namespace Cantera;

void dump_vector(std::ostream& os, const std::string& name, const std::vector<double>& v, bool last = false) {
    os << "      \"" << name << "\": [";
    for (size_t i = 0; i < v.size(); ++i) {
        os << std::setprecision(16) << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]" << (last ? "\n" : ",\n");
}

int main(int argc, char** argv) {
    std::string data_dir = "/home/nebula/cantera-pinned-oracle/data";
    if (argc > 1) {
        data_dir = argv[1];
    }
    addDataDirectory(data_dir);

    std::ofstream out("audit/native_thermo_family3_oracle.json");
    if (!out.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    out << "{\n  \"cases\": [\n";

    std::vector<std::string> mechs = {"h2o2.yaml", "gri30.yaml"};
    bool first_case = true;

    for (const auto& mech : mechs) {
        auto sol = newSolution(mech, "", "None");
        auto thermo = sol->thermo();
        size_t nsp = thermo->nSpecies();

        std::vector<double> temps = {300.0, 500.0, 1000.0, 1500.0, 2200.0};
        std::vector<double> pressures = {101325.0, 506625.0, 2026500.0};
        std::vector<std::string> comps;
        if (mech == "h2o2.yaml") {
            comps = {"H2:2, O2:1, AR:4", "H2:1, O2:2, AR:1", "H2O:0.8, H2:0.1, O2:0.1"};
        } else {
            comps = {"CH4:1, O2:2, N2:7.52", "CH4:1.5, O2:2, N2:7.52", "CO:1, H2:1, O2:1, N2:3.76"};
        }

        for (double T : temps) {
            for (double P : pressures) {
                for (const auto& comp : comps) {
                    thermo->setState_TPX(T, P, comp);

                    std::vector<double> mu(nsp), hbar(nsp), sbar(nsp), ubar(nsp), cpbar(nsp), vbar(nsp);
                    std::vector<double> mu0(nsp), h_rt(nsp), s_r(nsp), g_rt(nsp), u_rt(nsp), cp_r(nsp), v_std(nsp);
                    std::vector<double> h_rt_ref(nsp), s_r_ref(nsp), g_rt_ref(nsp), u_rt_ref(nsp), cp_r_ref(nsp), v_ref(nsp);
                    std::vector<double> act(nsp), act_coeff(nsp), ln_act_coeff(nsp);

                    thermo->getChemPotentials(mu);
                    thermo->getPartialMolarEnthalpies(hbar);
                    thermo->getPartialMolarEntropies(sbar);
                    thermo->getPartialMolarIntEnergies(ubar);
                    thermo->getPartialMolarCp(cpbar);
                    thermo->getPartialMolarVolumes(vbar);

                    thermo->getStandardChemPotentials(mu0);
                    thermo->getEnthalpy_RT(h_rt);
                    thermo->getEntropy_R(s_r);
                    thermo->getGibbs_RT(g_rt);
                    thermo->getIntEnergy_RT(u_rt);
                    thermo->getCp_R(cp_r);
                    thermo->getStandardVolumes(v_std);

                    thermo->getEnthalpy_RT_ref(h_rt_ref);
                    thermo->getEntropy_R_ref(s_r_ref);
                    thermo->getGibbs_RT_ref(g_rt_ref);
                    thermo->getIntEnergy_RT_ref(u_rt_ref);
                    thermo->getCp_R_ref(cp_r_ref);
                    thermo->getStandardVolumes_ref(v_ref);

                    thermo->getActivities(act);
                    thermo->getActivityCoefficients(act_coeff);
                    thermo->getLnActivityCoefficients(ln_act_coeff);

                    double H_mass = thermo->enthalpy_mass();
                    double U_mass = thermo->intEnergy_mass();
                    double S_mass = thermo->entropy_mass();
                    double V_mass = 1.0 / thermo->density();

                    // Inversion recovery tests
                    // 1. HP
                    thermo->setState_TP(T + 50.0, P);
                    thermo->setState_HP(H_mass, P, 1e-9);
                    double T_rec_hp = thermo->temperature();

                    // 2. UV
                    thermo->setState_TP(T - 50.0, P);
                    thermo->setState_UV(U_mass, V_mass, 1e-9);
                    double T_rec_uv = thermo->temperature();

                    // 3. SP
                    thermo->setState_TP(T + 50.0, P);
                    thermo->setState_SP(S_mass, P, 1e-9);
                    double T_rec_sp = thermo->temperature();

                    // 4. SV
                    thermo->setState_TP(T - 50.0, P);
                    thermo->setState_SV(S_mass, V_mass, 1e-9);
                    double T_rec_sv = thermo->temperature();

                    // Reset back to exact state
                    thermo->setState_TPX(T, P, comp);

                    double eq_ratio = thermo->equivalenceRatio();
                    double stoich_afr = 0.0;
                    try {
                        if (mech == "h2o2.yaml") {
                            stoich_afr = thermo->stoichAirFuelRatio("H2:1", "O2:1", ThermoBasis::molar);
                        } else {
                            stoich_afr = thermo->stoichAirFuelRatio("CH4:1", "O2:1, N2:3.76", ThermoBasis::molar);
                        }
                    } catch (...) {
                        stoich_afr = -1.0;
                    }

                    if (!first_case) out << ",\n";
                    first_case = false;

                    out << "    {\n";
                    out << "      \"mechanism\": \"" << mech << "\",\n";
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
                    out << "      \"enthalpy_mass\": " << H_mass << ",\n";
                    out << "      \"int_energy_mass\": " << U_mass << ",\n";
                    out << "      \"entropy_mass\": " << S_mass << ",\n";
                    out << "      \"gibbs_mass\": " << thermo->gibbs_mass() << ",\n";
                    out << "      \"cp_mass\": " << thermo->cp_mass() << ",\n";
                    out << "      \"cv_mass\": " << thermo->cv_mass() << ",\n";
                    out << "      \"isothermal_compressibility\": " << thermo->isothermalCompressibility() << ",\n";
                    out << "      \"thermal_expansion_coeff\": " << thermo->thermalExpansionCoeff() << ",\n";
                    out << "      \"sound_speed\": " << thermo->soundSpeed() << ",\n";
                    out << "      \"internal_pressure\": " << thermo->internalPressure() << ",\n";
                    out << "      \"equivalence_ratio\": " << eq_ratio << ",\n";
                    out << "      \"stoich_air_fuel_ratio\": " << stoich_afr << ",\n";
                    out << "      \"T_rec_hp\": " << T_rec_hp << ",\n";
                    out << "      \"T_rec_uv\": " << T_rec_uv << ",\n";
                    out << "      \"T_rec_sp\": " << T_rec_sp << ",\n";
                    out << "      \"T_rec_sv\": " << T_rec_sv << ",\n";

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
                    dump_vector(out, "enthalpy_RT_ref", h_rt_ref);
                    dump_vector(out, "entropy_R_ref", s_r_ref);
                    dump_vector(out, "gibbs_RT_ref", g_rt_ref);
                    dump_vector(out, "int_energy_RT_ref", u_rt_ref);
                    dump_vector(out, "cp_R_ref", cp_r_ref);
                    dump_vector(out, "standard_volumes_ref", v_ref);
                    dump_vector(out, "activities", act);
                    dump_vector(out, "activity_coefficients", act_coeff);
                    dump_vector(out, "ln_activity_coefficients", ln_act_coeff, true);

                    out << "    }";
                }
            }
        }
    }

    out << "\n  ]\n}\n";
    out.close();
    std::cout << "Successfully generated audit/native_thermo_family3_oracle.json" << std::endl;
    return 0;
}
