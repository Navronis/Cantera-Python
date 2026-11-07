// Comprehensive native test oracle for all SpeciesThermo models in Family 2
#include <iostream>
#include <iomanip>
#include <vector>
#include <span>
#include <fstream>
#include <map>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/MultiSpeciesThermo.h"
#include "cantera/thermo/SpeciesThermoInterpType.h"
#include "cantera/thermo/NasaPoly1.h"
#include "cantera/thermo/NasaPoly2.h"
#include "cantera/thermo/Nasa9Poly1.h"
#include "cantera/thermo/Nasa9PolyMultiTempRegion.h"
#include "cantera/thermo/ShomatePoly.h"
#include "cantera/thermo/ConstCpPoly.h"
#include "cantera/thermo/Mu0Poly.h"

using namespace Cantera;
using namespace std;

void eval_model(ofstream& out, const string& model_name, SpeciesThermoInterpType& model, const vector<double>& temps, bool is_first) {
    if (!is_first) out << ",\n";
    out << "    {\n";
    out << "      \"model\": \"" << model_name << "\",\n";
    out << "      \"report_type\": " << model.reportType() << ",\n";
    out << "      \"min_temp\": " << model.minTemp() << ",\n";
    out << "      \"max_temp\": " << model.maxTemp() << ",\n";
    out << "      \"ref_pressure\": " << model.refPressure() << ",\n";

    try {
        double hf = model.reportHf298();
        out << "      \"hf298\": " << setprecision(17) << hf << ",\n";
    } catch (Cantera::NotImplementedError&) {
        // Models without Hf298 override
    }

    out << "      \"points\": [\n";

    for (size_t i = 0; i < temps.size(); ++i) {
        double T = temps[i];
        double cp_R = 0.0, h_RT = 0.0, s_R = 0.0;
        model.updatePropertiesTemp(T, cp_R, h_RT, s_R);

        if (i > 0) out << ",\n";
        out << "        {\n";
        out << "          \"temperature\": " << T << ",\n";
        out << "          \"cp_R\": " << setprecision(17) << cp_R << ",\n";
        out << "          \"h_RT\": " << setprecision(17) << h_RT << ",\n";
        out << "          \"s_R\": " << setprecision(17) << s_R << "\n";
        out << "        }";
    }
    out << "\n      ]\n    }";
}

int main() {
    ofstream out("audit/native_thermo_family2_all_models_oracle.json");
    out << "{\n  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n  \"models\": [\n";

    // 1. NasaPoly1
    vector<double> nasa1_c = {3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00};
    NasaPoly1 nasa1(200.0, 1000.0, 101325.0, span<const double>(nasa1_c));
    eval_model(out, "NasaPoly1", nasa1, {250.0, 400.0, 700.0, 1000.0}, true);

    // 2. NasaPoly2
    vector<double> nasa2_c = {1000.0,
        3.69777829E+00, 6.13519789E-04, -1.25884282E-07, 1.77528148E-11, -1.13643547E-15, -1.23393018E+03, 3.18916559E+00,
        3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00};
    NasaPoly2 nasa2(200.0, 3500.0, 101325.0, span<const double>(nasa2_c));
    eval_model(out, "NasaPoly2", nasa2, {300.0, 800.0, 1000.0, 1200.0, 2000.0}, false);

    // 3. Nasa9Poly1
    vector<double> nasa9_c = {1.0e4, -1.0e2, 3.5, 1e-3, -1e-6, 1e-9, -1e-12, 100.0, 5.0};
    Nasa9Poly1 nasa9_1(200.0, 1000.0, 101325.0, span<const double>(nasa9_c));
    eval_model(out, "Nasa9Poly1", nasa9_1, {300.0, 500.0, 800.0, 1000.0}, false);

    // 4. Nasa9PolyMultiTempRegion
    vector<double> nasa9_multi_c = {2.0, 200.0, 1000.0, 1.0e4, -1.0e2, 3.5, 1e-3, -1e-6, 1e-9, -1e-12, 100.0, 5.0,
                                        1000.0, 3000.0, 1.2e4, -1.1e2, 3.8, 8e-4, -8e-7, 8e-10, -8e-13, 120.0, 5.5};
    Nasa9PolyMultiTempRegion nasa9_multi(200.0, 3000.0, 101325.0, span<const double>(nasa9_multi_c));
    eval_model(out, "Nasa9PolyMultiTempRegion", nasa9_multi, {300.0, 800.0, 1000.0, 1500.0, 2500.0}, false);

    // 5. ShomatePoly
    vector<double> shomate_c = {29.5, -3.0, 5.0, -1.0, 0.1, -10.0, 200.0};
    ShomatePoly shomate1(298.15, 1000.0, 101325.0, span<const double>(shomate_c));
    eval_model(out, "ShomatePoly", shomate1, {300.0, 500.0, 800.0, 1000.0}, false);

    // 6. ShomatePoly2
    vector<double> shomate2_c = {1000.0,
        29.5, -3.0, 5.0, -1.0, 0.1, -10.0, 200.0,
        32.0, -2.5, 4.0, -0.8, 0.05, -12.0, 210.0};
    ShomatePoly2 shomate2(298.15, 3000.0, 101325.0, span<const double>(shomate2_c));
    eval_model(out, "ShomatePoly2", shomate2, {300.0, 700.0, 1000.0, 1500.0, 2200.0}, false);

    // 7. ConstCpPoly
    vector<double> constcp_c = {298.15, 1.0e6, 2.0e5, 30000.0};
    ConstCpPoly constcp(200.0, 2000.0, 101325.0, span<const double>(constcp_c));
    eval_model(out, "ConstCpPoly", constcp, {300.0, 500.0, 1000.0, 1800.0}, false);

    // 8. Mu0Poly
    vector<double> mu0_c = {3.0, 0.0, 298.15, -1.0e5, 500.0, -1.2e5, 1000.0, -1.8e5};
    Mu0Poly mu0(298.15, 1000.0, 101325.0, span<const double>(mu0_c));
    eval_model(out, "Mu0Poly", mu0, {300.0, 450.0, 600.0, 900.0}, false);

    // 9. Enthalpy modification and reset on NasaPoly1
    double orig_h = nasa1.reportHf298();
    nasa1.modifyOneHf298(0, orig_h + 50000.0);
    double mod_h = nasa1.reportHf298();
    nasa1.resetHf298();
    double reset_h = nasa1.reportHf298();

    out << ",\n    {\n";
    out << "      \"model\": \"NasaPoly1_Hf298_Modification\",\n";
    out << "      \"orig_h\": " << setprecision(17) << orig_h << ",\n";
    out << "      \"modified_h\": " << setprecision(17) << mod_h << ",\n";
    out << "      \"reset_h\": " << setprecision(17) << reset_h << "\n";
    out << "    }";

    out << "\n  ]\n}\n";
    cout << "Wrote comprehensive Family 2 oracle results to audit/native_thermo_family2_all_models_oracle.json" << endl;
    return 0;
}
