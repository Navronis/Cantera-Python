#include <iostream>
#include <vector>
#include <cantera/core.h>
#include <cantera/thermo/PengRobinson.h>
#include <cantera/thermo/RedlichKwongMFTP.h>

using namespace Cantera;

int main() {
    try {
        std::string test_data = "/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/test/data";
        auto solPR = newSolution(test_data + "/thermo-models.yaml", "CO2-PR");
        auto pr = std::dynamic_pointer_cast<PengRobinson>(solPR->thermo());
        if (!pr) {
            std::cerr << "Failed to cast to PengRobinson\n";
            return 1;
        }
        double T = 300.0;
        double P = 101325.0;
        pr->setState_TP(T, P);
        double a = 0, b = 0, aAlpha = 0;
        pr->calculateAB(a, b, aAlpha);
        double Vroot[3] = {0, 0, 0};
        int nsol = pr->solveCubic(T, P, a, b, aAlpha, Vroot);
        std::cout << "PR: nsol=" << nsol << " roots=" << Vroot[0] << ", " << Vroot[1] << ", " << Vroot[2] << " molarVol=" << pr->molarVolume() << "\n";

        auto solRK = newSolution(test_data + "/co2_RK_example.yaml");
        auto rk = std::dynamic_pointer_cast<RedlichKwongMFTP>(solRK->thermo());
        if (!rk) {
            std::cerr << "Failed to cast to RedlichKwongMFTP\n";
            return 1;
        }
        rk->setState_TP(T, P);
        double aRK = 0, bRK = 0;
        rk->calculateAB(T, aRK, bRK);
        double VrootRK[3] = {0, 0, 0};
        int nsolRK = rk->solveCubic(T, P, aRK, bRK, VrootRK);
        std::cout << "RK: nsol=" << nsolRK << " roots=" << VrootRK[0] << ", " << VrootRK[1] << ", " << VrootRK[2] << " molarVol=" << rk->molarVolume() << "\n";
        return 0;
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
