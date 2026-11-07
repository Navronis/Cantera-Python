#!/usr/bin/env python3
"""Generate provenance/solver_wave4_equations.jsonl."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "provenance" / "solver_wave4_equations.jsonl"

equations = [
    {
        "equation_id": "EQ_WAVE4_FROT",
        "name": "Parker Rotational Collision Number Temperature Correction",
        "source_reference": "MultiTransport.cpp:27-33; Kee et al. (2003) Eq. (12.112)",
        "latex": "F_{\\mathrm{rot}}(T^*) = 1 + \\frac{\\pi^{3/2}}{2} \\sqrt{T^*} + \\left(\\frac{\\pi^2}{4} + 2\\right) T^* + \\pi^{3/2} (T^*)^{3/2}",
        "variables": {
            "tr": "Reduced temperature epsilon / (k_B * T)",
            "sqtr": "sqrt(tr)"
        },
        "purpose": "Calculates the temperature scaling factor for rotational collision relaxation number Z_rot."
    },
    {
        "equation_id": "EQ_WAVE4_STOCKMAYER_POTENTIAL",
        "name": "Stockmayer 12-6-3 Intermolecular Potential",
        "source_reference": "MMCollisionInt.h:28; Monchick & Mason (1961) Eq. (16)",
        "latex": "\\phi(r) = 4 \\epsilon \\left[ \\left(\\frac{\\sigma}{r}\\right)^{12} - \\left(\\frac{\\sigma}{r}\\right)^6 + \\delta \\left(\\frac{\\sigma}{r}\\right)^3 \\right]",
        "variables": {
            "epsilon": "Potential well depth [J]",
            "sigma": "Collision diameter [m]",
            "delta": "Reduced dipole interaction parameter"
        },
        "purpose": "Governs intermolecular collisions for polar and nonpolar gas molecules."
    },
    {
        "equation_id": "EQ_WAVE4_REDUCED_DIPOLE",
        "name": "Reduced Dipole Moment (Delta Star)",
        "source_reference": "GasTransport.cpp:342-343; Kee et al. (2003) Eq. (11.109)",
        "latex": "\\delta_{ij}^* = \\frac{1}{2} \\frac{\\mu_{ij}^2}{4 \\pi \\epsilon_0 \\epsilon_{ij} \\sigma_{ij}^3}",
        "variables": {
            "mu_ij": "Effective pair dipole moment [C*m]",
            "epsilon_ij": "Effective pair well depth [J]",
            "sigma_ij": "Effective hard-sphere pair diameter [m]",
            "epsilon_0": "Vacuum permittivity [F/m]"
        },
        "purpose": "Dimensionless dipole interaction strength used to interpolate collision integrals."
    },
    {
        "equation_id": "EQ_WAVE4_POLAR_NONPOLAR_CORRECTIONS",
        "name": "Dixon-Lewis Polar-Nonpolar Correction Factors",
        "source_reference": "GasTransport.cpp:445-454; Dixon-Lewis (1968)",
        "latex": "\\xi = 1 + \\frac{1}{4} \\alpha^* (\\mu_p^*)^2 \\sqrt{\\frac{\\epsilon_p}{\\epsilon_{np}}}, \\quad f_\\sigma = \\xi^{-1/6}, \\quad f_\\epsilon = \\xi^2",
        "variables": {
            "alpha_star": "alpha_np / sigma_np^3",
            "mu_p_star": "mu_p / sqrt(4 * pi * epsilon_0 * sigma_p^3 * eps_p)",
            "f_sigma": "Diameter scaling factor applied to sigma_ij",
            "f_eps": "Well-depth scaling factor applied to epsilon_ij"
        },
        "purpose": "Adjusts collision diameter and well depth for polar-induced dipole interactions."
    },
    {
        "equation_id": "EQ_WAVE4_SPECIES_VISCOSITY",
        "name": "Chapman-Enskog Pure Species Viscosity",
        "source_reference": "GasTransport.cpp:548-550; Kee et al. (2003) Eq. (12.100)",
        "latex": "\\eta_k = \\frac{5}{16} \\frac{\\sqrt{\\pi M_k k_B T / N_A}}{\\pi \\sigma_k^2 \\Omega^{(2,2)*}(T_k^*, \\delta_k^*)}",
        "variables": {
            "M_k": "Species molecular weight [kg/kmol]",
            "T": "Temperature [K]",
            "sigma_k": "Hard-sphere diameter [m]",
            "Omega22": "Collision integral Omega^(2,2)*"
        },
        "purpose": "Calculates dilute gas dynamic viscosity for pure species k."
    },
    {
        "equation_id": "EQ_WAVE4_WILKE_MIXTURE_VISCOSITY",
        "name": "Wilke Semi-Empirical Mixture Viscosity Rule",
        "source_reference": "GasTransport.cpp:78-81; Poling et al. (2001) Eq. (9-5.13)",
        "latex": "\\eta_{\\mathrm{mix}} = \\sum_{k=1}^K \\frac{\\eta_k X_k}{\\sum_{j=1}^K \\Phi_{kj} X_j}, \\quad \\Phi_{kj} = \\frac{[1 + (\\eta_k/\\eta_j)^{1/2} (M_j/M_k)^{1/4}]^2}{\\sqrt{8(1 + M_k/M_j)}}",
        "variables": {
            "eta_k": "Pure species viscosity [Pa*s]",
            "X_k": "Mole fraction of species k",
            "M_k": "Molecular weight [kg/kmol]",
            "Phi_kj": "Wilke interaction parameter matrix"
        },
        "purpose": "Computes gas mixture viscosity from pure species viscosities and mole fractions."
    },
    {
        "equation_id": "EQ_WAVE4_BINARY_DIFFUSION",
        "name": "First-Order Kinetic Theory Binary Diffusion Coefficient",
        "source_reference": "GasTransport.cpp:664-666; Kee et al. (2003) Eq. (12.105)",
        "latex": "\\mathcal{D}_{jk} = \\frac{3}{16} \\frac{\\sqrt{2 \\pi (k_B T)^3 / m_{jk}}}{P \\pi \\sigma_{jk}^2 \\Omega^{(1,1)*}(T_{jk}^*, \\delta_{jk}^*)}",
        "variables": {
            "m_jk": "Reduced mass M_j M_k / (N_A (M_j + M_k)) [kg]",
            "P": "Pressure [Pa]",
            "sigma_jk": "Effective collision diameter [m]",
            "Omega11": "Collision integral Omega^(1,1)*"
        },
        "purpose": "Computes binary diffusion coefficient between species j and k."
    },
    {
        "equation_id": "EQ_WAVE4_SELF_DIFFUSION",
        "name": "Self-Diffusion Coefficient with A* Collision Ratio",
        "source_reference": "MultiTransport.cpp:475-478; Kee et al. (2003) Eq. (12.115)",
        "latex": "\\mathcal{D}_{kk} = 1.2 \\frac{R T \\eta_k A^*_{kk}}{M_k P}",
        "variables": {
            "R": "Universal gas constant [J/kmol/K]",
            "T": "Temperature [K]",
            "eta_k": "Species viscosity [Pa*s]",
            "A_kk": "Collision integral ratio A* = Omega^(2,2)* / Omega^(1,1)*",
            "M_k": "Molecular weight [kg/kmol]"
        },
        "purpose": "Provides exact diagonal entry in binary diffusion matrix required by Stefan-Maxwell and L-matrix."
    },
    {
        "equation_id": "EQ_WAVE4_INTERNAL_MODES",
        "name": "Dimensionless Internal Heat Capacity",
        "source_reference": "MultiTransport.cpp:490-493; Dixon-Lewis (1968)",
        "latex": "c_{\\mathrm{int}, k} = \\frac{C_{p, k}^\\circ}{R} - \\frac{5}{2}",
        "variables": {
            "Cp_k": "Standard state constant-pressure molar heat capacity [J/kmol/K]",
            "R": "Gas constant [J/kmol/K]"
        },
        "purpose": "Extracts vibrational/electronic internal energy storage by subtracting translational contribution 2.5 R."
    },
    {
        "equation_id": "EQ_WAVE4_ROTATIONAL_RELAXATION",
        "name": "Temperature-Dependent Rotational Relaxation Number",
        "source_reference": "MultiTransport.cpp:470-473; Kee et al. (2003) Eq. (12.113)",
        "latex": "Z_{\\mathrm{rot}, k}(T) = \\max(1.0, Z_{\\mathrm{rot}, k}(298)) \\frac{F_{\\mathrm{rot}}(298)}{F_{\\mathrm{rot}}(T)}",
        "variables": {
            "Z_rot_298": "Rotational relaxation collision number at 298 K",
            "F_rot": "Parker temperature correction factor"
        },
        "purpose": "Quantifies energy transfer efficiency between translational and rotational modes."
    },
    {
        "equation_id": "EQ_WAVE4_L0000",
        "name": "L-Matrix Submatrix L00,00 Block",
        "source_reference": "MultiTransport.cpp:505-525; Kee et al. (2003) Eq. (12.121)",
        "latex": "L_{ij}^{00,00} = \\frac{16 T}{25} X_j \\left[ \\frac{M_j}{M_i} \\sum_{k=1}^K \\frac{X_k}{\\mathcal{D}_{ik}(1\\,\\text{Pa})} + \\frac{X_i}{\\mathcal{D}_{ij}(1\\,\\text{Pa})} \\right], \\quad L_{ii}^{00,00} = 0",
        "variables": {
            "X": "Mole fraction vector",
            "M": "Molecular weight vector",
            "D_unit": "Binary diffusion evaluated at 1 Pa"
        },
        "purpose": "Upper-left block of the 3K x 3K L-matrix system; inverted to produce multicomponent diffusion D_ij."
    },
    {
        "equation_id": "EQ_WAVE4_L0010",
        "name": "L-Matrix Submatrix L00,10 Block",
        "source_reference": "MultiTransport.cpp:527-545; Kee et al. (2003) Eq. (12.122)",
        "latex": "L_{ij}^{00,10} = -1.6 T X_i X_j \\frac{M_i (1.2 C_{ji}^* - 1)}{(M_j + M_i) \\mathcal{D}_{ji}}, \\quad L_{jj}^{00,10} = -\\sum_{i \\ne j} L_{ij}^{00,10}",
        "variables": {
            "Cstar": "Collision integral ratio C*"
        },
        "purpose": "Couples concentration gradients to temperature gradient driven fluxes."
    },
    {
        "equation_id": "EQ_WAVE4_L1010",
        "name": "L-Matrix Submatrix L10,10 Block",
        "source_reference": "MultiTransport.cpp:556-588; Kee et al. (2003) Eq. (12.124-12.125)",
        "latex": "L_{ij}^{10,10} = \\frac{16 T}{25} \\frac{X_i X_j M_i}{M_j \\mathcal{D}_{ij} (M_i + M_j)^2} \\left[ 13.75 M_j^2 - 3 M_j^2 B_{ij}^* - 4 M_j^2 A_{ij}^* \\left(1 + \\frac{5}{3\\pi} \\left( \\frac{c_{\\mathrm{rot}, j}}{Z_{\\mathrm{rot}, j}} + \\frac{c_{\\mathrm{rot}, i}}{Z_{\\mathrm{rot}, i}} \\right) \\right) \\right]",
        "variables": {
            "Bstar": "Collision integral ratio B*",
            "Astar": "Collision integral ratio A*",
            "crot": "Rotational heat capacity (0 monatomic, 1.0 linear, 1.5 nonlinear)",
            "Zrot": "Rotational relaxation number"
        },
        "purpose": "Translational thermal conduction block of the 3K x 3K L-matrix."
    },
    {
        "equation_id": "EQ_WAVE4_L1001",
        "name": "L-Matrix Submatrix L10,01 Block",
        "source_reference": "MultiTransport.cpp:590-612; Kee et al. (2003) Eq. (12.127)",
        "latex": "L_{ij}^{10,01} = \\frac{32 T}{5\\pi} \\frac{M_j X_j c_{\\mathrm{rot}, j}}{c_{\\mathrm{int}, j} Z_{\\mathrm{rot}, j}} \\frac{A_{ji}^* X_i}{(M_j + M_i) \\mathcal{D}_{ji}} \\quad (\\text{for } j \\text{ with internal modes})",
        "variables": {
            "cinternal": "Internal heat capacity"
        },
        "purpose": "Couples translational thermal conduction to internal molecular vibration/rotation."
    },
    {
        "equation_id": "EQ_WAVE4_L0101",
        "name": "L-Matrix Submatrix L01,01 Block",
        "source_reference": "MultiTransport.cpp:640-669; Kee et al. (2003) Eq. (12.130-12.131)",
        "latex": "L_{ii}^{01,01} = -\\frac{8}{\\pi} \\frac{M_i X_i^2 c_{\\mathrm{rot}, i}}{c_{\\mathrm{int}, i}^2 R \\eta_i Z_{\\mathrm{rot}, i}} - \\frac{4 T X_i}{c_{\\mathrm{int}, i}} \\sum_k \\frac{X_k}{\\mathcal{D}_{ik}} \\left[ 1 + \\frac{12 M_i c_{\\mathrm{rot}, i} A_{ik}^*}{5\\pi c_{\\mathrm{int}, i} Z_{\\mathrm{rot}, i} M_k} (1 - \\delta_{ik}) \\right]",
        "variables": {
            "eta_i": "Pure species viscosity"
        },
        "purpose": "Internal mode self-relaxation diagonal block."
    },
    {
        "equation_id": "EQ_WAVE4_THERMAL_CONDUCTIVITY",
        "name": "Multicomponent Mixture Thermal Conductivity",
        "source_reference": "MultiTransport.cpp:93-101; Kee et al. (2003) Eq. (12.128)",
        "latex": "\\lambda_{\\mathrm{multi}} = -4 \\sum_{k=1}^{2K} b_{K+k} a_{K+k}",
        "variables": {
            "b": "RHS vector (b[K:2K] = X_k, b[2K:3K] = X_k for internal modes)",
            "a": "Solution vector from L * a = b"
        },
        "purpose": "Computes exact multicomponent gas mixture thermal conductivity [W/m/K]."
    },
    {
        "equation_id": "EQ_WAVE4_THERMAL_DIFFUSION_COEFFS",
        "name": "Species Thermal Diffusion Coefficients (Soret Effect)",
        "source_reference": "MultiTransport.cpp:103-111; Kee et al. (2003) Eq. (12.126)",
        "latex": "D_k^T = \\frac{1.6}{R} M_k X_k a_k",
        "variables": {
            "a_k": "First K elements of solution vector a from L * a = b",
            "M_k": "Molecular weight [kg/kmol]",
            "R": "Universal gas constant [J/kmol/K]"
        },
        "purpose": "Computes thermal diffusion coefficients [kg/m/s] driving Soret mass transport."
    },
    {
        "equation_id": "EQ_WAVE4_MULTICOMPONENT_DIFFUSION_MATRIX",
        "name": "Multicomponent Diffusion Coefficient Matrix",
        "source_reference": "MultiTransport.cpp:398-406; Kee et al. (2003) Eq. (12.119)",
        "latex": "D_{ij} = \\frac{16 T \\bar{M}}{25 P M_j} X_i \\left[ (L^{00,00})^{-1}_{ij} - (L^{00,00})^{-1}_{ii} \\right], \\quad D_{ii} = 0",
        "variables": {
            "L0000_inv": "Matrix inverse of L00,00 block",
            "Mbar": "Mean molecular weight of the mixture [kg/kmol]",
            "P": "Pressure [Pa]"
        },
        "purpose": "Produces K x K multicomponent diffusion coefficient matrix [m^2/s]."
    },
    {
        "equation_id": "EQ_WAVE4_STEFAN_MAXWELL_SOLVER",
        "name": "Stefan-Maxwell Linear System with Mass Conservation Row Replacement",
        "source_reference": "MultiTransport.cpp:207-257; Kee et al. (2003) Eq. (12.167-12.170)",
        "latex": "A_{ij} = \\frac{X_i X_j}{\\mathcal{D}_{ij}}, \\quad A_{ii} = -\\sum_{j \\ne i} A_{ij}, \\quad A_{j_{\\max}, j} = Y_j, \\quad \\text{RHS}_{j_{\\max}} = 0",
        "variables": {
            "A": "Modified Stefan-Maxwell constraint matrix",
            "Y_j": "Mass fraction of species j",
            "j_max": "Index of species with maximum |grad X_j|"
        },
        "purpose": "Eliminates rank deficiency of Stefan-Maxwell equations and enforces exact zero mass flux sum."
    },
    {
        "equation_id": "EQ_WAVE4_DIFFUSIVE_MASS_FLUX",
        "name": "Multicomponent Diffusive Mass Flux with Soret Term",
        "source_reference": "MultiTransport.cpp:252-268; Kee et al. (2003) Eq. (12.168)",
        "latex": "j_k = \\frac{\\rho Y_k}{P} V_k - D_k^T \\frac{\\nabla T}{T}",
        "variables": {
            "V_k": "Diffusion velocity from Stefan-Maxwell solve",
            "rho": "Mass density [kg/m^3]",
            "D_k_T": "Thermal diffusion coefficient [kg/m/s]"
        },
        "purpose": "Computes species diffusive mass fluxes [kg/m^2/s] in arbitrary spatial dimensions."
    }
]

OUT_FILE.write_text("\n".join(json.dumps(eq) for eq in equations) + "\n", encoding="utf-8")
print(f"Generated {OUT_FILE} with {len(equations)} mathematical equations.")
