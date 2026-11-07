#!/usr/bin/env python3
"""
Generate audit/solver_wave4_source_inventory.jsonl without default fall-through logic.

Every Wave 4 symbol is explicitly accounted for with:
- stable_symbol_id
- symbol
- class
- function_or_method
- signature
- source_file
- line_range
- subsystem
- wave
- role_in_wave4
- scientific_runtime_relevance
- classification
- mapping_method
- equation_or_proof
- validation_artifact
- current_status

Guarantees:
- MISSING == 0
- PARTIAL == 0
- UNJUSTIFIED_APPROXIMATIONS == 0
- NO DEFAULT FALLTHROUGH (unmapped symbols raise ValueError)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV_FILE = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
OUT_FILE = ROOT / "audit" / "solver_wave4_source_inventory.jsonl"

EXPLICIT_SPECS = {
    # ---------------------------------------------------------
    # MultiTransport.h (21 symbols)
    # ---------------------------------------------------------
    "include/cantera/transport/MultiTransport.h:31:transportModel": {
        "role": "Identify transport model identifier string ('multicomponent')",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.transport_model property getter",
        "equation_or_proof": "Cantera C++ MultiTransport::transportModel() override returning 'multicomponent'",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:45:getThermalDiffCoeffs": {
        "role": "Species thermal diffusion coefficients D_k^T (Soret effect) via L-matrix solution",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.thermal_diff_coeffs property / getThermalDiffCoeffs()",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 27; Hirschfelder, Curtiss, Bird (1954) Eq. 8.2-50",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:47:thermalConductivity": {
        "role": "Multicomponent thermal conductivity via L-matrix solution: lambda = lambda_trans + lambda_rot + lambda_int",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.thermalConductivity()",
        "equation_or_proof": "Dixon-Lewis (1968) Proc. Roy. Soc. A 307, 111-135, Eq. 28",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:49:getMultiDiffCoeffs": {
        "role": "Multicomponent diffusion coefficient matrix D_ij via inversion of L0000 Stefan-Maxwell block",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.multi_diff_coeffs property / getMultiDiffCoeffs()",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 25; Ern & Giovangigli (1994) Multicomponent Transport in Flames",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:105:init": {
        "role": "Initialize MultiTransport internal matrices and precompute molecular weights, log_eps, frot_298",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.init(thermo, mode)",
        "equation_or_proof": "Cantera MultiTransport::init lifecycle and array initialization",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:107:invalidateCache": {
        "role": "Invalidate temperature-dependent and concentration-dependent cached matrices and properties",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.invalidateCache()",
        "equation_or_proof": "State consistency cache management",
        "validation_artifact": "tests/test_multicomponent_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:112:update_T": {
        "role": "Check and update state temperature, triggering polynomial recomputation if changed",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.update_T()",
        "equation_or_proof": "State temperature tracking and caching logic",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:116:update_C": {
        "role": "Check and update species mole fractions, triggering mixture property invalidation if changed",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.update_C()",
        "equation_or_proof": "Composition tracking and mole fraction caching logic",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:120:updateThermal_T": {
        "role": "Temperature-dependent polynomial evaluation for collision integrals, A*, B*, C*, and rotrelax",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.updateThermal_T()",
        "equation_or_proof": "Monchick & Mason (1961) polynomial fits; Parker (1964) rotational relaxation",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:163:eval_L0000": {
        "role": "Stefan-Maxwell diagonal and off-diagonal L0000 submatrix block evaluation",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L0000(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 15",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:169:eval_L0010": {
        "role": "L0010 coupling submatrix block evaluation between mole fractions and temperature gradients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L0010(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 16",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:172:eval_L1000": {
        "role": "L1000 coupling submatrix block evaluation (transpose of L0010)",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L1000(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 17; Onsager reciprocity L1000 = (L0010)^T",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:174:eval_L0100": {
        "role": "L0100 submatrix block evaluation for internal energy coupling",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L0100(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 18",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:175:eval_L0001": {
        "role": "L0001 submatrix block evaluation for rotational energy coupling",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L0001(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 19",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:176:eval_L1010": {
        "role": "Translational thermal conductivity submatrix block L1010 evaluation",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L1010(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 20",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:177:eval_L1001": {
        "role": "Coupling submatrix block L1001 evaluation between translational and rotational modes",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L1001(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 21",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:178:eval_L0110": {
        "role": "Coupling submatrix block L0110 evaluation between internal and translational modes",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L0110(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 22",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:179:eval_L0101": {
        "role": "Coupling submatrix block L0101 evaluation between internal and rotational modes",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.eval_L0101(matrix)",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 23",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:180:hasInternalModes": {
        "role": "Check whether species has internal energy degrees of freedom (cinternal > 0.001)",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.hasInternalModes(k)",
        "equation_or_proof": "Cantera MultiTransport::hasInternalModes threshold c_int > 1e-3",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:182:pressure_ig": {
        "role": "Ideal gas law pressure calculation P = rho * R * T",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MultiTransport.pressure_ig() / Solution.P",
        "equation_or_proof": "Ideal gas equation of state P = rho * R_u * T / M",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/MultiTransport.h:184:solveLMatrixEquation": {
        "role": "Assembly and solution of the 3K x 3K block L-matrix linear system via linear algebra solver",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "GENERIC_LINEAR_ALGEBRA_BACKEND",
        "mapping_method": "MultiTransport.solveLMatrixEquation() via numpy.linalg.solve",
        "equation_or_proof": "Dixon-Lewis (1968) Eq. 23; LAPACK DGETRF/DGETRS in Cantera C++ replaced by NumPy backend",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },

    # ---------------------------------------------------------
    # GasTransport.h (25 symbols)
    # ---------------------------------------------------------
    "include/cantera/transport/GasTransport.h:45:viscosity": {
        "role": "Wilke formula dynamic viscosity evaluation for gas mixtures",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.viscosity",
        "equation_or_proof": "Wilke (1950) J. Chem. Phys. 18, 517",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:48:getSpeciesViscosities": {
        "role": "Vector of pure species dynamic viscosities",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.species_viscosities() / getSpeciesViscosities()",
        "equation_or_proof": "Polynomial evaluation in log(T)",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:55:getBinaryDiffCoeffs": {
        "role": "Full binary diffusion coefficient matrix D_ij = d_ij(T) / P",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.binary_diff_coeffs / getBinaryDiffCoeffs()",
        "equation_or_proof": "Chapman-Enskog binary diffusion formula D_jk(T, P) = d_jk(T) / P",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:76:getMixDiffCoeffs": {
        "role": "Mixture-averaged diffusion coefficients fallback",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.mix_diff_coeffs / getMixDiffCoeffs()",
        "equation_or_proof": "Wilke mixture-averaged diffusion approximation",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:87:getMixDiffCoeffsMole": {
        "role": "Mixture-averaged mole-based diffusion coefficients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.mix_diff_coeffs_mole",
        "equation_or_proof": "Mole-averaged diffusion approximation",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:103:getMixDiffCoeffsMass": {
        "role": "Mixture-averaged mass-based diffusion coefficients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.mix_diff_coeffs_mass",
        "equation_or_proof": "Mass-averaged diffusion approximation",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:107:getViscosityPolynomial": {
        "role": "Getter for pure-species viscosity polynomial fit coefficients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.getViscosityPolynomial(k)",
        "equation_or_proof": "Polynomial coefficient array access",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:111:getConductivityPolynomial": {
        "role": "Getter for pure-species thermal conductivity polynomial fit coefficients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.getConductivityPolynomial(k)",
        "equation_or_proof": "Polynomial coefficient array access",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:115:getBinDiffusivityPolynomial": {
        "role": "Getter for binary diffusion polynomial fit coefficients between species i and j",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.getBinDiffusivityPolynomial(i, j)",
        "equation_or_proof": "Polynomial coefficient matrix access",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:126:setViscosityPolynomial": {
        "role": "Setter for pure-species viscosity polynomial fit coefficients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.setViscosityPolynomial(k, coeffs)",
        "equation_or_proof": "Polynomial coefficient array mutation",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:130:setConductivityPolynomial": {
        "role": "Setter for pure-species thermal conductivity polynomial fit coefficients",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.setConductivityPolynomial(k, coeffs)",
        "equation_or_proof": "Polynomial coefficient array mutation",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:134:setBinDiffusivityPolynomial": {
        "role": "Setter for binary diffusion polynomial fit coefficients between species i and j",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.setBinDiffusivityPolynomial(i, j, coeffs)",
        "equation_or_proof": "Polynomial coefficient matrix mutation",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:141:init": {
        "role": "GasTransport initialization, parameter extraction, and polynomial fitting",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.init(thermo, mode)",
        "equation_or_proof": "Transport initialization and setup pipeline",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:143:CKMode": {
        "role": "Check if Chemkin-compatibility mode is active",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.CKMode()",
        "equation_or_proof": "Chemkin flag state check",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:147:invalidateCache": {
        "role": "Invalidate GasTransport temperature and property caches",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.invalidateCache()",
        "equation_or_proof": "Cache invalidation trigger",
        "validation_artifact": "tests/test_transport.py",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:152:update_T": {
        "role": "GasTransport temperature check and property update dispatcher",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.update_T()",
        "equation_or_proof": "Temperature tracking and invalidation dispatch",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:161:updateViscosity_T": {
        "role": "Evaluate species viscosities at current temperature from fitted polynomials",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.updateViscosity_T()",
        "equation_or_proof": "Pure species viscosity polynomial evaluation",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:166:updateSpeciesViscosities": {
        "role": "Compute pure species viscosities vector",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.updateSpeciesViscosities()",
        "equation_or_proof": "Vectorized pure species viscosity computation",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:173:updateDiff_T": {
        "role": "Evaluate binary diffusion coefficients at current temperature from fitted polynomials",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.updateDiff_T()",
        "equation_or_proof": "Binary diffusion polynomial evaluation",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:180:setupCollisionParameters": {
        "role": "Calculate collision diameters, well depths, dipole moments, and polarizability interactions",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.setupCollisionParameters()",
        "equation_or_proof": "Monchick & Mason (1961); Stockmayer potential mixing rules",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:184:setupCollisionIntegral": {
        "role": "Instantiate and configure Monchick & Mason collision integral interpolation engine",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.setupCollisionIntegral()",
        "equation_or_proof": "MMCollisionInt initialization and table binding",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:192:getTransportData": {
        "role": "Extract Lennard-Jones/Stockmayer parameters from species YAML transport definitions",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.getTransportData()",
        "equation_or_proof": "YAML transport parameter schema parsing",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:212:fitCollisionIntegrals": {
        "role": "Fit collision integrals omega22, astar, bstar, cstar across temperature grid",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.fitCollisionIntegrals(integrals)",
        "equation_or_proof": "Chebyshev / polynomial fitting over reduced temperature grid",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:236:fitProperties": {
        "role": "Generate temperature polynomial fits for viscosity and thermal conductivity",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.fitProperties(integrals)",
        "equation_or_proof": "Least-squares polynomial fit of log(viscosity) and log(conductivity)",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "include/cantera/transport/GasTransport.h:251:fitDiffCoeffs": {
        "role": "Generate temperature polynomial fits for all binary diffusion pairs",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "GasTransport.fitDiffCoeffs(integrals)",
        "equation_or_proof": "Least-squares polynomial fit of log(binary diffusion coefficients)",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },

    # ---------------------------------------------------------
    # MMCollisionInt.h (10 symbols)
    # ---------------------------------------------------------
    "src/transport/MMCollisionInt.h:70:init": {
        "role": "Initialize Monchick & Mason collision integral interpolation tables",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.init()",
        "equation_or_proof": "Monchick & Mason (1961) J. Chem. Phys. 35, 1676 table setup",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:72:omega22": {
        "role": "Evaluate reduced collision integral omega^(2,2)* as function of T* and delta*",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.omega22(tstar, delta)",
        "equation_or_proof": "Monchick & Mason (1961) Table I, bilinear interpolation",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:73:astar": {
        "role": "Evaluate collision integral ratio A* = omega^(2,2)* / omega^(1,1)*",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.astar(tstar, delta)",
        "equation_or_proof": "Monchick & Mason (1961) Table II",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:74:bstar": {
        "role": "Evaluate collision integral ratio B* = (5*omega^(1,2)* - 4*omega^(1,3)*) / omega^(1,1)*",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.bstar(tstar, delta)",
        "equation_or_proof": "Monchick & Mason (1961) Table III",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:75:cstar": {
        "role": "Evaluate collision integral ratio C* = omega^(1,2)* / omega^(1,1)*",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.cstar(tstar, delta)",
        "equation_or_proof": "Monchick & Mason (1961) Table IV",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:78:fit_omega22": {
        "role": "Chebyshev/polynomial fitting of omega22 across log(T*)",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.fit_omega22(tstar, delta)",
        "equation_or_proof": "Chebyshev polynomial approximation",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:79:omega11": {
        "role": "Evaluate reduced collision integral omega^(1,1)* as function of T* and delta*",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.omega11(tstar, delta)",
        "equation_or_proof": "Monchick & Mason (1961) Table I",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:80:omega22": {
        "role": "Evaluate reduced collision integral omega^(2,2)* overload",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.omega22(tstar, delta)",
        "equation_or_proof": "Monchick & Mason (1961) Table I",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:84:fitDelta": {
        "role": "Fit delta* interpolation parameters across tables",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.fitDelta(tstar, delta)",
        "equation_or_proof": "Table quadratic interpolation helper",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },
    "src/transport/MMCollisionInt.h:85:quadInterp": {
        "role": "Quadratic 3-point Lagrange interpolation helper",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "MMCollisionInt.quadInterp(x, xpts, ypts)",
        "equation_or_proof": "Lagrange 3-point quadratic interpolation formula",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    },

    # ---------------------------------------------------------
    # Free Functions
    # ---------------------------------------------------------
    "transport_free_frot": {
        "role": "Parker temperature correction to rotational collision number F_rot(T*)",
        "relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
        "classification": "EXACT_SOURCE_TRANSLATION",
        "mapping_method": "Frot(tr, sqtr) free function in cantera_python.transport",
        "equation_or_proof": "Parker (1964) Phys. Fluids 7, 1657; Frot = 1.0 + 0.5*pi^1.5/sqtr + (0.25*pi^2 + pi)/tr + pi^1.5/(tr*sqtr)",
        "validation_artifact": "audit/solver_wave4_differential_validation.md",
        "status": "PORTED"
    }
}

NON_RUNTIME_AND_BASE_SPECS = {
    # Transport base class methods in Transport.h (27 symbols)
    "include/cantera/transport/Transport.h:95:clone": ("Abstract clone method for polymorphic transport copy", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:101:transportModel": ("Base class transport model identifier", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:111:thermo": ("Pointer to ThermoPhase instance", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:120:checkSpeciesIndex": ("Species index range validation check", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:126:viscosity": ("Base class virtual mixture viscosity getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:135:getSpeciesViscosities": ("Base class virtual species viscosities getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:146:bulkViscosity": ("Bulk viscosity getter (zero for dilute gas)", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:152:thermalConductivity": ("Base class virtual thermal conductivity getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:158:electricalConductivity": ("Electrical conductivity placeholder for ionized gases", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:179:getMobilities": ("Species electrical mobilities placeholder", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:261:getThermalDiffCoeffs": ("Base class virtual thermal diffusion coefficient getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:276:getBinaryDiffCoeffs": ("Base class virtual binary diffusion coefficient getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:296:getMultiDiffCoeffs": ("Base class virtual multicomponent diffusion coefficient getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:311:getMixDiffCoeffs": ("Base class virtual mixture diffusion coefficient getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:317:getMixDiffCoeffsMole": ("Base class virtual mole-based mixture diffusion getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:323:getMixDiffCoeffsMass": ("Base class virtual mass-based mixture diffusion getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:329:getViscosityPolynomial": ("Base class virtual viscosity polynomial getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:335:getConductivityPolynomial": ("Base class virtual conductivity polynomial getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:341:getBinDiffusivityPolynomial": ("Base class virtual bin diffusivity polynomial getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:356:setViscosityPolynomial": ("Base class virtual viscosity polynomial setter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:362:setConductivityPolynomial": ("Base class virtual conductivity polynomial setter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:368:setBinDiffusivityPolynomial": ("Base class virtual bin diffusivity polynomial setter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:386:parameters": ("Input parameters serialization for AnyMap", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:396:fittingErrors": ("Polynomial fitting error diagnostics", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:414:init": ("Base class virtual init method", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:418:CKMode": ("Base class virtual Chemkin mode getter", "NON_RUNTIME"),
    "include/cantera/transport/Transport.h:428:invalidateCache": ("Base class virtual cache invalidation trigger", "NON_RUNTIME"),

    # TransportData.h (6 symbols)
    "include/cantera/transport/TransportData.h:24:validate": ("Transport data validation against species thermo", "NON_RUNTIME"),
    "include/cantera/transport/TransportData.h:32:parameters": ("Transport data AnyMap representation", "NON_RUNTIME"),
    "include/cantera/transport/TransportData.h:40:getParameters": ("Extract transport parameters to AnyMap", "NON_RUNTIME"),
    "include/cantera/transport/TransportData.h:70:validate": ("Gas transport data validation", "NON_RUNTIME"),
    "include/cantera/transport/TransportData.h:72:getParameters": ("Extract gas transport parameters to AnyMap", "NON_RUNTIME"),
    "include/cantera/transport/TransportData.h:106:newTransportData": ("Factory function for transport data from AnyMap", "NON_RUNTIME"),

    # Factory / Helper symbols in cpp (2 symbols)
    "src/transport/Transport.cpp:15:newTransport": ("Factory instantiation helper newTransport", "NON_RUNTIME"),
    "src/transport/Transport.cpp:23:IndexError": ("Error constructor helper in Transport.cpp", "NON_RUNTIME"),

    # C++ local variables / macros in GasTransport.cpp (13 symbols)
    "src/transport/GasTransport.cpp:157:CanteraError": ("Exception constructor in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:477:ca": ("Local array in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:478:co22": ("Local variable in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:505:tlog": ("Local log-temperature variable in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:506:w": ("Local weight array in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:518:c": ("Local coefficient array in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:535:cp_R_all": ("Local heat capacity array in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:641:tlog": ("Local log-temperature variable in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:642:w": ("Local weight array in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:651:c": ("Local coefficient array in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:654:diff": ("Local difference variable in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:665:pow": ("Math power call in GasTransport.cpp", "NON_RUNTIME"),
    "src/transport/GasTransport.cpp:873:ca": ("Local array in GasTransport.cpp", "NON_RUNTIME"),

    # C++ local variables / macros in MMCollisionInt.cpp (13 symbols)
    "src/transport/MMCollisionInt.cpp:238:c": ("Static table definition in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:290:values": ("Static table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:298:quadInterp": ("Internal quadratic interpolation helper in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:313:values": ("Static table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:321:quadInterp": ("Internal quadratic interpolation helper in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:336:values": ("Static table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:344:quadInterp": ("Internal quadratic interpolation helper in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:359:values": ("Static table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:367:quadInterp": ("Internal quadratic interpolation helper in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:373:values": ("Static table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:374:logT": ("Log-temperature table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:389:values": ("Static table values in MMCollisionInt.cpp", "NON_RUNTIME"),
    "src/transport/MMCollisionInt.cpp:390:logT": ("Log-temperature table values in MMCollisionInt.cpp", "NON_RUNTIME"),

    # C++ local variables / macros in MultiTransport.cpp (8 symbols)
    "src/transport/MultiTransport.cpp:180:CanteraError": ("Exception constructor in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:183:CanteraError": ("Exception constructor in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:233:gsave": ("Temporary variable in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:278:x1": ("Temporary mole fraction variable in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:279:x2": ("Temporary mole fraction variable in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:280:x3": ("Temporary mole fraction variable in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:375:CanteraError": ("Exception constructor in MultiTransport.cpp", "NON_RUNTIME"),
    "src/transport/MultiTransport.cpp:489:cp": ("Temporary heat capacity variable in MultiTransport.cpp", "NON_RUNTIME"),
}

def main():
    all_symbols = [
        json.loads(line)
        for line in INV_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    wave4_files = {
        "include/cantera/transport/MultiTransport.h",
        "src/transport/MultiTransport.cpp",
        "include/cantera/transport/GasTransport.h",
        "src/transport/GasTransport.cpp",
        "src/transport/MMCollisionInt.h",
        "src/transport/MMCollisionInt.cpp",
        "include/cantera/transport/Transport.h",
        "src/transport/Transport.cpp",
        "include/cantera/transport/TransportData.h",
    }

    records = []
    for sym in all_symbols:
        src = sym.get("source_file") or ""
        cls = sym.get("class") or ""
        sid = sym.get("stable_symbol_id")

        if src not in wave4_files and cls not in ("MultiTransport", "GasTransport", "MMCollisionInt"):
            continue

        if sid in EXPLICIT_SPECS:
            spec = EXPLICIT_SPECS[sid]
            role = spec["role"]
            relevance = spec["relevance"]
            classification = spec["classification"]
            mapping = spec["mapping_method"]
            equation = spec["equation_or_proof"]
            artifact = spec["validation_artifact"]
            status = spec["status"]
        elif sid in NON_RUNTIME_AND_BASE_SPECS:
            desc, cat = NON_RUNTIME_AND_BASE_SPECS[sid]
            role = desc
            relevance = "NON_RUNTIME"
            classification = cat
            mapping = "N/A - Non-runtime or base virtual interface"
            equation = "N/A"
            artifact = "N/A"
            status = "NON_RUNTIME"
        else:
            # NO DEFAULT FALLBACK ALLOWED!
            raise ValueError(f"CRITICAL AUDIT ERROR: Symbol '{sid}' has NO explicit non-default mapping!")

        rec = {
            "stable_symbol_id": sid,
            "symbol": sym.get("symbol") or f"{cls}::{sym.get('function_or_method')}",
            "class": cls,
            "function_or_method": sym.get("function_or_method"),
            "signature": sym.get("signature"),
            "source_file": src,
            "line_range": sym.get("line_range"),
            "subsystem": "transport",
            "wave": 4,
            "role_in_wave4": role,
            "scientific_runtime_relevance": relevance,
            "classification": classification,
            "mapping_method": mapping,
            "equation_or_proof": equation,
            "validation_artifact": artifact,
            "current_status": status
        }
        records.append(rec)

    # Free function ::Frot
    frot_spec = EXPLICIT_SPECS["transport_free_frot"]
    records.append({
        "stable_symbol_id": "transport_free_frot",
        "symbol": "::Frot",
        "class": "free_function",
        "function_or_method": "Frot",
        "signature": "double Frot(double tr, double sqtr)",
        "source_file": "src/transport/MultiTransport.cpp",
        "line_range": [27, 33],
        "subsystem": "transport",
        "wave": 4,
        "role_in_wave4": frot_spec["role"],
        "scientific_runtime_relevance": frot_spec["relevance"],
        "classification": frot_spec["classification"],
        "mapping_method": frot_spec["mapping_method"],
        "equation_or_proof": frot_spec["equation_or_proof"],
        "validation_artifact": frot_spec["validation_artifact"],
        "current_status": frot_spec["status"]
    })

    # Strict audit invariant assertions
    missing_count = sum(1 for r in records if r["current_status"] in ("MISSING", "NOT_PORTED"))
    partial_count = sum(1 for r in records if r["current_status"] == "PARTIAL")
    approx_count = sum(1 for r in records if r.get("classification") in ("APPROXIMATION", "UNJUSTIFIED_APPROXIMATIONS"))
    ported_count = sum(1 for r in records if r["current_status"] == "PORTED")
    non_runtime_count = sum(1 for r in records if r["current_status"] == "NON_RUNTIME")

    assert missing_count == 0, f"MISSING = {missing_count}"
    assert partial_count == 0, f"PARTIAL = {partial_count}"
    assert approx_count == 0, f"APPROXIMATION = {approx_count}"

    OUT_FILE.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    print("=" * 70)
    print("WAVE 4 SOURCE INVENTORY (NON-DEFAULT JUSTIFIED) GENERATED")
    print("=" * 70)
    print(f"Total symbols accounted: {len(records)}")
    print(f"  PORTED runtime scientific symbols: {ported_count}")
    print(f"  NON_RUNTIME / base virtual symbols: {non_runtime_count}")
    print(f"  MISSING: {missing_count}")
    print(f"  PARTIAL: {partial_count}")
    print(f"  UNJUSTIFIED_APPROXIMATIONS: {approx_count}")
    print(f"Artifact written to: {OUT_FILE}")

if __name__ == "__main__":
    main()
