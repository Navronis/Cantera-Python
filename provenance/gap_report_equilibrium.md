# Ideal-gas equilibrium-constant checkpoint: remaining scope

`cantera_python.equilibrium` provides standard-state reaction changes and ideal-gas `Kp` / concentration `Kc` values for supplied stoichiometry. The `Kc` conversion follows `BulkKinetics::getEquilibriumConstants`, including fractional net gas stoichiometric coefficients.

This module does not calculate equilibrium compositions or solve any constrained equilibrium problem. It excludes VCS, MultiPhaseEquil, element-potential methods, non-ideal phases, electrochemistry, multiphase states, reaction parsing, and automatic reverse-rate integration.
