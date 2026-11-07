# Implementation plan

The port targets Cantera commit `726522be4e2a13454d8415b7ef799d621f665cf3`. Completion means equation coverage, API behavior, numerical convergence, failure semantics, and exact-pinned differential evidence. A passing unit suite alone is insufficient.

## Current milestone

The 2026-09-18 checkpoint closes hierarchical mechanism metadata preservation and adds reaction-rate multipliers plus integrated forward-difference reactor sensitivities. YAML round trips now retain user-defined document, phase, species, equation-of-state, and reaction records. Reactor sensitivity coefficients use Cantera's signed component normalization and were compared in one supplementary case with released Cantera 3.2. The backend remains slower than CVODES simultaneous sensitivities, species-enthalpy sensitivity is still absent, and exact-pinned sensitivity validation remains required.

## Ordered work

1. Harden TP/HP equilibrium over randomized feeds, lean/rich mixtures, dissociation temperatures, trace species, ions, and phase temperature bounds. Add analytic conditioning checks and exact-pinned fixtures. Then add UV, SV, and SP.
2. Complete gas kinetics managers: reaction orders, duplicate detection, third-body semantics, reversible rates, derivatives, electrochemistry, and plasma energy coupling. Validate every supported reaction family over pressure, temperature, and composition grids.
3. Replace approximate transport paths with pinned mixture-averaged and multicomponent algorithms, including collision-integral tables, conductivity, diffusion, ionized-gas behavior, and thermal diffusion.
4. Complete zero-dimensional reactor equations, walls, reservoirs, flow devices, ReactorNet tolerances, dense/sparse Jacobians, sensitivities, and stiff integration semantics. The pure-Python fallback remains useful but is not a SUNDIALS-equivalent solver.
5. Port nonideal thermo and TPX models, beginning with Redlich-Kwong and water/steam because they block high-pressure and cryogenic work.
6. Port interfaces, surface kinetics, coverage dependencies, electrochemical phases, and multiphase VCS equilibrium.
7. Port one-dimensional domains, Newton solves, grid refinement, transport coupling, flame classes, restart, and diagnostics.
8. Close public API, serialization, YAML round trips, units, aliases, SolutionArray behavior, and error compatibility. Port upstream tests by subsystem and audit all 5,784 inventoried symbols.

Every milestone must update the feature matrix, mechanism matrix, oracle statistics, tests, limitations, and `PORT_STATUS.json`. `PORT_COMPLETE` stays false until every required feature is present and exact-pinned validation passes.
