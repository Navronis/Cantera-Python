# Navronis readiness

This is a weighted engineering score for the stated gas-dynamics, propulsion, high-pressure, and cryogenic use case. It is not a claim of Cantera completeness.

| Capability | Weight | Credit | Evidence and gap |
|---|---:|---:|---|
| Ideal-gas thermodynamics | 10 | 7.5 | 10,050 exact-pinned states pass; not every thermo model or full temperature envelope is covered |
| Gas kinetics | 15 | 7.5 | Major gas-rate families exist; complete manager semantics and broad mechanism oracle sweeps remain open |
| TP/HP/UV equilibrium | 15 | 7.5 | TP and HP ideal-gas solver implemented; UV and nonideal/multiphase equilibrium absent |
| High-temperature dissociation | 5 | 2.5 | H2-air dissociation matches reference at 1 and 30 atm; broad fuels and ions unvalidated |
| Zero-D reactors | 10 | 5.0 | Basic constant-volume/pressure integration and conservation tests pass; stiff equivalence incomplete |
| Reactor trajectories | 5 | 1.25 | Short regression evidence exists; production ignition sweeps and sensitivities absent |
| High-pressure real gas | 10 | 0 | Cubic EOS absent |
| Cryogenic fluids | 10 | 0 | Water/steam, TPX, and cryogenic fluid models absent |
| Mixture transport | 5 | 2.5 | Partial mixture transport exists; exact-pinned range validation incomplete |
| Multicomponent transport | 5 | 0 | Absent |
| OpenFOAM export/integration | 5 | 0 | Absent |
| Parser and provenance | 5 | 2.5 | 31/34 bundled files parse; full phase construction and correctness are not certified |
| **Total** | **100** | **36.75** | **Not production ready** |

The next release gate is a randomized exact-pinned TP/HP equilibrium corpus with element, enthalpy, Gibbs, convergence, and failure checks, followed by UV equilibrium. Real-gas and cryogenic uses remain blocked until their equations of state are implemented and validated.

## Use-case scores

These are engineering gate scores on a 100-point scale using the evidence in the capability table. They are not percentages of Cantera source converted.

| Use case | Score | Main blockers |
|---|---:|---|
| Preliminary ideal-gas 0-D combustion | 55/100 | Complete ignition trajectories, stiff-solver equivalence, and production transport sweeps |
| Equilibrium chamber calculations | 45/100 | Randomized TP/HP oracle corpus, UV/SP, broader fuels, convergence and Gibbs-monotonicity evidence |
| High-pressure rocket-combustion properties | 20/100 | Real-gas EOS, high-pressure transport, propellant mechanisms, broad HP equilibrium validation |
| Cryogenic feed-system properties | 5/100 | Liquid/real-fluid EOS, saturation, phase boundaries, near-critical and inversion tests |
| Injector reacting CFD | 15/100 | Multicomponent/Soret transport, verified source terms, heat release, OpenFOAM export and coupled case |
| Full Cantera compatibility | 10/100 | 5,027 of 5,784 audited symbols remain unported, with flames, real fluids, multiphase and API closure missing |
