# Known limitations

- Ideal-gas TP, HP, UV, SV, and SP equilibrium paths are implemented. Multiphase VCS, charged-phase constraints, and nonideal activities are absent.
- Gas kinetics supports selected rate families and common homogeneous mechanisms. Complete reaction validation, electrochemistry, surface kinetics, derivatives, and all factory registrations remain incomplete.
- Mixture transport and ionized transport are partial. Multicomponent diffusion, thermal diffusion, and full pinned collision-integral behavior are not complete.
- Reactor classes, walls, common flow devices, and an adaptive pure-Python fallback exist. Reaction-multiplier sensitivities use integrated forward finite differences and are much slower than simultaneous CVODES sensitivities. Species-enthalpy sensitivities and full SUNDIALS behavior, event semantics, and preconditioning remain absent.
- Redlich-Kwong, Peng-Robinson, water/steam, TPX real fluids, and cryogenic phase models are absent.
- One-dimensional flames, grid refinement, reaction paths, and broad SolutionArray/API compatibility are absent.
- YAML output preserves nested user metadata at document, phase, species, equation-of-state, and reaction levels. Full AnyMap unit propagation, source locations, comments, and every upstream serialization option remain incomplete.
- The static runtime scan still contains explicit `NotImplementedError` paths for unsupported function, kinetics, phase, transport, and equilibrium models. These paths are deliberate failures rather than silent substitutions.
- The 34-file mechanism CSV establishes parser reachability for 31 files. It does not certify numerical correctness of every phase or reaction.
- The exact-pinned native fixture was replayed and hash-verified, but could not be regenerated during this run because the host denied WSL launch.
- The pinned thermo fixture spans 252.33–3498.01 K and 1 kPa–5 MPa. A new pinned-native sweep is still required to cover the requested 200–6000 K envelope and every polynomial boundary.
- Released Cantera 3.2.0 is installed only in the isolated validation environment. Runtime purity tests reject imports or loading of native Cantera from `cantera_python`.
