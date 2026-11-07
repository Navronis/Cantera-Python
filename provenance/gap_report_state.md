# HP state-inversion checkpoint: remaining scope

`cantera_python.state.set_state_td`, `set_state_hp`, `set_state_sp`, and `set_state_uv` are fixed-composition, ideal-gas-only state transformations. `set_state_td` directly applies the ideal-gas density relation; the others use bounded Newton updates in `ThermoPhase::setState_HPorUV` and `setState_SPorSV`. They return new immutable states at requested T/density, enthalpy/pressure, entropy/pressure, or internal-energy/specific-volume pairs.

They do not implement the full `ThermoPhase` state API, TV and other state-pair transformations, phase bounds, spinodal/stability handling, pure-fluid states, non-ideal equations of state, or equilibrium/transport/reactor solvers.

The ideal-gas DP path is implemented for both `IdealGasPhase` and `Solution`.
It preserves the pinned pressure-before-density validation order, independent
density mutation, derived-temperature mutation, and the post-exception state
for NaN and infinite inputs. Source and focused-test evidence is recorded in
`provenance/ideal_gas_set_state_dp_semantics_2026-09-18.json`. Exact executable
validation against the pinned build remains open.
