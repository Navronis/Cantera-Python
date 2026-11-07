# Solver Wave 3 TPX architecture

The pinned TPX stack has one stateful `Substance` base (`src/tpx/Sub.cpp`) and
eight source fluid models: water, carbon dioxide, nitrogen, oxygen, hydrogen,
methane, heptane, and HFC-134a. The Python implementation keeps the same
public state (`T`, density, saturation densities, quality, offsets), property
pair codes, saturation cache, and failure boundaries.

`cantera_python/tpx.py` owns the common state and inversion API. Its density
inversion uses a safeguarded Newton step for `TP`; scalar property pairs use a
bracketed bisection fallback. Two-phase properties are evaluated with the
source lever rule in specific volume. `cantera_python/tpx_exact.py` contains
the fluid equations translated from the pinned `.cpp` files. No cubic EOS,
CoolProp, REFPROP, native Cantera, or other runtime fluid library is used.

The Reynolds fluids (H2, CH4, N2, O2) share the 14-term temperature residual
basis and recursive density integrals. CO2 and heptane use their source
coefficient bases. Water uses the seven-term Reynolds/IAPWS-era formulation in
`tpx.py`, including its saturation and liquid-density tables. HFC-134a uses
the Tillner-Roth/Baehr Helmholtz terms and derives entropy from `u-f/T`, as in
the pinned header.

This is an implementation record, not a claim of live native parity. The
required pinned native executable is unavailable in the current environment;
therefore `SOLVER_WAVE_3_LIVE_NATIVE_VALIDATED` remains false.
