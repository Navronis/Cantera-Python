# Solver Wave 1: pinned thermodynamic inverse-state logic

Source commit: `726522be4e2a13454d8415b7ef799d621f665cf3`.

The public wrappers `ThermoPhase::setState_HP` (lines 200-210), `setState_UV`
(212-224), `setState_SP` (560-570), and `setState_SV` (572-582) save the
partial state, call a shared helper, and restore the saved state on any
exception. The helpers are `setState_HPorUV` (342-558) and `setState_SPorSV`
(585-781).

For HP, pressure is fixed and the residual is `R = Htarget - h_mass(T,P,X)`;
the derivative is `cp_mass`. For UV, density is fixed to `1/v`, the residual
is `Utarget - u_mass(T,rho,X)`, and the derivative is `cv_mass`. For SP and
SV the residuals are `Starget - s_mass`; the derivatives are `cp_mass/T` and
`cv_mass/T`, respectively, giving the source Newton increments
`dt=(Starget-S)*T/cp`.

The initial temperature is the current phase temperature, clipped inside the
temperature bounds by one kelvin. Bounds are `maxTemp()+0.1` and
`minTemp()-0.1` when enforcement is enabled; otherwise they are
`max(maxTemp()+1000,10*maxTemp)` and `clip(minTemp,SmallNumber,100)`.
Newton increments are clipped to ±100 K. The helper maintains upper and lower
property brackets, applies the source 0.75 bracket safeguard, retries up to
ten times when `cp`/`cv` is negative, and limits a trial temperature to no
less than one third of the previous temperature.

Convergence is exactly the source pair of tests: normalized property residual
below `rtol`, or relative temperature step below `rtol`. The iteration limit is
500. Invalid pressure or volume is rejected before iteration. Enforced bounds
raise when the target lies outside the reachable property range. The Python
immutable state layer returns a new state only after success, so its caller
retains the original state on failure, matching the public wrapper contract.

The pure-Python implementation is `_inverse_native` in
`cantera_python/state.py`; it is used by `set_state_hp`, `set_state_uv`,
`set_state_sp`, and `set_state_sv` without SciPy root-finding or native calls.
