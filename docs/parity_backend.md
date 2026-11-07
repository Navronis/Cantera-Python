# Parity backend

The parity backend is specified as SUNDIALS CVODES 7.5 for reactor ODEs and
IDAS 7.5 for FlowReactor DAEs, driven only by Python physics callbacks. Native
Cantera libraries and the `cantera`/`_cantera` modules are prohibited.

Parity requires matching the pinned source's state packing, tolerances, maximum
order and step controls, nonlinear iteration, linear solver selection,
sensitivity configuration, failure handling, and interpolation behavior.
The reactor ODE backend is operational through scikit-SUNDAE 1.1.3, bundling
SUNDIALS 7.5.0 with binary64 arithmetic. Dense BDF integration uses the Python
reactor RHS, pinned tolerances, maximum order 5, and a 20,000-step limit.

This is same-solver-family implementation evidence, not yet numerical parity
proof: exact pinned trajectory fixtures and complete native build settings are
still missing. CVODES forward sensitivities are not exposed by this binding and
therefore remain explicitly unimplemented in parity mode.
