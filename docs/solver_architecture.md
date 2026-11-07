# Solver architecture

The physical-model layer supplies thermodynamic properties, reaction rates,
transport coefficients, and reactor residuals. Numerical integration is a
separate layer. This separation prevents a solver choice from changing the
governing equations.

`ReactorNet(..., solver_backend="legacy")` is the currently operational path.
It uses SciPy BDF, may retry with LSODA, and uses an adaptive RK4 fallback when
SciPy is unavailable. It is retained for compatibility and is classified as
`APPROXIMATE`.

`solver_backend="parity"` uses the standalone scikit-SUNDAE CVODE binding,
which bundles SUNDIALS 7.5.0. It calls the Python reactor RHS and does not load
Cantera or `_cantera`. Reactor ODE integration is operational. The binding does
not expose CVODES forward sensitivities, so a parity sensitivity request raises
`NotImplementedError` rather than silently substituting finite differences.

`solver_backend="native_python"` is reserved for the independent variable-order
stiff BDF/Newton implementation. NumPy and SciPy linear algebra kernels may be
used, but SUNDIALS and native Cantera are forbidden. Until that implementation
exists, advancing also raises `NotImplementedError`.

The authoritative algorithm inventory is
`audit/complete_solver_inventory.jsonl`. Scientific status is recorded in
`audit/scientific_status.json`; symbol availability is tracked separately and
does not prove numerical equivalence.
