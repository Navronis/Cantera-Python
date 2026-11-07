# Dependencies

The runtime does not import or load native Cantera. Its declared general-purpose dependencies are:

- `numpy`: state arrays, stoichiometric operations, kinetics vectorization, and numerical property calculations.
- `ruamel.yaml`: parsing Cantera YAML input while preserving YAML 1.2 scalar behavior (notably species names such as `NO`).
- `scipy`: generic stiff ODE and sparse numerical kernels used by the legacy reactor backend. Cantera-specific reactor states and governing equations remain in editable Python.
- `scikit-sundae==1.1.3` (optional `parity` extra): standalone CVODE and IDA bindings bundling SUNDIALS 7.5.0. The operational reactor parity path uses CVODE; FlowReactor IDA residual integration remains unimplemented. This package does not load native Cantera.

Native `cantera` is an optional validation-only dependency. It belongs in a separate reference environment and must not be installed as an implementation dependency.

The standalone CVODE path uses the same SUNDIALS version and solver family as the pinned source, but exact-pinned trajectory equivalence is not yet proven. CVODES sensitivities are not exposed by the binding. Preconditioning, full ReactorNet semantics, IDAS FlowReactor integration, and exact solver configuration remain explicit completion gaps.
