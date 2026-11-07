# Native Python solver

The independent backend will implement a variable-order stiff BDF integrator,
Newton nonlinear iteration, structured reactor Jacobians, factorization reuse,
and simultaneous forward sensitivities. It may use NumPy/SciPy numerical
kernels, but it may not load SUNDIALS or native Cantera.

This backend is currently **not implemented**. The existing adaptive RK4 code
is a compatibility fallback for the legacy backend and is not the independent
stiff solver described here. Selecting `native_python` fails explicitly so the
project cannot report a generic integrator as the requested algorithm.
