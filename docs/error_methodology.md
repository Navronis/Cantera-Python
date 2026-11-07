# Numerical error methodology

Every native-versus-Python scalar comparison records absolute, relative,
mixed-tolerance, binary64 ULP, and (for two positive finite values) logarithmic
error. The shared implementation is `cantera_python.validation`.

For Python value $x_p$ and native value $x_n$:

$$E_{abs}=|x_p-x_n|$$

$$E_{rel}=\frac{|x_p-x_n|}{\max(|x_n|,\epsilon_{floor})}$$

$$E_{mixed}=\frac{|x_p-x_n|}{atol+rtol|x_n|}$$

A mixed-tolerance comparison passes only when $E_{mixed}\le 1$. Relative
error is retained for diagnostic use, but is marked non-meaningful when the
native magnitude is below the declared near-zero floor. Positive quantities
may also report

$$E_{log}=|\log_{10}(x_p)-\log_{10}(x_n)|.$$

ULP distance uses the ordered IEEE-754 binary64 bit representation. It is
reported for finite direct scalar results and is not used as a universal pass
criterion for iterative solvers, discontinuous root selection, or
ill-conditioned quantities.

`tools/build_error_audit.py` applies this framework to the hash- and
commit-gated focused pinned fixture and writes `audit/error_distribution.json`
and `audit/ulp_distribution.json`. Those files describe only cases represented
in that fixture. They do not certify untested subsystems.
