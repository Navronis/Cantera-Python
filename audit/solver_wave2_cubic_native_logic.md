# Solver Wave 2: pinned cubic EOS logic

Pinned source: `726522be4e2a13454d8415b7ef799d621f665cf3`.

The common implementation is `MixtureFugacityTP::solveCubic` in
`src/thermo/MixtureFugacityTP.cpp:774-1014`. It solves a normalized cubic in
molar volume, `an*v^3 + bn*v^2 + cn*v + dn = 0`, using the depressed-cubic
center `xN=-bn/(3*an)`, `delta2=(bn^2-3*an*cn)/(9*an^2)`,
`h=2*an*sqrt(delta2)*delta2`, `yN=2*bn^3/(27*an^2)-bn*cn/(3*an)+dn`, and
`disc=yN^2-4*an^2*delta2^3`.

For positive discriminant Cantera uses the signed real cube-root Cardano
branch. For negative discriminant it uses the three cosine roots and stores
them in native branch order: `Vroot[0]=beta` (small-volume liquid),
`Vroot[1]=gamma` (middle unstable), `Vroot[2]=alpha` (large-volume gas).
For zero discriminant it distinguishes a triple root from a repeated-root
pair and preserves the native liquid/gas ordering. A 64-epsilon cancellation
test identifies a triple root. The ideal-gas limit has a ten-step Z iteration.

Every physical root must be finite and greater than `b*(1+1e-12)` (or zero if
`b` is negative). Each physical root receives up to 20 Newton polishing steps;
the native method under-relaxes a step by 0.1 when the residual gets worse and
rejects a root that fails its scaled residual test. A one-root result is signed
`-1` for a liquid/supercritical branch and `+1` for a gas branch. A repeated
pair is signed `-2` when it represents two distinct roots.

Peng-Robinson constructs `bn=b-RT/P`, `cn=-(2RTb/P-aAlpha/P+3b^2)`, and
`dn=b^2RT/P+b^3-aAlpha*b/P` in `PengRobinson.cpp:763-779`.
Redlich-Kwong constructs `bn=-RT/P`, `cn=-(RTb/P-a/(P*sqrt(T))+b^2)`, and
`dn=-a*b/(P*sqrt(T))` in `RedlichKwongMFTP.cpp:869-886`.

`densityCalc` filters physical roots, sorts them for selection, and chooses the
smallest root for a requested liquid, largest for gas/supercritical, or uses
the previous molar-volume guess for an undefined phase. The middle root is
never selected as a stable phase. `liquidVolEst` starts at the larger of the
estimated saturation pressure and the caller pressure, increasing pressure by
4% until a liquid branch is available or 100 attempts are exhausted.

The Python implementation is in `cantera_python/cubic_eos.py`. It now follows
the native discriminant cancellation, root order, physical-root filter,
Newton polishing/under-relaxation, branch signs, and saturation-pressure
starting rule. TPX/IAPWS remains outside this wave.
