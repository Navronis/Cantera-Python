# Wave 3 `cp` / `cv` source formula audit
Repository snapshot identity: `fc63583ec6249d35444d7e68c5971a96b67aa7051af9d75904ffafd151bdc2ca` (`WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH`).

Reference source: pinned Cantera commit `726522be4e2a13454d8415b7ef799d621f665cf3`.

This mapping was derived by reading the executable statements in native `src/tpx/Sub.cpp` and Python `cantera_python/tpx.py`. It corrects the earlier descriptions of `dt` and of the quantity differenced for `cp`.

## `cv`: isochoric entropy difference

Native implementation: `outputs/cantera-original/src/tpx/Sub.cpp:44-80`. Python implementation: `cantera_python/tpx.py:301-324`.

Both implementations return NaN when `TwoPhase(true)` is true. Otherwise they set `Tsave = T`, `dt = 1e-4 * T`, `T1 = max(Tmin, Tsave - dt)`, and `T2 = min(Tmax, Tsave + dt)`. They perturb temperature using `set_T`, which holds the density/specific volume fixed. They sample entropy as `s1 = s(T1, v)` and `s2 = s(T2, v)`. If the initial state has quality exactly 0 or 1 and a perturbed endpoint enters the two-phase region, that endpoint is reset to `Tsave`; this makes the difference one-sided at that boundary. They restore `Tsave` at the original density and return:

\[
c_v = T_{save}\,\frac{s_2-s_1}{T_2-T_1}.
\]

The code differences entropy, not internal energy. The nominal `dt/T` is `1e-4`; clipping and boundary handling can make the effective interval smaller or one-sided.

## `cp`: isobaric entropy difference with saturation-aware endpoints

Native implementation: `outputs/cantera-original/src/tpx/Sub.cpp:82-132`. Python implementation: `cantera_python/tpx.py:326-364`.

Both return positive infinity in the two-phase region. Otherwise they set `Tsave = T`, `dt = 1e-4 * T`, save `RhoSave`, and save the initial pressure `p0`. The liquid or vapor path is selected from `Rho >= Rhf`.

For a liquid state, the lower endpoint is `T1 = max(Tmin, Tsave-dt)` and is evaluated at `(T1,p0)`. The upper endpoint is limited by `min(Tsat(p0), Tsave+dt)`; when saturation clips it, the code evaluates saturated liquid `(T2,x=0)`, otherwise it evaluates `(T2,p0)`.

For a vapor state, the lower endpoint is limited by `max(Tsat(p0), Tsave-dt)`; when saturation clips it, the code evaluates saturated vapor `(T1,x=1)`, otherwise `(T1,p0)`. The upper endpoint is `min(Tmax,Tsave+dt)` and is evaluated at `(T2,p0)`.

Both paths sample entropy at their selected endpoints and restore the original state with `(Tsave, 1/RhoSave)`, i.e. the original temperature and density. The returned quantity is:

\[
c_p = T_{save}\,\frac{s_2-s_1}{T_2-T_1}.
\]

The code differences entropy, not enthalpy. `dt/T` is nominally `1e-4`; saturation and temperature-limit logic can alter the actual interval and phase of an endpoint.

## Parity and limitations

The native and Python source use the same nominal step and the same entropy-difference forms in the mapped branches. This is source inspection, not proof that every fluid-specific `Set`, saturation, exception, or state-restoration path produces identical intermediate states.

The fresh 9,608-case production comparison is summarized in `wave3_metric_reconciliation.md`. A separate 120-row step sweep has now run against the live pinned oracle at `dt/T = 1e-3, 5e-4, 1e-4, 5e-5, 1e-5`. That sweep uses manually parameterized centered entropy differences on the native side, rather than the production `Sub.cpp` clipping and saturation-aware endpoint logic. It measures the chosen estimators but cannot establish why production `cp` / `cv` outputs differ. The current cause disposition remains `CAUSE_NOT_FULLY_ESTABLISHED`; no truncation-error attribution is made.
