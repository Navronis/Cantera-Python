# Implementation dependency graph

```text
units / composition / elements / exceptions
             |
       species + parser + registries
             |
      species thermo + phase state
             |
 thermo standard states / activities / derivatives
             |
 kinetics rates + reaction stoichiometry
             |
 kinetics managers + transport state
             |
 equilibrium / reactors / sensitivities
             |
       one-dimensional flames
```

Transport and kinetics share species thermo and state data. Equilibrium depends on thermo, composition, and numerical residual infrastructure. Reactors depend on thermo, kinetics, and integrators. One-dimensional flames depend on all of those plus transport, Jacobians, nonlinear solves, and mesh management. YAML/factories cross-cut every layer and are a critical-path blocker for mechanism compatibility.
