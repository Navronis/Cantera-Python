# Fixed-stoichiometry checkpoint: remaining scope

`cantera_python.stoich.StoichiometricSubstance` implements selected scalar equations from the pinned `StoichSubstance` and `SingleSpeciesTP` source: constant-volume pressure updates, standard-state pressure corrections, pressure-independent internal energy, activities, concentrations, and single-species partial-molar properties.

It is not the mutable Cantera `ThermoPhase` hierarchy. Input-file/YAML construction, factories, derivatives, `setToEquilState`, phase equilibrium, kinetics, transport, reactors, and numerical solvers remain unported.
