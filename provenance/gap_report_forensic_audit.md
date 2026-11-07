# Forensic audit limitations and findings

The lexical symbol inventory and code-math scan are deliberately conservative. A name match is classified as partial and does not establish equation equivalence. Function bodies, inline templates, data tables, registrations, and tests require continued human/source-level classification.

The current Python runtime has no native Cantera dependency. The exact pinned native oracle has not been built; released Cantera 3.2.0 remains supplementary only. Full YAML parsing, transport, equilibrium solvers, reactors, one-dimensional flames, TPX/water, and public API compatibility remain unresolved.

The SUNDIALS checkout contains Windows long-path missing/generated answer fixtures. Evidence is preserved in `source_integrity_sundials.md`; no reset was performed.
