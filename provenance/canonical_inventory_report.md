# Canonical 5,784-symbol inventory report

Pinned source commit: `726522be4e2a13454d8415b7ef799d621f665cf3`

Canonical inventory: `provenance/canonical_5784_symbol_inventory.jsonl`

Generator: `tools/build_canonical_symbol_inventory.py`

## Count and identity reconciliation

The legacy inventory contains exactly 5,784 non-empty JSONL rows and exactly
5,784 unique symbol IDs. The canonical inventory retains those IDs unchanged as
`stable_symbol_id`; it does not remove examples, wrappers, utilities, or
apparently non-runtime C++ symbols from the denominator.

All rows identify the pinned commit. The 5,784 symbols reference 375 distinct
source files. Every referenced file exists in `outputs/cantera-original`, each
file has one internally consistent recorded SHA-256 value, and all 375 recorded
hashes match the current pinned-source bytes. No count, ID, commit, missing-file,
or source-hash discrepancy was found.

The generated artifact has 5,784 rows, 5,784 unique stable IDs, and every row
contains all 17 fields required by the canonical schema. After deterministic
scope enrichment, its SHA-256 digest is
`624cc1337df254fd24c4dad2b9dc49663b78341a7c024c72298b5f5bb43791c4`.

## Status reconciliation

The legacy status counts are preserved in every row:

| Legacy status | Count |
| --- | ---: |
| `EQUIVALENT_PROVEN` | 421 |
| `PARTIALLY_PORTED` | 336 |
| `NOT_PORTED` | 5,027 |
| **Total** | **5,784** |

The legacy audit generator assigned `EQUIVALENT_PROVEN` when a class was in a
hard-coded proven-class list, or when a native function name appeared in the
Python definition-name set and the source filename matched a ported stem. That
is useful discovery evidence, but it is not per-symbol proof of source-body,
branch, unit, state, mutation, derivative, boundary, exception, parser, and
exact pinned-native equivalence. The strict canonical status therefore does
not promote any of those 421 rows to either validated final status.

The corrected scanner found that 754 frozen IDs are executable `return` or
`throw` statements and five are constructor initializer-list or continued-expression
lines rather than C++ declarations. They remain in the mandated
denominator and each carries the four-part non-runtime proof recorded in
`symbol_discovery_false_positive_reconciliation.json`.

The canonical status counts are:

| Current status | Count |
| --- | ---: |
| `NON_RUNTIME_CPP_ONLY_PROVEN` | 759 |
| `IMPLEMENTED_UNVALIDATED` | 16 |
| `PARTIAL` | 678 |
| `NOT_PORTED` | 4,331 |
| Final dispositions | 759 |
| **Total** | **5,784** |

The original 757 touched rows were the union of the 421 legacy equivalent rows
and 336 legacy partial rows; 128 of those were fabricated statement rows and
are now non-runtime dispositions. The historical statement “421 of 5,784
equivalent-proven (7.28%)” remains recorded as a legacy audit snapshot. Under
the stricter final-status contract, 5,025 rows still require an allowed final
disposition and `FULL_PORT_COMPLETE` remains false.

## Schema reconciliation

The legacy file supplied only ID, subsystem, source location and hash, one
heuristically tracked class, symbol name/type, a broad algorithm category,
candidate Python name, status, and prose notes. It did not extract namespaces,
signatures, visibility, callers, callees, factory registrations, or explicit
runtime relevance.

The canonical file includes every required key. A comment- and literal-aware
lexical pass tracks balanced C/C++ namespace and class scopes and class access
labels. It extracts a normalized literal declaration from the already audited
source line. All 375 referenced files were scope-balanced under this parser.

| Field | Deterministically extracted | Deterministically absent / not applicable | Unresolved |
| --- | ---: | ---: | ---: |
| Namespace | 5,232 | 552 | 0 |
| Class scope | 4,231 | 1,553 | 0 |
| One-line source signature | 5,784 | 0 | 0 |
| Class visibility | 4,231 | 1,553 free/namespace functions | 0 |

The legacy class scanner did not pop C++ scopes. The lexical pass agrees with
its class candidate for 3,446 rows and finds a different class for 785 rows.
The canonical `class` field uses the lexical result and retains the former
value separately as `legacy_class_candidate` for traceability.

Scientific/runtime relevance is routed to candidate queues from source path
and the legacy algorithm category. These are work classifications, not final
proof that a symbol is required or safely excludable:

| Relevance candidate | Count |
| --- | ---: |
| `SCIENTIFIC_RUNTIME_CANDIDATE` | 4,070 |
| `NON_SYMBOL_INVENTORY_ARTIFACT` | 759 |
| `RUNTIME_UTILITY_CANDIDATE` | 447 |
| `VALIDATION_ONLY_CANDIDATE` | 444 |
| `NON_LIBRARY_SAMPLE_CANDIDATE` | 54 |
| `BINDING_RUNTIME_CANDIDATE` | 8 |
| `NON_LIBRARY_EXAMPLE_CANDIDATE` | 2 |
| **Total** | **5,784** |

Unextracted dependency and registration facts remain encoded as `null` or
`NOT_AUDITED`, with an explicit `metadata_extraction_status`, rather than
guessed. In particular:

- `class` records the balanced lexical class scope. The legacy candidate stays
  available for discrepancy review.
- `python_mapping` preserves the legacy name/class candidate and labels it
  `LEGACY_NAME_OR_CLASS_CANDIDATE_UNVERIFIED`; absence is `UNASSIGNED`.
- call graph and model/factory registration remain open audit work.
- candidate relevance must still be promoted or rejected through body and
  dependency evidence before it can support a final disposition.
- every open row has a concrete `required_next_action`; the 759 proven lexical
  artifacts have `NON_RUNTIME_CPP_ONLY_PROVEN` final dispositions.
- seven `UnitStack` callable rows have source-reviewed Python mappings and
  regression evidence in `symbol_implementation_evidence.jsonl`; they remain `PARTIAL`
  until fresh exact-pinned oracle and independent-review evidence exists.

## Reproduction and closure gate

Run:

```powershell
python tools/build_canonical_symbol_inventory.py
python tools/build_symbol_closure_dashboard.py
python tools/reconcile_symbol_discovery.py
python -m unittest tests.test_canonical_inventory
```

The generator fails if the denominator changes, stable IDs collide, a source
commit differs, a referenced source file is missing, or a pinned source hash
does not match. The closure dashboard reports 759 fully dispositioned rows and
5,025 remaining rows at this checkpoint. This is deliberately stricter than
the legacy 7.28% snapshot and does not claim scientific progress from schema
normalization alone.
