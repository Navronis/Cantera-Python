# Cubic root-selection matrix

| Native condition | Selected volume | Python behavior |
|---|---:|---|
| Three physical roots, requested liquid | `Vroot[0]` | smallest physical root |
| Three physical roots, requested gas | `Vroot[2]` | largest physical root |
| Three physical roots, undefined phase | previous-volume comparison with middle root | same comparison |
| Two physical roots | lower/upper roots; middle is midpoint for selection | same |
| One root, gas or undefined | sole root if branch sign permits | same |
| One root, requested liquid but gas branch | failure sentinel `-2` | same |
| Supercritical root below critical volume | branch sign `-1` | same |
| Root at or below `b*(1+1e-12)` | rejected as nonphysical | same |
| Triple critical root | one usable root | one usable root |

The native solver does not use a generic polynomial root package. It uses
Cardano/cosine roots followed by Newton polishing. The Python implementation
does the same.
