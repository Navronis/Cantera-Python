"""Authoritative state-revision tracking and revision-keyed caches.

Pinned semantics being reproduced: ThermoPhase mutates a single authoritative
state; downstream quantities (species thermo, mixture properties, kinetics
rate constants, transport) are recomputed whenever the state changes
(see updateThermo / State::temperature etc.). A pure-Python port must provide
the same guarantee: no stale quantity can survive a state mutation.

This module provides the shared revision system for phase state:

* ``StateRevisions`` - independent monotonic counters per state aspect
  (temperature, pressure/density, composition, electric potential, coverage).
* ``RevisionCache`` - a cache whose entries are keyed by a revision snapshot;
  any state mutation changes the key, so stale entries can never be returned.
"""
from __future__ import annotations

from typing import Any, Callable, Hashable

ASPECTS = ('temperature', 'pressure', 'composition', 'potential', 'coverage')


class StateRevisions:
    """Monotonic revision counters for one authoritative phase state."""

    __slots__ = ('_counters',)

    def __init__(self):
        self._counters = {a: 0 for a in ASPECTS}

    def bump(self, aspect: str) -> int:
        """Record a mutation of *aspect* and return the new revision."""
        if aspect not in self._counters:
            raise KeyError(f'unknown state aspect {aspect!r}; valid: {ASPECTS}')
        self._counters[aspect] += 1
        return self._counters[aspect]

    def bump_temperature(self) -> int:
        return self.bump('temperature')

    def bump_pressure(self) -> int:
        return self.bump('pressure')

    def bump_composition(self) -> int:
        return self.bump('composition')

    def bump_potential(self) -> int:
        return self.bump('potential')

    def bump_coverage(self) -> int:
        return self.bump('coverage')

    @property
    def snapshot(self) -> tuple:
        """Immutable snapshot of all revisions; suitable as a cache key."""
        return tuple(self._counters[a] for a in ASPECTS)

    def __getitem__(self, aspect: str) -> int:
        return self._counters[aspect]


class RevisionCache:
    """Cache keyed by a :class:`StateRevisions` snapshot.

    ``cache.get(key, compute)`` recomputes whenever the tracked state has
    changed since the stored entry was computed. Entries from older revisions
    are not returned; they may remain stored but are unreachable.
    """

    __slots__ = ('_revisions', '_store')

    def __init__(self, revisions: StateRevisions):
        self._revisions = revisions
        self._store: dict[Hashable, Any] = {}

    @property
    def hits_revisions(self) -> tuple:
        return self._revisions.snapshot

    def get(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return the cached value for *key* at the current revision.

        Recomputes (and restores) when the state revision advanced, when the
        key is new, or when the stored value is not a scalar-compatible cache
        entry. A mutable computed value is copied in by the caller's compute
        function returning a fresh object each time.
        """
        stamp = (self._revisions.snapshot, key)
        if stamp in self._store:
            return self._store[stamp]
        value = compute()
        self._store[stamp] = value
        return value

    def clear(self) -> None:
        """Drop all cached entries (used when phase identity changes)."""
        self._store.clear()


class RevisionedState:
    """Mixin providing revision tracking to a phase implementation.

    The phase owns one authoritative ``StateRevisions``; state setters bump
    the relevant aspect; property computations use a ``RevisionCache`` so
    repeated reads without mutation are deterministic and mutations always
    invalidate.
    """

    def _init_revisions(self):
        self._revisions = StateRevisions()
        self._cache = RevisionCache(self._revisions)

    @property
    def state_revisions(self) -> StateRevisions:
        return self._revisions

    def _note_temperature_change(self):
        return self._revisions.bump_temperature()

    def _note_pressure_change(self):
        return self._revisions.bump_pressure()

    def _note_composition_change(self):
        return self._revisions.bump_composition()

    def _note_potential_change(self):
        return self._revisions.bump_potential()

    def _note_coverage_change(self):
        return self._revisions.bump_coverage()
