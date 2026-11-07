"""Tests for Family 3 Kinetics failure semantics and input validation."""

import pytest
import numpy as np
from pathlib import Path
from cantera_python.surface import Interface, IdealSurfacePhase, InterfaceKinetics
from cantera_python.constants import CanteraError, InputFileError

ROOT = Path(__file__).resolve().parents[3]
MECH_PATH = ROOT / "outputs" / "cantera-original" / "data" / "ptcombust.yaml"

def test_invalid_site_density():
    with pytest.raises((ValueError, CanteraError)):
        IdealSurfacePhase(species=[], site_density=-1.0)
    with pytest.raises((ValueError, CanteraError)):
        IdealSurfacePhase(species=[], site_density=0.0)

def test_invalid_coverage_assignment():
    surf = Interface(str(MECH_PATH), 'Pt_surf')
    with pytest.raises((ValueError, CanteraError, KeyError)):
        surf.coverages = "NONEXISTENT_SPECIES:1.0"

def test_species_index_out_of_bounds():
    surf = Interface(str(MECH_PATH), 'Pt_surf')
    with pytest.raises((ValueError, IndexError, KeyError)):
        surf.kinetics_species_index("UNRECOGNIZED_SPECIES")

def test_multiplier_out_of_bounds_handling():
    surf = Interface(str(MECH_PATH), 'Pt_surf')
    kin = surf.kinetics
    # Multiplier on non-existent reaction should not crash but return default 1.0
    kin.setMultiplier(9999, 2.5)
    assert kin.multiplier(9999) == 2.5
    assert kin.multiplier(0) == 1.0

def test_invalid_phase_index():
    surf = Interface(str(MECH_PATH), 'Pt_surf')
    kin = surf.kinetics
    with pytest.raises((IndexError, ValueError)):
        # Accessing out of range phase index
        _ = surf._phases[99]

def test_negative_timestep_advance():
    surf = Interface(str(MECH_PATH), 'Pt_surf')
    cov_before = np.copy(surf.coverages)
    surf.advanceCoverages(-1.0)
    # Negative dt should be a no-op
    np.testing.assert_allclose(surf.coverages, cov_before)
