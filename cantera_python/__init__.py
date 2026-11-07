"""An incomplete, Cantera-independent Python implementation of selected models.

Runtime equations and solver orchestration are implemented in this package;
NumPy and SciPy provide numerical array and integration routines. No native
Cantera module is imported. See README.md and provenance/ for supported scope.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / 'cantera-original' / 'data'

from .constants import GasConstant, OneAtm, CanteraError
one_atm = OneAtm
from .composition import CompositionError, SpeciesRegistry, parse_comp_string
from .elements import COMMON_ELEMENTS, Element, ElementError, ElementRegistry
from .reaction_equation import (ParsedReactionEquation, ReactionEquationError,
                                are_duplicate_reactions, duplicate_stoichiometry_ratio,
                                parse_reaction_equation)
from .units import (UnitError, Units, UnitStack, UnitSystem, convert,
                    convert_activation_energy, parse_units)
from .mechanism import (Mechanism, MechanismError, MechanismPhase, MechanismReaction,
                        MechanismSpecies, load_mechanism)
from .ideal_gas import IdealGasMixture, IdealGasPhase, Species
from .cubic_eos import (MixtureFugacityTP, PengRobinson, RedlichKwongMFTP,
                        FLUID_UNSTABLE, FLUID_UNDEFINED, FLUID_SUPERCRIT,
                        FLUID_GAS, FLUID_LIQUID_0)
from .surface import (IdealSurfacePhase, SurfaceSpecies, Interface,
                    InterfaceReaction, InterfaceKinetics)
from .ideal_solid import IdealSolidSolution
from .phases import (SingleSpeciesTP, SurfPhase, EdgePhase, MetalPhase,
                    IdealSolidSolnPhase, CoverageDependentSurfPhase)
from .stoich import StoichSubstance

from .stoich import StoichiometricSubstance
from .fixed_stoichiometry import FixedStoichiometryPhase
from .equilibrium import (EquilibriumError, IdealGasEquilibrium,
                          ReactionStoichiometry, equilibrate_ideal_gas, ChemEquil)
from .state import (set_state_td, set_state_hp, set_state_sp, set_state_uv,
                    set_state_tp, set_state_dp, set_state_sv)
from .thermo import (NasaPoly1, NasaPoly2, Nasa9Poly1,
                     Nasa9PolyMultiTempRegion, ShomatePoly, ShomatePoly2,
                     ConstCpPoly, Mu0Poly, create_species_thermo)
from .multi_species_thermo import MultiSpeciesThermo
from .rates import (ArrheniusRate, TwoTempPlasmaRate, BlowersMaselRate,
                    ElectronCollisionPlasmaRate,
                    LindemannRate, TroeRate, SriRate, TsangRate,
                    PlogRate, ChebyshevRate, LinearBurkeCollider, LinearBurkeRate)
from .rates import (CustomFunc1Rate,
                    VibrationalRelaxationRate, ConstantVibrationalRelaxationRate,
                    CastelaVibrationalRelaxationRate, StarikovskiyVibrationalRelaxationRate,
                    MultiStateResolvedVibrationalRelaxationRate,
                    InterfaceArrheniusRate, StickingArrheniusRate,
                    InterfaceBlowersMaselRate, StickingBlowersMaselRate,
                    ReactionRateFactory, newReactionRate)
from .kinetics import (ElementaryReaction, FalloffReaction, ThirdBody,
                     GasKinetics, Kinetics, BulkKinetics, ReactionPathDiagram,
                     Reaction, Group, SpeciesNode, Path, ReactionPathBuilder,
                     KineticsFactory, newKinetics, newReactionPathDiagram,
                     EdgeKinetics, create_reaction)
from .transport import (DustyGasTransport, GasTransportData, IonGasTransport,
                        MixTransport, MultiTransport, UnityLewisTransport, HighPressureGasTransportBase, HighPressureGasTransport, ChungHighPressureGasTransport)
from .reactor import (IdealGasReactor, IdealGasConstPressureReactor, ReactorNet,
                      Reservoir, Wall, FlowDevice, MassFlowController, Valve, PressureController,
                      Reactor, ConstPressureReactor, FlowReactor, ReactorSurface, ReactorDelegator,
                      MoleReactor, IdealGasMoleReactor, ConstPressureMoleReactor)
from .solution import DustyGas, Solution
from .solution_array import SolutionArray
from .oned import (Domain1D, Boundary1D, Inlet1D, Outlet1D, OutletRes1D, Empty1D,
                   Symm1D, SymmetryPlane1D, Surf1D, Surface1D,
                   ReactingSurf1D, ReactingSurface1D,
                   Flow1D, FreeFlow, UnstrainedFlow, AxisymmetricFlow, IonFlow,
                   Refiner, OneDim, Sim1D, FlameBase, FreeFlame, BurnerFlame,
                   CounterflowDiffusionFlame,
                   DomainFactory, newDomain, newBoundary1D, newFlow1D)
from .vcs import MultiPhase, vcs_equilibrate, VCSError, vcs_VolPhase, VCS_SOLVE, MultiPhaseEquil
from .tpx import (Substance, Water, CarbonDioxide, Nitrogen, Oxygen,
                  Hydrogen, Methane, Heptane, HFC134a, newSubstance, TPXError)
from .electrolytes import WaterProps, DebyeHuckel, HMWSoln, ElectrolyteError
from .numerics import (GeneralMatrix, BandMatrix, DenseMatrix, SystemJacobian,
                       AdaptivePreconditioner, FuncEval, Integrator,
                       CVodesIntegrator, IdasIntegrator, NumericsError)
from .vpss import (VPStandardStateTP, MolalityVPSSTP, MargulesVPSSTP,
                   RedlichKisterVPSSTP, PDSS_HKF, LatticeSolidPhase, VPSSError)
from .anymap import AnyBase, AnyValue, AnyMap, Application, AnyMapError

__version__ = '0.3.0'
UPSTREAM_COMMIT = '726522be4e2a13454d8415b7ef799d621f665cf3'
PORT_COMPLETE = False

