import os

oned_path = os.path.join(os.path.dirname(__file__), "..", "cantera_python", "oned.py")
with open(oned_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Boundary1D.__init__
b1d_target = """    def __init__(
        self,
        phase: Solution | None = None,
        name: str = "",
        **kwargs,
    ):
        if kwargs:
            first = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument '{first}'")"""

b1d_repl = """    def __init__(
        self,
        phase: Solution | None = None,
        name: str = "",
        **kwargs,
    ):
        if "id" in kwargs:
            name = kwargs.pop("id")
        if kwargs:
            first = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument '{first}'")"""

assert b1d_target in text, "b1d_target not found"
text = text.replace(b1d_target, b1d_repl, 1)

# 2. Update Inlet1D.__init__
inlet_target = """    def __init__(self, *args, name: str = "inlet", **kwargs):
        if len(args) > 0 and args[0] is None:
            raise TypeError("Argument 'phase' has incorrect type (expected Solution, got NoneType)")
        phase = args[0] if len(args) > 0 else kwargs.pop("phase", None)
        super().__init__(phase=phase, name=name, **kwargs)"""

inlet_repl = """    def __init__(self, *args, name: str = "inlet", **kwargs):
        if "id" in kwargs:
            name = kwargs.pop("id")
        if len(args) > 0 and args[0] is None:
            raise TypeError("Argument 'phase' has incorrect type (expected Solution, got NoneType)")
        phase = args[0] if len(args) > 0 else kwargs.pop("phase", None)
        super().__init__(phase=phase, name=name, **kwargs)"""

assert inlet_target in text, "inlet_target not found"
text = text.replace(inlet_target, inlet_repl, 1)

# 3. Update IonFlow.__init__ and add domain_type property
ionflow_target = """    def __init__(self, phase: Solution, id: str = "flame", points: int = 1):
        super().__init__(phase, id=id, points=points)
        self.m_speciesCharge: list[float] = [float(self.gas.charge(k)) for k in range(self.m_nsp)]"""

ionflow_repl = """    def __init__(self, phase: Solution, id: str = "flame", points: int = 1):
        super().__init__(phase, id=id, points=points)
        self.m_nsp = self.nsp
        self.m_speciesCharge: list[float] = [float(self.gas.charge(k)) for k in range(self.m_nsp)]"""

assert ionflow_target in text, "ionflow_target not found"
text = text.replace(ionflow_target, ionflow_repl, 1)

iontype_target = """    def domainType(self) -> str:
        if self.m_isFree:
            return "free-ion-flow"
        if self.m_usesLambda:
            return "axisymmetric-ion-flow"
        return "unstrained-ion-flow"

    def resize(self, components: int, points: int):"""

iontype_repl = """    def domainType(self) -> str:
        if self.m_isFree:
            return "free-ion-flow"
        if self.m_usesLambda:
            return "axisymmetric-ion-flow"
        return "unstrained-ion-flow"

    @property
    def domain_type(self) -> str:
        return self.domainType()

    def resize(self, components: int, points: int):"""

assert iontype_target in text, "iontype_target not found"
text = text.replace(iontype_target, iontype_repl, 1)

# 4. Update transport_model detection in IonFlow.__init__
iontrans_target = """        # Validate transport model
        trans_model = getattr(self.gas.transport, "transportModel", getattr(self.gas.transport, "transport_model", ""))
        if not trans_model or trans_model == "none":
            raise CanteraError("IonFlow::IonFlow",
                "An appropriate transport model\\nshould be set when instantiating the Solution ('gas') object.")"""

iontrans_repl = """        # Validate transport model
        trans_model = getattr(self.gas, "transport_model", "")
        if not trans_model:
            trans = getattr(self.gas, "transport", None)
            if trans is not None:
                tm_obj = getattr(trans, "transportModel", getattr(trans, "transport_model", ""))
                trans_model = tm_obj() if callable(tm_obj) else str(tm_obj)
        if not trans_model or trans_model == "none":
            raise CanteraError("IonFlow::IonFlow",
                "An appropriate transport model\\nshould be set when instantiating the Solution ('gas') object.")"""

assert iontrans_target in text, "iontrans_target not found"
text = text.replace(iontrans_target, iontrans_repl, 1)

# 5. Update DomainFactory transport check and creators
factory_target = """    def _register_default_creators(self):
        self.reg("inlet", lambda sol, id: Inlet1D(sol, id=id))
        self.reg("empty", lambda sol, id: Empty1D(sol, id=id))
        self.reg("symmetry-plane", lambda sol, id: Symm1D(sol, id=id))
        self.reg("outlet", lambda sol, id: Outlet1D(sol, id=id))
        self.reg("outlet-reservoir", lambda sol, id: OutletRes1D(sol, id=id))
        self.reg("surface", lambda sol, id: Surf1D(sol, id=id))
        self.reg("reacting-surface", lambda sol, id: ReactingSurf1D(sol, id=id))
        self.reg("gas-flow", lambda sol, id: Flow1D(sol, id=id))
        self.reg("ion-flow", lambda sol, id: IonFlow(sol, id=id))
        self.reg("free-flow", self._create_free_flow)
        self.reg("axisymmetric-flow", self._create_axisymmetric_flow)
        self.reg("unstrained-flow", self._create_unstrained_flow)

    def _create_free_flow(self, sol: Solution, id: str) -> Flow1D:
        model = getattr(sol.transport, "transportModel", getattr(sol.transport, "transport_model", ""))
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setFreeFlow()
        return flow

    def _create_axisymmetric_flow(self, sol: Solution, id: str) -> Flow1D:
        model = getattr(sol.transport, "transportModel", getattr(sol.transport, "transport_model", ""))
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setAxisymmetricFlow()
        return flow

    def _create_unstrained_flow(self, sol: Solution, id: str) -> Flow1D:
        model = getattr(sol.transport, "transportModel", getattr(sol.transport, "transport_model", ""))
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setUnstrainedFlow()
        return flow"""

factory_repl = """    @staticmethod
    def _transport_model(sol: Solution) -> str:
        tm = getattr(sol, "transport_model", "")
        if not tm:
            trans = getattr(sol, "transport", None)
            if trans is not None:
                tm_obj = getattr(trans, "transportModel", getattr(trans, "transport_model", ""))
                tm = tm_obj() if callable(tm_obj) else str(tm_obj)
        return tm

    def _register_default_creators(self):
        self.reg("inlet", lambda sol, id: Inlet1D(sol, id=id))
        self.reg("empty", lambda sol, id: Empty1D(sol, id=id))
        self.reg("symmetry-plane", lambda sol, id: Symm1D(sol, id=id))
        self.reg("outlet", lambda sol, id: Outlet1D(sol, id=id))
        self.reg("outlet-reservoir", lambda sol, id: OutletRes1D(sol, id=id))
        self.reg("surface", lambda sol, id: Surf1D(sol, id=id))
        self.reg("reacting-surface", lambda sol, id: ReactingSurf1D(sol, id=id))
        self.reg("gas-flow", lambda sol, id: Flow1D(sol, id=id))
        self.reg("ion-flow", lambda sol, id: IonFlow(sol, id=id))
        self.reg("free-flow", self._create_free_flow)
        self.reg("axisymmetric-flow", self._create_axisymmetric_flow)
        self.reg("unstrained-flow", self._create_unstrained_flow)

    def _create_free_flow(self, sol: Solution, id: str) -> Flow1D:
        model = self._transport_model(sol)
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setFreeFlow()
        return flow

    def _create_axisymmetric_flow(self, sol: Solution, id: str) -> Flow1D:
        model = self._transport_model(sol)
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setAxisymmetricFlow()
        return flow

    def _create_unstrained_flow(self, sol: Solution, id: str) -> Flow1D:
        model = self._transport_model(sol)
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setUnstrainedFlow()
        return flow"""

assert factory_target in text, "factory_target not found"
text = text.replace(factory_target, factory_repl, 1)

with open(oned_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Successfully applied updates to oned.py.")
