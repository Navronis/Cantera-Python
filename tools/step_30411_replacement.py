class Group:
    """Atomic group for reaction path diagrams.

    Replicates Cantera's Group class in include/cantera/kinetics/Group.h
    """
    def __init__(self, elements=None):
        if elements is None:
            self.m_comp = []
            self.m_sign = -999
        elif isinstance(elements, int):
            self.m_comp = [0] * elements
            self.m_sign = 0
        elif isinstance(elements, Group):
            self.m_comp = list(elements.m_comp)
            self.m_sign = elements.m_sign
        elif isinstance(elements, (list, tuple, np.ndarray)):
            self.m_comp = [int(x) for x in elements]
            self.m_sign = 0
            self.validate()
        elif isinstance(elements, dict):
            self.m_comp = [int(v) for v in elements.values()]
            self.comp = dict(elements)
            self.m_sign = 0
            self.validate()
        else:
            self.m_comp = [int(x) for x in elements]
            self.m_sign = 0
            self.validate()

    def validate(self) -> None:
        if self.m_sign == -999:
            return
        self.m_sign = 0
        ok = True
        for m in range(len(self.m_comp)):
            val = self.m_comp[m]
            if val != 0:
                if self.m_sign == 0:
                    self.m_sign = 1 if val > 0 else -1
                elif self.m_sign * val < 0:
                    ok = False
                    break
        if not ok:
            self.m_sign = -999

    def valid(self) -> bool:
        return self.m_sign != -999

    def __bool__(self) -> bool:
        return self.valid()

    def sign(self) -> int:
        return int(self.m_sign)

    def size(self) -> int:
        return len(self.m_comp)

    def nAtoms(self, m=None) -> int:
        if m is None:
            return sum(abs(x) for x in self.m_comp)
        if not self.m_comp or m >= len(self.m_comp):
            return 0
        return self.m_comp[m]

    def __iadd__(self, other):
        for m in range(min(len(self.m_comp), len(other.m_comp))):
            self.m_comp[m] += other.m_comp[m]
        self.validate()
        return self

    def __isub__(self, other):
        for m in range(min(len(self.m_comp), len(other.m_comp))):
            self.m_comp[m] -= other.m_comp[m]
        self.validate()
        return self

    def __imul__(self, a: int):
        for m in range(len(self.m_comp)):
            self.m_comp[m] *= a
        self.validate()
        return self

    def sum(self, other):
        res = Group(self)
        res += other
        return res

    def __add__(self, other):
        return self.sum(other)

    def diff(self, other):
        res = Group(self)
        res -= other
        return res

    def __sub__(self, other):
        return self.diff(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Group):
            return False
        return self.m_comp == other.m_comp

    def fmt(self, esymbols=None) -> str:
        s = "("
        first = True
        for m in range(len(self.m_comp)):
            nm = self.m_comp[m]
            if nm != 0:
                if not first:
                    s += "-"
                sym = esymbols[m] if (esymbols is not None and m < len(esymbols)) else f"E{m}"
                s += sym
                if nm != 1:
                    s += str(nm)
                first = False
        s += ")"
        return s


class SpeciesNode:
    """Node representing a species in a reaction path diagram.

    Replicates SpeciesNode in include/cantera/kinetics/ReactionPath.h
    """
    def __init__(self, number: int = -1, name: str = "", value: float = 0.0):
        self.number = number
        self.name = name
        self.value = float(value)
        self.visible = False
        self.m_in = 0.0
        self.m_out = 0.0
        self.m_paths: List["Path"] = []

    def path(self, n: int):
        if 0 <= n < len(self.m_paths):
            return self.m_paths[n]
        return None

    def nPaths(self) -> int:
        return len(self.m_paths)

    def addPath(self, path: "Path") -> None:
        self.m_paths.append(path)
        if path.begin() is self:
            self.m_out += path.flow()
        elif path.end() is self:
            self.m_in += path.flow()
        else:
            raise ValueError("SpeciesNode.addPath: path added to wrong node")

    def outflow(self) -> float:
        return float(self.m_out)

    def inflow(self) -> float:
        return float(self.m_in)

    def netOutflow(self) -> float:
        return float(self.m_out - self.m_in)

    def printPaths(self) -> None:
        for p in self.m_paths:
            b = p.begin().name if p.begin() else ""
            e = p.end().name if p.end() else ""
            print(f"{b} -->  {e}:   {p.flow()}")


class Path:
    """Path connecting two species nodes in a reaction path diagram.

    Replicates Path in include/cantera/kinetics/ReactionPath.h
    """
    def __init__(self, begin: SpeciesNode, end: SpeciesNode):
        self.m_a = begin
        self.m_b = end
        self.m_total = 0.0
        self.m_rxn: Dict[int, float] = {}
        self.m_label: Dict[str, float] = {}
        if begin is not None:
            begin.addPath(self)
        if end is not None:
            end.addPath(self)

    def begin(self) -> SpeciesNode:
        return self.m_a

    def end(self) -> SpeciesNode:
        return self.m_b

    def otherNode(self, n: SpeciesNode) -> Optional[SpeciesNode]:
        if n is self.m_a:
            return self.m_b
        elif n is self.m_b:
            return self.m_a
        return None

    def flow(self) -> float:
        return float(self.m_total)

    def setFlow(self, v: float) -> None:
        self.m_total = float(v)

    def nReactions(self) -> int:
        return len(self.m_rxn)

    def reactionMap(self) -> Dict[int, float]:
        return dict(self.m_rxn)

    def addReaction(self, rxnNumber: int, value: float, label: str = "") -> None:
        v = float(value)
        self.m_rxn[int(rxnNumber)] = self.m_rxn.get(int(rxnNumber), 0.0) + v
        self.m_total += v
        if label:
            self.m_label[label] = self.m_label.get(label, 0.0) + v

    def writeLabel(self, threshold: float = 0.005) -> str:
        if not self.m_label or self.m_total <= 0.0:
            return ""
        s = ""
        for species, value in sorted(self.m_label.items()):
            v = value / self.m_total
            if len(self.m_label) == 1:
                s += f"{species}\\l"
            elif v > threshold:
                s += species
                percent = int(100.0 * v + 0.5)
                if percent < 100:
                    s += f" ({percent}%)\\l"
                else:
                    s += "\\l"
        return s


class ReactionPathBuilder:
    """Reaction path diagram builder.

    Replicates ReactionPathBuilder in include/cantera/kinetics/ReactionPath.h
    and src/kinetics/ReactionPath.cpp.
    """
    def __init__(self):
        self.m_nr = 0
        self.m_ns = 0
        self.m_nel = 0
        self.m_ropf = []
        self.m_ropr = []
        self.m_x = []
        self.m_reac = []
        self.m_prod = []
        self.m_elatoms = None
        self.m_atoms = None
        self.m_sgroup = []
        self.m_elementSymbols = []
        self.m_enamemap = {}
        self.m_determinate = []
        self.m_transfer = {}
        self._log = ""

    def findElements(self, kin) -> None:
        self.m_enamemap.clear()
        self.m_elementSymbols.clear()
        self.m_nel = 0
        n_phases = kin.nPhases() if callable(getattr(kin, "nPhases", None)) else 1
        for ip in range(n_phases):
            p = kin.thermo(ip) if callable(getattr(kin, "thermo", None)) else getattr(kin, "thermo", kin)
            n_elem = p.nElements() if callable(getattr(p, "nElements", None)) else len(p.element_names)
            for m in range(n_elem):
                ename = p.elementName(m) if callable(getattr(p, "elementName", None)) else p.element_names[m]
                if ename not in self.m_enamemap:
                    self.m_elementSymbols.append(ename)
                    self.m_enamemap[ename] = self.m_nel + 1
                    self.m_nel += 1

        n_tot_sp = kin.nTotalSpecies() if callable(getattr(kin, "nTotalSpecies", None)) else len(kin.species_names)
        self.m_atoms = np.zeros((n_tot_sp, self.m_nel), dtype=float)
        for m in range(self.m_nel):
            k = 0
            for ip in range(n_phases):
                p = kin.thermo(ip) if callable(getattr(kin, "thermo", None)) else getattr(kin, "thermo", kin)
                mlocal = p.elementIndex(self.m_elementSymbols[m]) if callable(getattr(p, "elementIndex", None)) else None
                if mlocal is None and hasattr(p, "element_names") and self.m_elementSymbols[m] in p.element_names:
                    mlocal = p.element_names.index(self.m_elementSymbols[m])
                if mlocal is not None and mlocal >= 0:
                    n_sp = p.nSpecies() if callable(getattr(p, "nSpecies", None)) else len(p.species_names)
                    for kp in range(n_sp):
                        if hasattr(p, "nAtoms") and callable(p.nAtoms):
                            self.m_atoms[k, m] = p.nAtoms(kp, mlocal)
                        elif hasattr(p, "n_atoms") and callable(p.n_atoms):
                            self.m_atoms[k, m] = p.n_atoms(p.species_names[kp], self.m_elementSymbols[m])
                        k += 1

    def init(self, kin, logfile=None) -> int:
        self.m_transfer.clear()
        self.m_elementSymbols.clear()
        self.findElements(kin)
        self.m_ns = kin.nTotalSpecies() if callable(getattr(kin, "nTotalSpecies", None)) else len(kin.species_names)
        self.m_nr = kin.nReactions() if callable(getattr(kin, "nReactions", None)) else len(kin.reactions)

        allProducts = [[] for _ in range(self.m_nr)]
        allReactants = [[] for _ in range(self.m_nr)]

        for i in range(self.m_nr):
            rxn = kin.reaction(i) if callable(getattr(kin, "reaction", None)) else kin.reactions[i]
            for k in range(self.m_ns):
                sp_name = kin.kineticsSpeciesName(k) if callable(getattr(kin, "kineticsSpeciesName", None)) else kin.species_names[k]
                if hasattr(kin, "reactantStoichCoeff") and callable(kin.reactantStoichCoeff):
                    r_coeff = int(round(kin.reactantStoichCoeff(k, i)))
                else:
                    r_coeff = int(round(rxn.reactants.get(sp_name, 0.0)))
                for _ in range(r_coeff):
                    allReactants[i].append(k)

                if hasattr(kin, "productStoichCoeff") and callable(kin.productStoichCoeff):
                    p_coeff = int(round(kin.productStoichCoeff(k, i)))
                else:
                    p_coeff = int(round(rxn.products.get(sp_name, 0.0)))
                for _ in range(p_coeff):
                    allProducts[i].append(k)

        self.m_reac = [[] for _ in range(self.m_nr)]
        self.m_prod = [[] for _ in range(self.m_nr)]
        self.m_elatoms = np.zeros((self.m_nel, self.m_nr), dtype=float)
        self.m_determinate = [True] * self.m_nr
        self.m_x = [0.0] * self.m_ns

        for i in range(self.m_nr):
            net = {}
            for r_sp in allReactants[i]:
                net[r_sp] = net.get(r_sp, 0) - 1
            for p_sp in allProducts[i]:
                net[p_sp] = net.get(p_sp, 0) + 1

            for k in range(self.m_ns):
                val = net.get(k, 0)
                if val < 0:
                    for _ in range(-val):
                        self.m_reac[i].append(k)
                elif val > 0:
                    for _ in range(val):
                        self.m_prod[i].append(k)

            for k in self.m_reac[i]:
                for m in range(self.m_nel):
                    self.m_elatoms[m, i] += self.m_atoms[k, m]

        self.m_sgroup = []
        for j in range(self.m_ns):
            comp = [int(self.m_atoms[j, m]) for m in range(self.m_nel)]
            self.m_sgroup.append(Group(comp))

        for i in range(self.m_nr):
            nr = len(self.m_reac[i])
            np_ = len(self.m_prod[i])
            self.m_determinate[i] = True
            for m in range(self.m_nel):
                nar = sum(1 for j in range(nr) if self.m_atoms[self.m_reac[i][j], m] > 0)
                nap = sum(1 for j in range(np_) if self.m_atoms[self.m_prod[i][j], m] > 0)
                if nar > 1 and nap > 1:
                    self.m_determinate[i] = False
                    break

        self.findGroups(logfile, kin)
        return 1

    def findGroups(self, logfile, s) -> int:
        log_lines = []
        for i in range(self.m_nr):
            rxn = s.reaction(i) if callable(getattr(s, "reaction", None)) else s.reactions[i]
            eq = rxn.equation() if callable(getattr(rxn, "equation", None)) else getattr(rxn, "equation", "")
            log_lines.append(f"\nReaction {i+1}: {eq}")

            if self.m_determinate[i]:
                log_lines.append(" ... OK.")
            elif len(self.m_reac[i]) == 2 and len(self.m_prod[i]) == 2:
                kr0, kr1 = self.m_reac[i][0], self.m_reac[i][1]
                kp0, kp1 = self.m_prod[i][0], self.m_prod[i][1]
                r0, r1 = self.m_sgroup[kr0], self.m_sgroup[kr1]
                p0, p1 = self.m_sgroup[kp0], self.m_sgroup[kp1]

                b0 = p0 - r0
                b1 = p1 - r0
                if b0.valid() and b1.valid():
                    log_lines.append(" ... ambiguous.")
                elif not b0.valid() and not b1.valid():
                    log_lines.append(" ... cannot express as A + BC = AB + C")

                if b0.valid():
                    if b0.sign() > 0:
                        group_a0, group_b0, group_c0 = r0, b0, p1
                        self.m_transfer.setdefault(i, {})
                        self.m_transfer[i].setdefault(0, {})[0] = r0
                        self.m_transfer[i].setdefault(1, {})[0] = b0
                        self.m_transfer[i].setdefault(1, {})[1] = p1
                    else:
                        b0_inv = Group(b0)
                        b0_inv *= -1
                        group_a0, group_b0, group_c0 = r1, b0_inv, p0
                        self.m_transfer.setdefault(i, {})
                        self.m_transfer[i].setdefault(1, {})[1] = r1
                        self.m_transfer[i].setdefault(0, {})[1] = b0_inv
                        self.m_transfer[i].setdefault(0, {})[0] = p0

                    d_str = "   [<= default]" if b1.valid() else ""
                    log_lines.append(f"     {group_a0.fmt(self.m_elementSymbols)} + {group_b0.fmt(self.m_elementSymbols)}{group_c0.fmt(self.m_elementSymbols)} = {group_a0.fmt(self.m_elementSymbols)}{group_b0.fmt(self.m_elementSymbols)} + {group_c0.fmt(self.m_elementSymbols)}{d_str}")

                if b1.valid():
                    if b1.sign() > 0:
                        group_a1, group_b1, group_c1 = r0, b1, p0
                        if not b0.valid():
                            self.m_transfer.setdefault(i, {})
                            self.m_transfer[i].setdefault(0, {})[1] = r0
                            self.m_transfer[i].setdefault(1, {})[1] = b0
                            self.m_transfer[i].setdefault(1, {})[0] = p0
                    else:
                        b1_inv = Group(b1)
                        b1_inv *= -1
                        group_a1, group_b1, group_c1 = r1, b1_inv, p1
                        if not b0.valid():
                            self.m_transfer.setdefault(i, {})
                            self.m_transfer[i].setdefault(1, {})[0] = r1
                            self.m_transfer[i].setdefault(0, {})[0] = b0
                            self.m_transfer[i].setdefault(0, {})[1] = p1
                    log_lines.append(f"     {group_a1.fmt(self.m_elementSymbols)} + {group_b1.fmt(self.m_elementSymbols)}{group_c1.fmt(self.m_elementSymbols)} = {group_a1.fmt(self.m_elementSymbols)}{group_b1.fmt(self.m_elementSymbols)} + {group_c1.fmt(self.m_elementSymbols)}")
            else:
                log_lines.append("... cannot parse. [ignored]")

        self._log = "\n".join(log_lines)
        return 1

    def _reactionLabel(self, i: int, kr: int, nr: int, slist: list, s) -> str:
        label = ""
        for j in range(nr):
            if j != kr:
                name = s.kineticsSpeciesName(slist[j]) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[slist[j]]
                label += f" + {name}"
        rxn = s.reaction(i) if callable(getattr(s, "reaction", None)) else s.reactions[i]
        rtype = rxn.type() if callable(getattr(rxn, "type", None)) else getattr(rxn, "type", "")
        if str(rtype).startswith("three-body"):
            label += " + M "
        elif str(rtype).startswith("falloff"):
            label += " (+ M)"
        return label

    def build(self, s, element: str, r, quiet: bool = True) -> int:
        if element not in self.m_enamemap:
            return -1
        m = self.m_enamemap[element] - 1
        r.element = element

        ropf = s.forward_rates_of_progress if hasattr(s, "forward_rates_of_progress") else s.getFwdRatesOfProgress()
        if callable(ropf):
            ropf = ropf()
        ropr = s.reverse_rates_of_progress if hasattr(s, "reverse_rates_of_progress") else s.getRevRatesOfProgress()
        if callable(ropr):
            ropr = ropr()

        in_nodes = r.included()
        out_nodes = r.excluded()

        n_tot_sp = s.nTotalSpecies() if callable(getattr(s, "nTotalSpecies", None)) else len(s.species_names)
        status = [0] * n_tot_sp
        for name in in_nodes:
            idx = s.kineticsSpeciesIndex(name) if callable(getattr(s, "kineticsSpeciesIndex", None)) else s.species_names.index(name)
            if idx >= 0:
                status[idx] = 1
        for name in out_nodes:
            idx = s.kineticsSpeciesIndex(name) if callable(getattr(s, "kineticsSpeciesIndex", None)) else s.species_names.index(name)
            if idx >= 0:
                status[idx] = -1

        threshold = 0.0

        for i in range(self.m_nr):
            ropf_i = float(ropf[i])
            ropr_i = float(ropr[i])

            if self.m_elatoms[m, i] > 0:
                nr = len(self.m_reac[i])
                np_ = len(self.m_prod[i])

                for kr in range(nr):
                    kkr = self.m_reac[i][kr]
                    fwdlabel = self._reactionLabel(i, kr, nr, self.m_reac[i], s)

                    for kp in range(np_):
                        kkp = self.m_prod[i][kp]
                        revlabel = ""
                        for j in range(np_):
                            if j != kp:
                                name = s.kineticsSpeciesName(self.m_prod[i][j]) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[self.m_prod[i][j]]
                                revlabel += f" + {name}"
                        rxn = s.reaction(i) if callable(getattr(s, "reaction", None)) else s.reactions[i]
                        rtype = rxn.type() if callable(getattr(rxn, "type", None)) else getattr(rxn, "type", "")
                        if str(rtype).startswith("three-body"):
                            revlabel += " + M "
                        elif str(rtype).startswith("falloff"):
                            revlabel += " (+ M)"

                        if (kkr != kkp and self.m_atoms[kkr, m] > 0 and self.m_atoms[kkp, m] > 0
                                and status[kkr] >= 0 and status[kkp] >= 0):
                            if (self.m_atoms[kkp, m] < self.m_elatoms[m, i]) and (self.m_atoms[kkr, m] < self.m_elatoms[m, i]):
                                g = self.m_transfer.get(i, {})
                                if not g:
                                    f = 0.0
                                else:
                                    transfer_grp = g.get(kr, {}).get(kp)
                                    if not transfer_grp or not transfer_grp.valid():
                                        f = 0.0
                                    else:
                                        f = float(transfer_grp.nAtoms(m))
                            else:
                                f = float(self.m_atoms[kkp, m] * self.m_atoms[kkr, m] / self.m_elatoms[m, i])

                            fwd = ropf_i * f
                            rev = ropr_i * f
                            force_incl = (status[kkr] == 1 or status[kkp] == 1)

                            fwd_incl = (fwd > threshold) or (fwd > 0.0 and force_incl)
                            rev_incl = (rev > threshold) or (rev > 0.0 and force_incl)

                            if fwd_incl or rev_incl:
                                if not r.hasNode(kkr):
                                    sp_name = s.kineticsSpeciesName(kkr) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[kkr]
                                    r.addNode(kkr, sp_name, self.m_x[kkr])
                                if not r.hasNode(kkp):
                                    sp_name = s.kineticsSpeciesName(kkp) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[kkp]
                                    r.addNode(kkp, sp_name, self.m_x[kkp])

                            if fwd_incl:
                                r.linkNodes(kkr, kkp, int(i), fwd, fwdlabel)
                            if rev_incl:
                                r.linkNodes(kkp, kkr, -int(i), rev, revlabel)

        return 1


class ReactionPathDiagram:
    """Reaction path analysis diagram representation and graph exporter.

    Replicates Cantera's ReactionPathDiagram in include/cantera/kinetics/ReactionPath.h
    and src/kinetics/ReactionPath.cpp
    """
    def __init__(self, kinetics=None, element: str = "H"):
        self.kinetics = kinetics
        self.element = str(element)
        self.title = ""
        self.bold_color = "blue"
        self.normal_color = "steelblue"
        self.dashed_color = "gray"
        self.dot_options = ""
        self.bold_threshold = 0.2
        self.normal_threshold = 0.05
        self.label_threshold = 0.001
        self.threshold = 0.005
        self.scale = -1.0
        self.arrow_width = -1.0
        self.show_details = False
        self.x_size = -1.0
        self.y_size = -1.0
        self.flow_type = "NetFlow"
        self.m_font = "Helvetica"
        self.m_local = None
        self.m_include = []
        self.m_exclude = []
        self.m_nodes: Dict[int, SpeciesNode] = {}
        self.m_speciesNumber: List[int] = []
        self.m_paths: Dict[int, Dict[int, Path]] = {}
        self.m_pathlist: List[Path] = []
        self.m_rxns = set()
        self.m_flxmax = 0.0
        self.m_isBuilt = False
        self.m_builder = ReactionPathBuilder()
        if kinetics is not None:
            self.m_builder.init(self.kinetics)

    def maxFlow(self) -> float:
        return float(self.m_flxmax)

    def flow(self, k1: int, k2: int) -> float:
        p = self.m_paths.get(k1, {}).get(k2)
        return float(p.flow()) if p is not None else 0.0

    def netFlow(self, k1: int, k2: int) -> float:
        return self.flow(k1, k2) - self.flow(k2, k1)

    def hasNode(self, k: int) -> bool:
        return k in self.m_nodes

    def node(self, k: int) -> Optional[SpeciesNode]:
        return self.m_nodes.get(k)

    def path(self, k1_or_n: int, k2: int = None):
        if k2 is None:
            return self.m_pathlist[k1_or_n] if 0 <= k1_or_n < len(self.m_pathlist) else None
        return self.m_paths.get(k1_or_n, {}).get(k2)

    def nPaths(self) -> int:
        return len(self.m_pathlist)

    def nNodes(self) -> int:
        return len(self.m_nodes)

    def addNode(self, k: int, nm: str, x: float = 0.0) -> None:
        if k not in self.m_nodes:
            nd = SpeciesNode(number=k, name=nm, value=x)
            self.m_nodes[k] = nd
            self.m_speciesNumber.append(k)

    def linkNodes(self, k1: int, k2: int, rxn: int, value: float, legend: str = "") -> None:
        ff = self.m_paths.get(k1, {}).get(k2)
        if ff is None:
            ff = Path(self.m_nodes[k1], self.m_nodes[k2])
            self.m_paths.setdefault(k1, {})[k2] = ff
            self.m_pathlist.append(ff)
        ff.addReaction(rxn, value, legend)
        self.m_rxns.add(rxn)
        self.m_flxmax = max(ff.flow(), self.m_flxmax)

    def include(self, name) -> None:
        if isinstance(name, (list, tuple)):
            self.m_include.extend(name)
        else:
            self.m_include.append(str(name))

    def exclude(self, name) -> None:
        if isinstance(name, (list, tuple)):
            self.m_exclude.extend(name)
        else:
            self.m_exclude.append(str(name))

    def included(self) -> list:
        return list(self.m_include)

    def excluded(self) -> list:
        return list(self.m_exclude)

    def species(self) -> list:
        return list(self.m_speciesNumber)

    def reactions(self) -> list:
        flmax = 0.0
        for p in self.m_pathlist:
            flmax = max(p.flow(), flmax)
        rxns = set()
        for p in self.m_pathlist:
            for iRxn, flux in p.reactionMap().items():
                if flmax > 0.0 and (flux / flmax) > self.threshold:
                    rxns.add(iRxn)
        return sorted(rxns)

    def displayOnly(self, k: int = -1) -> None:
        self.m_local = k if k >= 0 else None

    def display_only(self, k: int = -1) -> None:
        self.displayOnly(k)

    def setFont(self, font: str) -> None:
        self.m_font = str(font)

    def flowType(self) -> str:
        return str(self.flow_type)

    def setFlowType(self, fType: str) -> None:
        if fType not in ("NetFlow", "OneWayFlow"):
            raise ValueError(f"Unknown flow type '{fType}'")
        self.flow_type = fType

    def build(self, verbose: bool = False) -> None:
        self.m_builder.build(self.kinetics, self.element, self, quiet=not verbose)
        self.m_isBuilt = True

    def getLog(self) -> str:
        return getattr(self.m_builder, '_log', '')

    @property
    def log(self) -> str:
        return self.getLog()

    def getData(self) -> str:
        if not self.m_isBuilt:
            self.build()
        lines = [self.title]
        names = [self.m_nodes[k].name for k in self.m_speciesNumber]
        lines.append(" ".join(names) + " ")
        n_nodes = len(self.m_speciesNumber)
        for i1 in range(n_nodes):
            k1 = self.m_speciesNumber[i1]
            for i2 in range(i1 + 1, n_nodes):
                k2 = self.m_speciesNumber[i2]
                f1 = self.flow(k1, k2)
                f2 = self.flow(k2, k1)
                lines.append(f"{self.m_nodes[k1].name} {self.m_nodes[k2].name} {f1:g} {-f2:g}")
        return "\n".join(lines) + "\n"

    def get_data(self) -> str:
        return self.getData()

    def writeData(self, s) -> None:
        text = self.getData()
        if hasattr(s, "write"):
            s.write(text)
        elif isinstance(s, (str, Path)):
            with open(s, "w", encoding="utf-8") as f:
                f.write(text)

    def exportToDot(self, s=None) -> str:
        if not self.m_isBuilt:
            self.build()
        lines = []
        flmax = 0.0
        lines.append(f"digraph {self.title or 'reaction_paths'} {{")
        lines.append("center=1;")

        if self.x_size > 0.0:
            ys = self.x_size if self.y_size < 0.0 else self.y_size
            lines.append(f'size = "{self.x_size:g},{ys:g}";')

        if self.dot_options:
            lines.append(self.dot_options)

        if self.flow_type == "NetFlow":
            if self.scale <= 0.0:
                for i1 in range(self.nNodes()):
                    k1 = self.m_speciesNumber[i1]
                    self.node(k1).visible = False
                    for i2 in range(i1 + 1, self.nNodes()):
                        k2 = self.m_speciesNumber[i2]
                        flx = abs(self.netFlow(k1, k2))
                        flmax = max(flx, flmax)
            else:
                flmax = self.scale
            flmax = max(flmax, 1e-10)

            for i1 in range(self.nNodes()):
                k1 = self.m_speciesNumber[i1]
                for i2 in range(i1 + 1, self.nNodes()):
                    k2 = self.m_speciesNumber[i2]
                    flx = self.netFlow(k1, k2)
                    if self.m_local is not None and k1 != self.m_local and k2 != self.m_local:
                        flx = 0.0
                    if flx != 0.0:
                        if flx > 0.0:
                            kbegin, kend = k1, k2
                            flxratio = flx / flmax
                        else:
                            kbegin, kend = k2, k1
                            flxratio = -flx / flmax

                        if flxratio >= self.threshold:
                            self.node(kbegin).visible = True
                            self.node(kend).visible = True
                            edge_str = f"s{kbegin} -> s{kend}[fontname=\"{self.m_font}\""
                            if self.arrow_width < 0:
                                import math
                                lwidth = 1.0 - 4.0 * math.log10(flxratio / self.threshold) / math.log10(self.threshold) + 1.0
                                edge_str += f", penwidth={lwidth:.3g}, arrowsize={min(6.0, 0.5 * lwidth):.3g}"
                            else:
                                edge_str += f", penwidth={self.arrow_width:g}, arrowsize={flxratio + 1:.3g}"

                            edge_str += f', color="0.7, {flxratio + 0.5:.3g}, 0.9"\n'
                            if flxratio > self.label_threshold:
                                edge_str += f', label=" {flxratio:.3g}"'
                                if self.show_details:
                                    edge_str += "\\l"
                            edge_str += "];"
                            lines.append(edge_str)
        else:
            if self.scale < 0:
                for p in self.m_pathlist:
                    flmax = max(p.flow(), flmax)
            else:
                flmax = self.scale

            for p in self.m_pathlist:
                flxratio = p.flow() / flmax if flmax > 0.0 else 0.0
                if self.m_local is not None and p.begin().number != self.m_local and p.end().number != self.m_local:
                    flxratio = 0.0
                if flxratio > self.threshold:
                    p.begin().visible = True
                    p.end().visible = True
                    edge_str = f"s{p.begin().number} -> s{p.end().number}[fontname=\"{self.m_font}\""
                    if self.arrow_width < 0:
                        import math
                        lwidth = 1.0 - 4.0 * math.log10(flxratio / self.threshold) / math.log10(self.threshold) + 1.0
                        edge_str += f", penwidth={lwidth:.3g}, arrowsize={min(6.0, 0.5 * lwidth):.3g}"
                    else:
                        edge_str += f", penwidth={self.arrow_width:g}, arrowsize={flxratio + 1:.3g}"

                    edge_str += f', color="0.7, {flxratio + 0.5:.3g}, 0.9"\n'
                    if flxratio > self.label_threshold:
                        edge_str += f', label=" {flxratio:.3g}"'
                    edge_str += "];"
                    lines.append(edge_str)

        for k in sorted(self.m_nodes.keys()):
            node = self.m_nodes[k]
            if node.visible:
                lines.append(f's{k} [ fontname="{self.m_font}", label="{node.name}"];')

        lines.append(f' label = "Scale = {flmax:.2g}\\l {self.title}";')
        lines.append(f' fontname = "{self.m_font}";')
        lines.append("}\n")
        res = "\n".join(lines)
        if s is not None:
            if hasattr(s, "write"):
                s.write(res)
            elif isinstance(s, (str, Path)):
                with open(s, "w", encoding="utf-8") as f:
                    f.write(res)
        return res

    def getDot(self) -> str:
        return self.exportToDot()

    def get_dot(self) -> str:
        return self.getDot()

    def write_dot(self, filename) -> None:
        self.exportToDot(filename)

    def findMajorPaths(self, threshold: float, lda: int, a) -> None:
        netmax = 0.0
        n_nodes = self.nNodes()
        for n in range(n_nodes):
            for m in range(n + 1, n_nodes):
                k1 = self.m_speciesNumber[n]
                k2 = self.m_speciesNumber[m]
                fl = abs(self.netFlow(k1, k2))
                netmax = max(fl, netmax)
        for n in range(n_nodes):
            for m in range(n + 1, n_nodes):
                k1 = self.m_speciesNumber[n]
                k2 = self.m_speciesNumber[m]
                fl = abs(self.netFlow(k1, k2))
                if netmax > 0.0 and fl > threshold * netmax:
                    a[lda * k1 + k2] = 1

    def add(self, other) -> None:
        for n in range(self.nPaths()):
            p = self.path(n)
            k1 = p.begin().number
            k2 = p.end().number
            p.setFlow(p.flow() + other.flow(k1, k2))


def newReactionPathDiagram(kin, element: str) -> ReactionPathDiagram:
    """Create a new reaction path diagram for element in kinetics manager."""
    return ReactionPathDiagram(kin, element)


from .surface import InterfaceKinetics

class EdgeKinetics(InterfaceKinetics):
    """Heterogeneous kinetics on 1D edge boundaries."""
    def kineticsType(self) -> str:
        return "Edge"


class Reaction:
    """Reaction class representing an elementary or composite chemical reaction.

    Replicates Cantera's Reaction in include/cantera/kinetics/Reaction.h
    """
    def __init__(self, equation="", rate=None, tbody=None, reactants=None, products=None, kin=None, node=None, orders=None, reversible=True, duplicate=False, **kwargs):
        self._reactants = dict(reactants or {})
        self._products = dict(products or {})
        self._orders = dict(orders or {})
        self._rate = rate
        self._third_body = tbody
        self.reversible = bool(reversible)
        self.duplicate = bool(duplicate)
        self.allow_nonreactant_orders = False
        self.allow_negative_orders = False
        self._valid = True
        self._callbacks = {}
        self.id = ""
        self.input = {}
        self.rate_units = {}

        if node is not None or isinstance(equation, dict):
            n = node if node is not None else equation
            self.setParameters(n, kin)
        elif equation:
            self.setEquation(str(equation), kin)
            if rate is not None:
                self.setRate(rate)
            if tbody is not None:
                self._third_body = tbody
        elif self._reactants or self._products:
            if self._rate is not None:
                self.setRate(self._rate)
            self.check()

    @property
    def reactants(self) -> dict:
        return self._reactants

    @reactants.setter
    def reactants(self, val: dict) -> None:
        self._reactants = dict(val)

    @property
    def products(self) -> dict:
        return self._products

    @products.setter
    def products(self, val: dict) -> None:
        self._products = dict(val)

    @property
    def orders(self) -> dict:
        return self._orders

    @orders.setter
    def orders(self, val: dict) -> None:
        self._orders = dict(val)

    def reactantString(self) -> str:
        res = []
        for sp, stoich in sorted(self._reactants.items()):
            if stoich != 1.0:
                res.append(f"{stoich:g} {sp}")
            else:
                res.append(sp)
        s = " + ".join(res)
        if self._third_body is not None:
            s += self._third_body.collider()
        return s

    def productString(self) -> str:
        res = []
        for sp, stoich in sorted(self._products.items()):
            if stoich != 1.0:
                res.append(f"{stoich:g} {sp}")
            else:
                res.append(sp)
        s = " + ".join(res)
        if self._third_body is not None:
            s += self._third_body.collider()
        return s

    def equation(self) -> str:
        arrow = " <=> " if self.reversible else " => "
        return self.reactantString() + arrow + self.productString()

    def setEquation(self, eq: str, kin=None) -> None:
        from .reaction_equation import parse_reaction_equation
        eq_str = str(eq).strip()
        if "<=>" in eq_str:
            self.reversible = True
        elif "=>" in eq_str:
            self.reversible = False
        elif "=" in eq_str:
            self.reversible = True

        r_comp, p_comp, tbody = parse_reaction_equation(eq_str)
        self._reactants = r_comp
        self._products = p_comp
        if tbody:
            self._third_body = tbody
        self.check()

    def type(self) -> str:
        if self._rate is None:
            raise ValueError("Reaction::type: Empty Reaction does not have a type")
        rtype = self._rate.type() if callable(getattr(self._rate, "type", None)) else getattr(self._rate, "type", "Arrhenius")
        subtype = self._rate.subType() if callable(getattr(self._rate, "subType", None)) else getattr(self._rate, "subType", "")
        if subtype:
            return f"{rtype}-{subtype}"
        if self._third_body is not None:
            return f"three-body-{rtype}"
        return str(rtype)

    def calculateRateCoeffUnits(self, kin=None) -> dict:
        u = {"units": "standardConcentrationUnits/s"}
        if self._rate is not None and hasattr(self._rate, "rate_units"):
            u["rate_units"] = self._rate.rate_units
        elif hasattr(self, "rate_units"):
            u["rate_units"] = self.rate_units
        return u

    def check(self) -> None:
        if not self.allow_nonreactant_orders:
            for name in self._orders:
                if name not in self._reactants:
                    raise ValueError(f"Reaction order specified for non-reactant species '{name}'")
        if not self.allow_negative_orders:
            for name, order in self._orders.items():
                if order < 0.0:
                    raise ValueError(f"Negative reaction order specified for species '{name}'")
        if self.reversible and self._orders:
            raise ValueError("Reaction orders may only be given for irreversible reactions")
        if self._rate is not None and hasattr(self._rate, "check"):
            self._rate.check(self.equation())

    def validate(self, kin=None) -> None:
        self.check()
        if kin is not None:
            self.checkSpecies(kin)

    def parameters(self, withInput: bool = True) -> dict:
        p = {"equation": self.equation(), "type": self.type()}
        if self.duplicate:
            p["duplicate"] = True
        if self._orders:
            p["orders"] = dict(self._orders)
        if self.allow_negative_orders:
            p["negative-orders"] = True
        if self.allow_nonreactant_orders:
            p["nonreactant-orders"] = True
        if self._rate is not None and hasattr(self._rate, "parameters"):
            p.update(self._rate.parameters())
        if self._third_body is not None and hasattr(self._third_body, "getParameters"):
            self._third_body.getParameters(p)
        if withInput and self.input:
            p.update(self.input)
        return p

    def getParameters(self, reactionNode: dict) -> None:
        params = self.parameters(withInput=False)
        reactionNode.update(params)

    def setParameters(self, node: dict, kin=None) -> None:
        if not node:
            raise ValueError("Cannot set reaction parameters from empty node.")
        self.input = dict(node)
        if "equation" in node:
            self.setEquation(node["equation"], kin)
        if "duplicate" in node:
            self.duplicate = bool(node["duplicate"])
        if "orders" in node:
            self._orders = dict(node["orders"])
        if "negative-orders" in node:
            self.allow_negative_orders = bool(node["negative-orders"])
        if "nonreactant-orders" in node:
            self.allow_nonreactant_orders = bool(node["nonreactant-orders"])
        if "id" in node:
            self.id = str(node["id"])
        if "rate-constant" in node or "rate" in node:
            from .rates import newReactionRate
            self._rate = newReactionRate(node)

    def valid(self) -> bool:
        return bool(self._valid)

    def setValid(self, valid: bool) -> None:
        self._valid = bool(valid)

    def checkBalance(self, kin) -> None:
        balr, balp = {}, {}
        for name, stoich in self._products.items():
            ph = kin.speciesPhase(name) if hasattr(kin, "speciesPhase") else kin
            k = ph.speciesIndex(name) if hasattr(ph, "speciesIndex") else ph.species_names.index(name)
            n_elem = ph.nElements() if hasattr(ph, "nElements") else len(ph.element_names)
            for m in range(n_elem):
                ename = ph.elementName(m) if hasattr(ph, "elementName") else ph.element_names[m]
                balr.setdefault(ename, 0.0)
                natoms = ph.nAtoms(k, m) if hasattr(ph, "nAtoms") and callable(ph.nAtoms) else ph.n_atoms(name, ename)
                balp[ename] = balp.get(ename, 0.0) + stoich * natoms

        for name, stoich in self._reactants.items():
            ph = kin.speciesPhase(name) if hasattr(kin, "speciesPhase") else kin
            k = ph.speciesIndex(name) if hasattr(ph, "speciesIndex") else ph.species_names.index(name)
            n_elem = ph.nElements() if hasattr(ph, "nElements") else len(ph.element_names)
            for m in range(n_elem):
                ename = ph.elementName(m) if hasattr(ph, "elementName") else ph.element_names[m]
                natoms = ph.nAtoms(k, m) if hasattr(ph, "nAtoms") and callable(ph.nAtoms) else ph.n_atoms(name, ename)
                balr[ename] = balr.get(ename, 0.0) + stoich * natoms

        ok = True
        msg = ""
        for elem, br in balr.items():
            bp = balp.get(elem, 0.0)
            scale = max(abs(br), abs(bp))
            diff = abs(bp - br)
            if diff > 1e-4 * scale:
                ok = False
                msg += f"  {elem}           {br}           {bp}\n"
        if not ok:
            raise ValueError(f"The following reaction is unbalanced: {self.equation()}\n  Element    Reactants    Products\n{msg}")

    def checkSpecies(self, kin) -> bool:
        skip = kin.skipUndeclaredSpecies() if hasattr(kin, "skipUndeclaredSpecies") else False
        undeclared = []
        for sp in list(self._reactants.keys()) + list(self._products.keys()) + list(self._orders.keys()):
            idx = kin.kineticsSpeciesIndex(sp, False) if hasattr(kin, "kineticsSpeciesIndex") else (kin.species_names.index(sp) if sp in kin.species_names else None)
            if idx is None or idx < 0:
                undeclared.append(sp)

        if undeclared:
            if skip:
                return False
            raise ValueError(f"Reaction '{self.equation()}' contains undeclared species: {undeclared}")

        self.checkBalance(kin)
        return True

    def usesElectrochemistry(self, kin=None) -> bool:
        if kin is None:
            return False
        n_phases = kin.nPhases() if hasattr(kin, "nPhases") else 1
        e_counter = [0.0] * n_phases
        for name, stoich in self._products.items():
            kkin = kin.kineticsSpeciesIndex(name) if hasattr(kin, "kineticsSpeciesIndex") else kin.species_names.index(name)
            iph = kin.speciesPhaseIndex(kkin) if hasattr(kin, "speciesPhaseIndex") else 0
            p = kin.thermo(iph) if hasattr(kin, "thermo") else kin
            kphase = p.speciesIndex(name) if hasattr(p, "speciesIndex") else p.species_names.index(name)
            chg = p.charge(kphase) if hasattr(p, "charge") else 0.0
            e_counter[iph] += stoich * chg
        for name, stoich in self._reactants.items():
            kkin = kin.kineticsSpeciesIndex(name) if hasattr(kin, "kineticsSpeciesIndex") else kin.species_names.index(name)
            iph = kin.speciesPhaseIndex(kkin) if hasattr(kin, "speciesPhaseIndex") else 0
            p = kin.thermo(iph) if hasattr(kin, "thermo") else kin
            kphase = p.speciesIndex(name) if hasattr(p, "speciesIndex") else p.species_names.index(name)
            chg = p.charge(kphase) if hasattr(p, "charge") else 0.0
            e_counter[iph] -= stoich * chg
        return any(abs(e) > 0.001 for e in e_counter)

    def rate(self):
        return self._rate

    def setRate(self, r) -> None:
        if r is None:
            raise ValueError("Reaction::setRate: Reaction rate must not be empty")
        self._rate = r
        for cb in list(self._callbacks.values()):
            cb()

    def thirdBody(self):
        return self._third_body

    def usesThirdBody(self) -> bool:
        return self._third_body is not None

    def registerSetRateCallback(self, callback_id, callback) -> None:
        self._callbacks[callback_id] = callback

    def removeSetRateCallback(self, callback_id) -> None:
        self._callbacks.pop(callback_id, None)


class KineticsFactory:
    """Factory for kinetics managers matching Cantera's KineticsFactory."""
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None
    def newKinetics(self, model: str = ""):
        m = model.lower().replace("-", "").replace("_", "")
        if m in ("", "none", "gas", "bulk", "idealgas", "bulkkinetics", "gaskinetics"):
            return GasKinetics()
        elif m in ("interface", "surface", "interfacekinetics"):
            from .surface import InterfaceKinetics
            return InterfaceKinetics()
        elif m in ("edge", "edgekinetics"):
            return EdgeKinetics()
        raise ValueError(f"Unknown kinetics model '{model}'")


def newKinetics(model_or_phases="", phaseNode=None, rootNode=None, soln=None) -> Kinetics:
    """Create a new Kinetics instance."""
    if isinstance(model_or_phases, str):
        return KineticsFactory.factory().newKinetics(model_or_phases)
    elif isinstance(model_or_phases, (list, tuple)):
        from .surface import InterfaceKinetics
        phases = list(model_or_phases)
        if len(phases) == 1 and phases[0].nDim() == 3:
            kin = GasKinetics()
            kin.addPhase(phases[0])
            return kin
        elif any(p.nDim() == 2 for p in phases):
            kin = InterfaceKinetics()
            for p in phases:
                kin.addPhase(p)
            return kin
        elif any(p.nDim() == 1 for p in phases):
            kin = EdgeKinetics()
            for p in phases:
                kin.addPhase(p)
            return kin
        else:
            kin = GasKinetics()
            for p in phases:
                kin.addPhase(p)
            return kin
    return GasKinetics()