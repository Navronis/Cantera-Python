with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    c = f.read()

old_block = """        self.bold_threshold = 0.2
        self.normal_threshold = 0.05
        self.label_threshold = 0.001
        self.threshold = 0.005"""

new_block = """        self.bold_threshold = 0.2
        self.m_bold_threshold = 0.2
        self.normal_threshold = 0.05
        self.label_threshold = 0.001
        self.m_threshold = 0.005"""

old_methods = """    def setThreshold(self, th: float):
        self.threshold = float(th)

    def threshold(self) -> float:
        return float(self.threshold) if hasattr(self, "threshold") else 0.005

    def setBoldThreshold(self, th: float):
        self.bold_threshold = float(th)

    def boldThreshold(self) -> float:
        return float(self.bold_threshold) if hasattr(self, "bold_threshold") else 0.2"""

new_methods = """    class _CallableFloat(float):
        def __call__(self):
            return float(self)

    @property
    def threshold(self):
        return self._CallableFloat(self.m_threshold)

    @threshold.setter
    def threshold(self, val):
        self.m_threshold = float(val)

    def setThreshold(self, th: float):
        self.m_threshold = float(th)

    def setBoldThreshold(self, th: float):
        self.m_bold_threshold = float(th)
        self.bold_threshold = float(th)

    def boldThreshold(self) -> float:
        return float(self.m_bold_threshold)"""

assert old_block in c, "old_block not found"
assert old_methods in c, "old_methods not found"
c = c.replace(old_block, new_block).replace(old_methods, new_methods)
with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("ReactionPathDiagram updated.")
