import numpy as np

class Perceptron:
    def __init__(self, dim:int, nl=None) -> None:
        if nl is not None: assert callable(nl)
        self.W = np.random.rand(dim)
        self.b = np.random.rand(1)[0]
        self.dim = dim
        self.nl = nl if nl is not None else lambda x: x

    def forward(self, x:np.ndarray) -> float:
        return self.nl(np.dot(self.W, x) + self.b)