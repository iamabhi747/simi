import numpy as np
from .Perceptron import Perceptron

class DenseLayer:
    def __init__(self, input_dim:int, output_dim:int, nl=None):
        if nl is not None: assert callable(nl)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nl = nl if nl is not None else lambda x: x
        self.perceptrons = [Perceptron(input_dim, self.nl) for _ in range(output_dim)]

    def forward(self, x):
        return np.array([p.forward(x) for p in self.perceptrons])