import numpy as np
from .NonLinear import NonLinear
from .NonLinearFunctions import IDENTITY

class Perceptron:
    def __init__(self, dim:int, nl=None) -> None:
        if nl is not None: assert isinstance(nl, NonLinear)
        self.W = np.random.rand(dim)
        self.b = np.random.rand(1)[0]
        self.dim = dim
        self.nl = nl if nl is not None else IDENTITY
        self.prev_layer_nl_activations = None
        self.prev_my_activation = 0.0
        self.prev_my_nl_activation = 0.0
        self.camulative_dC_dW = np.zeros(dim)
        self.camulative_dC_dB = 0.0
        self.camulative_count = 0

    def forward(self, x:np.ndarray) -> float:
        self.prev_activations   = x
        self.prev_my_activation = np.dot(self.W, x) + self.b
        self.prev_my_nl_activation = self.nl.call(self.prev_my_activation)
        return self.prev_my_nl_activation
    
    def backward(self, dC_dA:float) -> np.ndarray:
        dA_dZ = self.nl.derivative(self.prev_my_activation)
        dZ_dW = self.prev_activations
        dZ_dB = 1

        dC_dZ = dC_dA * dA_dZ

        dC_dW = dZ_dW * dC_dZ
        dC_dB = dZ_dB * dC_dZ
        dC_dX = self.W * dC_dZ

        self.camulative_dC_dW += dC_dW
        self.camulative_dC_dB += dC_dB
        self.camulative_count += 1
        return dC_dX
    
    def update(self, lr:float) -> None:
        self.W -= lr * self.camulative_dC_dW / self.camulative_count
        self.b -= lr * self.camulative_dC_dB / self.camulative_count
        self.camulative_dC_dW = np.zeros(self.dim)
        self.camulative_dC_dB = 0.0
        self.camulative_count = 0