import numpy as np
from .NonLinear import NonLinear

IDENTITY = NonLinear("IDENTITY", lambda x: x, lambda x: 1)
RELU     = NonLinear("RELU", lambda x: x if x >= 0 else 0, lambda x: 1 if x >= 0 else 0)
SIGMOID  = NonLinear("SIGMOID", lambda x: 1/(1+np.exp(-x)), lambda x: np.exp(-x)/(1+np.exp(-x))**2)
TANH     = NonLinear("TANH", lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2)

NONLINEAR_FUNCTIONS = {
    "IDENTITY": IDENTITY,
    "RELU": RELU,
    "SIGMOID": SIGMOID,
    "TANH": TANH
}