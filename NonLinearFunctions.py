import numpy as np
from .NonLinear import NonLinear

IDENTITY = NonLinear("IDENTITY", lambda x: x)
RELU     = NonLinear("RELU", lambda x: x if x >= 0 else 0)
SIGMOID  = NonLinear("SIGMOID", lambda x: 1/(1+np.exp(-x)))
TANH     = NonLinear("TANH", lambda x: np.tanh(x))

NONLINEAR_FUNCTIONS = {
    "RELU": RELU,
    "SIGMOID": SIGMOID,
    "TANH": TANH,
}