from .NonLinear import NonLinear
import numpy as np

def relu_function(x):
    if isinstance(x, np.ndarray):
        return np.array([max(0, i) for i in x])
    return max(0, x)

def relu_derivative(x):
    if isinstance(x, np.ndarray):
        return np.array([1 if i > 0 else 0 for i in x])
    return 1 if x > 0 else 0

RELU = NonLinear("RELU", relu_function, relu_derivative)