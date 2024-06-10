from .NonLinear import NonLinear
import math

def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid_function(x)
    return s * (1 - s)

SIGMOID = NonLinear("SIGMOID", sigmoid_function, sigmoid_derivative)