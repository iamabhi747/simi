from .NonLinear import NonLinear

def identity_function(x):
    return x

def identity_derivative(x):
    return 1

IDENTITY = NonLinear("IDENTITY", identity_function, identity_derivative)