import numpy as np
from . import cost

class NeuralNetwork:
    def __init__(self, layers: list) -> None:
        for i in range(1, len(layers)):
            assert layers[i-1].output_dim == layers[i].input_dim
        self.layers = layers

        self.input_dim  = layers[0].input_dim
        self.output_dim = layers[-1].output_dim

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, X: np.ndarray, Y: np.ndarray, cf: cost.CostFunction) -> np.ndarray:
        Y_hat = self.forward(X)
        dC_dA = cf.derv(Y, Y_hat)

        for layer in reversed(self.layers):
            dC_dA = layer.backward(dC_dA)

        return dC_dA
    
    def update(self, lr: float) -> None:
        for layer in self.layers:
            layer.update(lr)

    def flush_cache(self) -> None:
        for layer in self.layers:
            layer.flush_cache()