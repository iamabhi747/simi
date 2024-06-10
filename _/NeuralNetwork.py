import numpy as np
from .DenseLayer import DenseLayer
from .NonLinearFunctions import NONLINEAR_FUNCTIONS

class NeuralNetwork:
    def __init__(self, layers:list):
        for i in range(1, len(layers)):
            assert layers[i-1].output_dim == layers[i].input_dim
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dC_dA):
        for layer in reversed(self.layers):
            dC_dA = layer.backward(dC_dA)
        return dC_dA
    
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
    
    def save(self, filename:str):
        with open(filename, 'wb') as f:
            for layer in self.layers:
                f.write(f"{layer.type} {layer.nl.type} {layer.input_dim} {layer.output_dim}\n".encode('utf-8'))
                for p in layer.perceptrons:
                    a = f.write(p.W.tobytes())
                    b = f.write(p.b.tobytes())
    
    @staticmethod
    def load(filename:str):
        layers = []
        with open(filename, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8')
                if not line:
                    break
                parts = line.split()

                if parts[0] == "DENSE":
                    layer = DenseLayer(int(parts[2]), int(parts[3]), NONLINEAR_FUNCTIONS[parts[1]])
                    input_dim  = int(parts[2])
                    output_dim = int(parts[3])

                    for i in range(output_dim):
                        W = np.frombuffer(f.read(8*input_dim), dtype=np.float64)
                        b = np.frombuffer(f.read(8), dtype=np.float64)[0]
                        layer.perceptrons[i].W = W
                        layer.perceptrons[i].b = b
                else:
                    raise Exception(f"Unknown layer type: {parts[0]}")
                
                layers.append(layer)

        return NeuralNetwork(layers)