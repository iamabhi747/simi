import numpy as np

class NeuralNetwork:
    def __init__(self, layers:list):
        for i in range(1, len(layers)):
            assert layers[i-1].output_dim == layers[i].input_dim
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x