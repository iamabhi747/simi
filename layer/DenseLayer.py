import numpy as np
from .. import nonlinear

class DenseLayer:
    def __init__(self, input_dim:int, output_dim:int, nl=None):
        self._name = 'DenseLayer'
        self.input_dim  = input_dim
        self.output_dim = output_dim

        ## Weights [input_dim, output_dim]
        #  [ [] * output_dim ] * input_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.1

        ## Biases
        self.b = np.random.randn(output_dim) * 0.1

        ## Non-linear function
        self.nl = nl if nl is not None and isinstance(nl, nonlinear.NonLinear) else nonlinear.IDENTITY

        ## Store activation for backpropagation
        self.list_prv_a = []
        self.list_z     = []

        ## Store gradients for update
        self.dC_dW = np.zeros(self.W.shape)
        self.dC_dB = np.zeros(self.b.shape)
        self.count = 0

    def to_bytes(self, byteorder='big'):
        out = b''
        out += len(self._name).to_bytes(4, byteorder)
        out += self._name.encode('utf-8')
        out += self.input_dim.to_bytes(4, byteorder)
        out += self.output_dim.to_bytes(4, byteorder)
        out += len(self.nl.name).to_bytes(4, byteorder)
        out += self.nl.name.encode('utf-8')
        return out

    ## Forward pass
    # X: ndarray [input_dim, 1] (column vector)
    def forward(self, X:np.ndarray):
        assert X.shape == (self.input_dim,)                  , "Input shape should be ({},) got {}".format(self.input_dim, X.shape)

        ## Z [output_dim,]
        #     Input    *         Weights         +    Biases
        # [input_dim,] * [input_dim, output_dim] + [output_dim,]
        z = np.dot(X, self.W) + self.b

        ## Activation [output_dim,]
        # Non-linearity
        a = self.nl(z)

        ## Store activations for backpropagation
        self.list_prv_a.append(X)
        self.list_z    .append(z)

        ## Return activation
        return a
    
    ## Backward pass
    # dC_dA: ndarray [output_dim,]
    def backward(self, dC_dA:np.ndarray):
        ### Theory

        ## Chain rule
        #  dC_dW  = dC_dA * dA_dZ * dZ_dW
        #  dC_dB  = dC_dA * dA_dZ * dZ_dB
        #  dC_dAp = dC_dA * dA_dZ * dZ_dAp

        ## EQN 1
        #       A       =     NL(Z)
        # [output_dim,] = [output_dim,]

        ## EQN 2
        #       Z       =     prv_A    *            W            +       B
        # [output_dim,] = [input_dim,] * [input_dim, output_dim] + [output_dim,]

        #     dA_dZ     =  derv(NL)(Z)       (by EQN 1)
        # [output_dim,] = [output_dim,]

        #    dZ_dW     =    prv_A
        # [input_dim,] = [input_dim,]

        #         dZ_dAp          =            W
        # [input_dim, output_dim] = [input_dim, output_dim]

        #    dZ_dB     =      1

        ## dC_dW [input_dim, output_dim]
        # Represents the gradient of the cost function w.r.t the weights
        #
        #          dC_dW          =      dZ_dW     * [dC_dA . dA_dZ]
        # [input_dim, output_dim] = [input_dim, 1] *  [output_dim,]

        ## dC_dB [output_dim,]
        # Represents the gradient of the cost function w.r.t the biases
        #
        #          dC_dB          =      dZ_dB     * [dC_dA . dA_dZ]
        #       [output_dim,]     =       [1]      *  [output_dim,]

        ## dC_dAp [input_dim, 1]
        # Represents the gradient of the cost function w.r.t the previous layer's activation
        #
        #     dC_dAp     =         dZ_dAp          *  [dC_dA . dA_dZ].reshape
        # [input_dim, 1] = [input_dim, output_dim] *  [output_dim, 1]
        #
        # reshape dC_dAp: [input_dim, 1] -> [input_dim,]

        prv_a = self.list_prv_a.pop()
        z     = self.list_z    .pop()

        dA_dZ = self.nl.derv(z)
        dZ_dW = prv_a.reshape(-1, 1)

        p1 = dC_dA * dA_dZ

        dC_dW  = np.dot(dZ_dW, np.array([p1]))
        dC_dB  = p1
        dC_dAp = np.dot(self.W, p1.reshape(-1, 1)).reshape(-1)

        ## Store gradients for update
        self.dC_dW = self.dC_dW + dC_dW
        self.dC_dB = self.dC_dB + dC_dB
        self.count += 1

        ## Return gradient w.r.t previous layer's activation
        return dC_dAp
    
    ## Update weights and biases
    # lr: float
    def update(self, lr:float):
        if self.count == 0: return

        self.W = self.W + lr * self.dC_dW / self.count
        self.b = self.b + lr * self.dC_dB / self.count

        self.dC_dW = np.zeros(self.W.shape)
        self.dC_dB = np.zeros(self.b.shape)
        self.count = 0

    ## Flush cache
    def flush_cache(self):
        self.list_prv_a = []
        self.list_z     = []